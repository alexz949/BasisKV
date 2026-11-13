"""Capture and fit Qwen3.5 gated Values and Gated-DeltaNet output projections."""
import argparse
from pathlib import Path
import torch

from calibration.common import add_model_arguments, add_window_arguments, load_teacher, read_windows, setup
from core.gated_value import GatedVCapture, GatedVBlock, fit_gated_v
from core.als import damp_covariance, quadratic_from_target, initialize_weighted_svd, fit_joint
from core.router_io import save_tensors, write_json, read_json
from safetensors.torch import load_file


@torch.inference_mode()
def capture(args):
    windows, splits = read_windows(args)
    model, config, geometry, attention, _ = load_teacher(args)
    assert geometry.model_type == 'qwen3_5'
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    rows, handles, grams, counts, active = {}, [], {}, {}, {}
    def capture_attention(module, query, key, value, attention_mask, **kwargs):
        output, weights = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        if module.layer_idx in rows:
            rows[module.layer_idx]['z'] = output.detach().cpu().flatten(0, 1).contiguous()
        return output, weights
    name = 'basiskv_gated_capture'
    ALL_ATTENTION_FUNCTIONS.register(name, capture_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
    model.config._attn_implementation = name
    for layer, module in attention.items():
        rows[layer] = {}
        weight = module.o_proj.weight.detach().T.reshape(config.num_attention_heads, module.head_dim, -1).cpu().contiguous()
        save_tensors(args.output / f'weights_{layer:03d}.safetensors', {'weight': weight})
        def gate_hook(module, inputs, output, layer=layer, dim=module.head_dim):
            rows[layer]['gate'] = output.view(*output.shape[:-1], -1, 2 * dim).chunk(2, -1)[1].sigmoid().detach().cpu().flatten(0, 1).contiguous()
        def output_hook(module, inputs, output, layer=layer):
            target = output if module.bias is None else output - module.bias
            rows[layer]['target'] = target.detach().cpu().flatten(0, 1).contiguous()
        handles.extend([module.q_proj.register_forward_hook(gate_hook), module.o_proj.register_forward_hook(output_hook)])
    gdn = {i: block.linear_attn.out_proj for i, block in enumerate(model.model.layers) if hasattr(block, 'linear_attn')}
    for layer, module in gdn.items():
        grams[layer] = {}
        def gram_hook(module, inputs, layer=layer):
            split = active['split']
            x = inputs[0].flatten(0, 1).double()
            if split not in grams[layer]:
                grams[layer][split] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
            grams[layer][split].addmm_(x.T, x)
        handles.append(module.register_forward_pre_hook(gram_hook))
    for split, ids in splits.items():
        active['split'] = split
        counts[split] = len(ids) * args.sequence_length
        for index in ids:
            model.model(windows[index:index+1].to(model.get_input_embeddings().weight.device, torch.long), use_cache=False)
            for layer, data in rows.items():
                save_tensors(args.output / f'l{layer:03d}/{split}_{index:03d}.safetensors', data)
        for layer in gdn:
            grams[layer][split] = grams[layer][split].cpu()
    for handle in handles:
        handle.remove()
    for layer, module in gdn.items():
        save_tensors(args.output / f'gdn_{layer:03d}.safetensors',
            dict(weight=module.weight.detach().cpu().contiguous(),
                 **{split: value / counts[split] for split, value in grams[layer].items()}))
    write_json(args.output / 'manifest.json', dict(status='complete', geometry=geometry.to_dict(),
        attention_layers=list(attention), gdn_layers=list(gdn), splits=splits, sequence_length=args.sequence_length))


def read_capture(root, split, layer, mapping):
    files = sorted((root / f'l{layer:03d}').glob(f'{split}_*.safetensors'))
    assert files
    records = [load_file(str(path)) for path in files]
    weight = load_file(str(root / f'weights_{layer:03d}.safetensors'))['weight']
    data = {key: torch.cat([record[key] for record in records]) for key in ('z','gate','target')}
    return GatedVCapture(data['z'],data['gate'],weight,data['target'],mapping)


@torch.no_grad()
def fit(args):
    manifest = read_json(args.statistics / 'manifest.json')
    geometry = manifest['geometry']
    mapping = torch.arange(geometry['num_heads']) // (geometry['num_heads'] // geometry['num_kv_heads'])
    layers = manifest['attention_layers'] if args.layers == 'all' else list(map(int,args.layers.split(',')))
    args.output.mkdir(parents=True, exist_ok=True)
    bank = {}
    for layer in layers:
        train = read_capture(args.statistics, 'fit', layer, mapping)
        heldout = read_capture(args.statistics, 'heldout', layer, mapping)
        if args.rank == geometry['head_dim']:
            eye = torch.eye(args.rank).expand(geometry['num_kv_heads'], -1, -1).contiguous()
            result = dict(E_V=eye, R_V=eye, history=[])
        else:
            result = fit_gated_v(train,args.rank,heldout=heldout,encoder_sweeps=args.sweeps,device=args.device,
                chunk_rows=args.chunk_rows,linear_max_iter=args.cg_iterations,encoder_preconditioner='separable')
        bank[layer] = {key:result[key].cpu().to(torch.bfloat16) for key in ('E_V','R_V')}
        op = GatedVBlock(heldout, bank[layer]['E_V'].to(args.device).float(), block='decoder', chunk_rows=args.chunk_rows)
        error = 2 * len(heldout.z) * op.loss(bank[layer]['R_V'].to(args.device).float()) / float(heldout.target.double().square().sum())
        write_json(args.output / f'layer_{layer:03d}.json',dict(history=result['history'],rank=args.rank,heldout_relative_mse=error))
    torch.save(dict(format='basiskv.qwen35.gated_v_als.v1', layers=bank), args.output / 'value.pt')
    records = []
    for layer in manifest['gdn_layers']:
        payload = load_file(str(args.statistics / f'gdn_{layer:03d}.safetensors'))
        weight = payload['weight'].float().to(args.device)
        width = weight.shape[1]
        assert width % args.tp_sources == 0
        dim = width // args.tp_sources
        rank = int(dim * .75)
        covariance = payload['fit'].to(args.device).reshape(args.tp_sources,dim,args.tp_sources,dim).permute(0,2,1,3)
        covariance, _ = damp_covariance(covariance, 1e-7)
        target = weight.T.reshape(args.tp_sources, dim, -1)
        mapping_gdn = torch.arange(args.tp_sources,device=args.device)
        initial, _ = initialize_weighted_svd(covariance.float(),target,mapping_gdn,rank)
        encoder, decoder, _ = fit_joint(quadratic_from_target(covariance.float(),target),initial,mapping_gdn,
            sweeps=args.sweeps,cg_iterations=args.cg_iterations,work_dtype=torch.float32)
        records.append(dict(layer_index=layer,private_encoders=encoder.cpu().to(torch.bfloat16),
            joint_decoder_weight=decoder.flatten(0,1).T.cpu().to(torch.bfloat16)))
    torch.save(dict(format='basiskv.qwen35.gdn_private_ag_joint_factors.v2', schema_version=1,layers=records),args.output / 'gdn.pt')


def main():
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage',required=True)
    c=stages.add_parser('capture');add_model_arguments(c);add_window_arguments(c);c.add_argument('--output',type=Path,required=True)
    f=stages.add_parser('fit');f.add_argument('--statistics',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--rank',type=int,required=True);f.add_argument('--sweeps',type=int,default=12)
    f.add_argument('--cg-iterations',type=int,default=200);f.add_argument('--chunk-rows',type=int,default=256)
    f.add_argument('--tp-sources',type=int,default=8);f.add_argument('--layers',default='all');f.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    (capture if args.stage=='capture' else fit)(args)


if __name__ == '__main__':
    main()
