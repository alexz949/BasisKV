"""Archived uniform/KL schedules used by the paper's Qwen3 generation appendix."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from core.factor_io import sha256
from core.router_io import read_json
from inference.tp import folded_projection_weights, head_slices


class BasisKVAppendixFoldedQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        config = vllm_config.model_config.hf_config
        group = get_tp_group()
        assert get_pp_group().world_size == 1
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        self.factor_root = Path(config.basisserve_c1_checkpoint_dir)
        path = self.factor_root / 'manifest.json'
        assert sha256(path) == config.basisserve_c1_manifest_sha256
        self.factor_manifest = read_json(path)
        manifest = self.factor_manifest
        assert manifest['format'] in ('basisserve.qwen3_8b.iclr_v_factors.v1', 'basisserve.qwen3_32b.iclr_v_factors.v1')
        assert manifest['status'] == 'complete'
        compression = manifest['compression']
        assert compression['method'] in ('c1-two-sided-kl', 'c1-uniform')
        assert compression['projection'] == 'joint_v_o'
        assert len(manifest['layers']) == config.num_hidden_layers
        self.factor_geometry = dict(num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads)
        head_slices(self.factor_geometry, group.rank_in_group, group.world_size)
        expected = config.num_hidden_layers * config.num_key_value_heads * compression['equivalent_rank_target']
        assert sum(sum(row['ranks']) for row in manifest['layers']) == expected
        artifact = manifest['artifact']
        assert sha256(self.factor_root / artifact['file']) == artifact['sha256']
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @torch.no_grad()
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        group = get_tp_group()
        groups, heads = head_slices(self.factor_geometry, group.rank_in_group, group.world_size)
        for index, layer in enumerate(self.model.layers):
            record = self.factor_manifest['layers'][index]
            assert record['layer'] == index
            path = self.factor_root / record['file']
            assert sha256(path) == record['sha256']
            payload = load_file(str(path))
            ranks = torch.tensor(record['ranks'])
            if 'source_ranks' in payload:
                assert torch.equal(payload['source_ranks'], ranks)
            encoder = payload['value_coordinate_encoders'][groups].clone()
            decoder = payload['head_output_decoders'][heads].clone()
            attention = layer.self_attn
            local_ranks = ranks[groups]
            maximum = encoder.shape[-1]
            assert decoder.shape[1] == maximum
            assert bool(((local_ranks > 0) & (local_ranks <= attention.head_dim)).all())
            valid = torch.arange(maximum)[None, :] < local_ranks[:, None]
            encoder *= valid[:, None, :]
            decoder *= valid.repeat_interleave(attention.num_heads // attention.num_kv_heads, dim=0)[:, :, None]
            offset = (attention.num_heads + attention.num_kv_heads) * attention.head_dim
            value = attention.qkv_proj.weight[offset:]
            folded_v, folded_o = folded_projection_weights(value, encoder, decoder)
            value.copy_(folded_v)
            attention.o_proj.weight.copy_(folded_o)
        return loaded
