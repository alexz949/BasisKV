"""Tables 12 and 13: Value compression and routing quality with quantized KV caches.

Quantize-dequantize simulations on the compressed model. `nuq4` is KVQuant's
4-bit non-uniform quantization with dense-and-sparse outliers: Keys pre-RoPE
after the Key norm with static per-channel thresholds, Value latents per token
across the KV heads with dynamic thresholds. `fp8` is E4M3 with one scale per
token and KV head for Keys and one scale per routing page and KV head for Value
latents. Both act on the module outputs that feed the cache, so attention and
the Value-to-Key base see the dequantized values; residual codes stay BF16.

    python -m appendix.kv_quantization calibrate --model MODEL --value-factors FACTORS \\
        --kvquant external/KVQuant --output QUANTIZERS
    python -m appendix.kv_quantization ppl --format nuq4 --quantizers QUANTIZERS -- \\
        --model MODEL --value-factors FACTORS --c4-windows C4/windows.safetensors --output OUT
    python -m appendix.kv_quantization longbench --format fp8 -- \\
        --model MODEL --value-factors FACTORS --method basiskv --router ROUTER --data DATA --output OUT

`calibrate` fits the NUQ4 codebooks with KVQuant's own simulated quantizer
(SqueezeAILab/KVQuant, file quant/kvquant/simquant_module_quantizer.py) from
Fisher-weighted WikiText-2 training segments; `fp8` needs no calibration.
Arguments after `--` go to `eval.ppl` or `eval.longbench` unchanged.
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from core.router_io import read_json, write_json
from inference.model import attention_layers, install_factors, load_model

log = logging.getLogger("basiskv.kv_quantization")
E4M3_MAX = 448.0
BITS, SPARSITY = 4, 0.99


def load_kvquant(root):
    path = Path(root).expanduser().resolve() / "quant" / "kvquant" / "simquant_module_quantizer.py"
    spec = importlib.util.spec_from_file_location("kvquant_simquant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantization_points(model):
    """{name: (module, groups, width)} for the K and V outputs that feed every layer's cache."""
    from inference.attention import ValueAttention
    from inference.model import install_value_factors
    if not all(isinstance(module, ValueAttention) for module in attention_layers(model).values()):
        install_value_factors(model, None)
    points = {}
    for layer, attention in attention_layers(model).items():
        points[f"{layer}.k"] = (attention.k_norm, attention.groups, attention.head_dim)
        points[f"{layer}.v"] = (attention.v_proj, attention.groups, attention.rank)
    return points


def rows(output, groups, width):
    """Module output as [tokens, groups * width] rows."""
    return output.reshape(-1, groups * width)


def nuq4_roundtrip(upstream, quantizer, name, output, groups, width):
    flat = rows(output, groups, width).float()
    hi, lo, lut = quantizer
    hi, lo = hi.flatten().to(flat.device), lo.flatten().to(flat.device)
    if name.endswith(".v"):
        mask = upstream.get_outliers_dynamic(flat, channel=-1, thresh=SPARSITY)
        restored = upstream.quant_fn_nuq_recon(flat, bits=BITS, qchannel=-1, dynamicquantization=True, include_sparse=True,
                                               outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    else:
        mask = upstream.get_outliers(flat, channel=0, outlier_threshold_upper=hi, outlier_threshold_lower=lo)
        restored = upstream.quant_fn_nuq_recon(flat, bits=BITS, qchannel=0, dynamicquantization=False, include_sparse=True,
                                               outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    assert torch.isfinite(restored).all(), f"non-finite NUQ4 reconstruction at {name}"
    return restored.reshape(output.shape).to(output.dtype)


def e4m3_roundtrip(value, dims):
    """E4M3 round trip with one absmax scale over `dims`."""
    work = value.float()
    scale = (work.abs().amax(dim=dims, keepdim=True) / E4M3_MAX).clamp_min(1e-30)
    return ((work / scale).to(torch.float8_e4m3fn).float() * scale).to(value.dtype)


def fp8_roundtrip(name, output, groups, width, page_size):
    batch, tokens = output.shape[0], output.shape[1]
    cells = output.reshape(batch, tokens, groups, width)
    if name.endswith(".k"):  # one scale per token and KV head
        return e4m3_roundtrip(cells, (-1,)).reshape(output.shape)
    pages = -(-tokens // page_size)  # one scale per routing page and KV head
    padded = F.pad(cells, (0, 0, 0, 0, 0, pages * page_size - tokens)).reshape(batch, pages, page_size, groups, width)
    restored = e4m3_roundtrip(padded.transpose(2, 3), (-2, -1)).transpose(2, 3)
    return restored.reshape(batch, pages * page_size, groups, width)[:, :tokens].reshape(output.shape)


def install_hooks(model, fmt, *, quantizers=None, upstream=None, page_size=4):
    handles = []
    for name, (module, groups, width) in quantization_points(model).items():
        if fmt == "nuq4":
            hook = lambda m, x, y, name=name, g=groups, w=width: nuq4_roundtrip(upstream, quantizers[name], name, y, g, w)
        elif name.endswith('.k'):
            layer = int(name.split('.')[0])
            module = attention_layers(model)[layer].key_cache_transform
            hook = lambda m, x, y: e4m3_roundtrip(y, (-1,))
        else:
            hook = lambda m, x, y, name=name, g=groups, w=width: fp8_roundtrip(name, y, g, w, page_size)
        handles.append(module.register_forward_hook(hook))
    return handles


def calibrate(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    upstream = load_kvquant(args.kvquant)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args.model, dtype=args.dtype, device=args.device)
    install_factors(model, value_factors=args.value_factors)
    points = quantization_points(model)
    text = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = torch.tensor(tokenizer("\n\n".join(text["text"]), add_special_tokens=False)["input_ids"])
    generator = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, ids.numel() - args.segment_length, (args.segments,), generator=generator).tolist()
    records = {name: [] for name in points}

    def capture(name, groups, width, output):
        entry = [rows(output.detach(), groups, width).float().cpu(), None]
        records[name].append(entry)
        output.register_hook(lambda grad: entry.__setitem__(1, rows(grad, groups, width).float().square().cpu()))

    handles = [module.register_forward_hook(lambda m, x, y, name=name, g=groups, w=width: capture(name, g, w, y))
               for name, (module, groups, width) in points.items()]
    device = model.get_input_embeddings().weight.device
    for index, start in enumerate(starts):
        tokens = ids[start : start + args.segment_length][None].to(device)
        embedded = model.get_input_embeddings()(tokens).detach().requires_grad_(True)
        logits = model(inputs_embeds=embedded, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum")
        loss.backward()
        log.info("segment %d/%d: loss %.2f", index + 1, args.segments, float(loss))
        del embedded, logits, loss
    for handle in handles:
        handle.remove()
    quantizers = {}
    for name in points:
        data = torch.cat([entry[0] for entry in records[name]])
        fisher = torch.cat([entry[1] for entry in records[name]])
        assert torch.isfinite(data).all() and torch.isfinite(fisher).all() and fisher.sum() > 0, f"bad statistics at {name}"
        fitter = SimpleNamespace(out=data, perchannel=True, qchannel=-1 if name.endswith(".v") else 0, bits=BITS, nsamples=args.segments)
        quantizers[name] = upstream.SimQuant.quantize(fitter, include_sparse=True, sparsity_threshold=SPARSITY, nuq=True,
                                                     fisher=fisher / fisher.mean(), first_few_fp16=-1)
        records[name] = None
        log.info("codebook %s", name)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(quantizers, args.output / "quantizers.pt")
    write_json(args.output / "manifest.json", dict(
        status="complete", model=str(args.model), value_factors=str(args.value_factors), bits=BITS, sparsity_threshold=SPARSITY,
        keys="k_norm output (pre-RoPE); static per-channel NUQ4 with calibrated outlier thresholds",
        values="v_proj output (Value latent); dynamic per-token NUQ4 across KV heads with 1% outliers",
        fisher="squared gradient of the summed next-token cross entropy, normalized by its mean per tensor",
        calibration=dict(dataset="Salesforce/wikitext wikitext-2-raw-v1 train", segments=args.segments,
                         segment_length=args.segment_length, seed=args.seed, starts=starts)))
    log.info("wrote %s", args.output / "quantizers.pt")


def evaluate(args, rest):
    import eval.common
    import eval.longbench
    import eval.ppl

    quantizers = upstream = None
    if args.format == "nuq4":
        assert read_json(args.quantizers / "manifest.json")["status"] == "complete", "incomplete quantizers"
        quantizers = torch.load(args.quantizers / "quantizers.pt", map_location="cpu", weights_only=False)
        upstream = load_kvquant(args.kvquant)
    original = eval.common.load_model

    def quantized(settings):
        model, tokenizer = original(settings)
        install_hooks(model, args.format, quantizers=quantizers, upstream=upstream, page_size=args.page_size)
        log.info("installed %s quantize-dequantize hooks on %d layers", args.format, len(attention_layers(model)))
        return model, tokenizer

    eval.common.load_model = eval.ppl.load_model = quantized
    runner = eval.ppl if args.command == "ppl" else eval.longbench
    runner.main(rest)
    if "--output" in rest:
        output = Path(rest[rest.index("--output") + 1])
        write_json(output / "quantization.json", dict(
            format=args.format, page_size=args.page_size if args.format == "fp8" else None,
            quantizers=str(args.quantizers) if args.quantizers else None,
            keys="cache input, post-RoPE" if args.format == "fp8" else "k_norm output, pre-RoPE", values="v_proj output, Value latent"))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cal = commands.add_parser("calibrate", help="fit NUQ4 codebooks with KVQuant")
    cal.add_argument("--model", type=Path, required=True)
    cal.add_argument("--value-factors", type=Path, required=True)
    cal.add_argument("--kvquant", type=Path, default=Path("external/KVQuant"), help="KVQuant repository checkout")
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--segments", type=int, default=16)
    cal.add_argument("--segment-length", type=int, default=2048)
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    cal.add_argument("--device", default="cuda:0")
    for name, text in (("ppl", "Table 12: eval.ppl with a quantized cache"), ("longbench", "Table 13: eval.longbench with a quantized cache")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--format", choices=("nuq4", "fp8"), required=True)
        sub.add_argument("--quantizers", type=Path, help="calibrate output (nuq4)")
        sub.add_argument("--kvquant", type=Path, default=Path("external/KVQuant"))
        sub.add_argument("--page-size", type=int, default=4, help="fp8 Value scale group; the routing page size")
        sub.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the evaluator after --")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        calibrate(args)
    else:
        evaluate(args, args.rest[1:] if args.rest[:1] == ["--"] else args.rest)


if __name__ == "__main__":
    main()
