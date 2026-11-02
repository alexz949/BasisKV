"""Measure the single-layer rank probes that drive core.allocate.

The dense model is the teacher. The anchor model installs the uniform anchor-rank
factors in every attention layer. For each layer l, the anchor model is modified
in that layer alone, once to the low and once to the high probe rank, and the
full-vocabulary KL(teacher || model) averaged over every prediction position of
the profile windows is recorded relative to the anchor model:

    D-_l = KL(M_{l <- r_low}) - KL(M_anchor),   D+_l = KL(M_{l <- r_high}) - KL(M_anchor).

    python -m calibration.probe_kl --model MODEL --windows WINDOWS.safetensors \\
        --banks 32=B32 64=B64 96=B96 --anchor-rank 64 --low-rank 32 --high-rank 96 --output PROBES

Layers can be split across workers with --layer-shard-index/--layer-shard-count;
run the same command with --merge afterwards.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from core.allocate import PROBES_FORMAT, parse_banks
from core.factor_io import DECODER_KEY, ENCODER_KEY, read_layer, read_results
from core.router_io import read_json, sha256, write_json
from inference.attention import ValueAttention
from inference.model import attention_layers, load_model

log = logging.getLogger("basiskv.probe_kl")
CHUNK = 256  # positions per softmax chunk; a full 2048 x 150k-vocabulary FP32 log-softmax is 1.2 GB per copy


def bank_modules(model, banks):
    """{rank: {layer: ValueAttention}} folded from the uniform banks; the natives are the teacher."""
    natives = attention_layers(model)
    modules = {}
    for rank, directory in banks.items():
        results = read_results(directory)
        modules[rank] = {}
        for layer in natives:
            record = read_layer(directory, layer, results["fit_config"])
            assert record is not None, f"layer {layer} of {directory} is missing or corrupt"
            tensors = load_file(str(directory / record["artifact"]["file"]))
            modules[rank][layer] = ValueAttention(natives[layer], tensors[ENCODER_KEY], tensors[DECODER_KEY])
    return natives, modules


def set_layer(model, layer, module):
    model.model.layers[layer].self_attn = module


@torch.no_grad()
def log_probabilities(model, input_ids):
    """FP32 log-probabilities for positions 1..T-1, taken CHUNK positions at a time to bound the softmax memory."""
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[1], CHUNK):
        out[:, start : start + CHUNK] = F.log_softmax(logits[:, start : start + CHUNK].float(), dim=-1)
    return out


@torch.no_grad()
def kl_to_teacher(teacher, model, input_ids):
    """Sum over positions of KL(teacher || model) in FP64, and the position count."""
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
    total = 0.0
    for start in range(0, logits.shape[1], CHUNK):
        student = F.log_softmax(logits[:, start : start + CHUNK].float(), dim=-1)
        reference = teacher[:, start : start + CHUNK]
        total += float((reference.exp() * (reference - student)).sum(dim=-1).double().sum())
    return total, teacher.shape[0] * teacher.shape[1]


def measure(model, windows, natives, modules, *, anchor, low, high, layers, batch_size):
    """KL sums over all windows for the anchor model and every layer probe."""
    sums = {"anchor": 0.0, **{(layer, rank): 0.0 for layer in layers for rank in (low, high)}}
    positions = 0
    for start in range(0, len(windows), batch_size):
        input_ids = windows[start : start + batch_size].to(next(model.parameters()).device).long()
        for layer, native in natives.items():
            set_layer(model, layer, native)
        teacher = log_probabilities(model, input_ids)
        for layer in natives:
            set_layer(model, layer, modules[anchor][layer])
        total, count = kl_to_teacher(teacher, model, input_ids)
        sums["anchor"] += total
        positions += count
        for layer in layers:
            for rank in (low, high):
                set_layer(model, layer, modules[rank][layer])
                sums[(layer, rank)] += kl_to_teacher(teacher, model, input_ids)[0]
            set_layer(model, layer, modules[anchor][layer])
        log.info("windows %d/%d", min(start + batch_size, len(windows)), len(windows))
    anchor_kl = sums["anchor"] / positions
    rows = [
        {
            "layer": layer,
            "low_kl": sums[(layer, low)] / positions,
            "high_kl": sums[(layer, high)] / positions,
            "delta_low": sums[(layer, low)] / positions - anchor_kl,
            "delta_high": sums[(layer, high)] / positions - anchor_kl,
            "anchor_kl": anchor_kl,
        }
        for layer in layers
    ]
    return anchor_kl, positions, rows


def shard_path(output, index, count):
    return output / ("probes.json" if count == 1 else f"probes_{index:02d}.json")


def merge(args, banks):
    shards = [read_json(shard_path(args.output, index, args.layer_shard_count)) for index in range(args.layer_shard_count)]
    settings = lambda shard: {k: v for k, v in shard.items() if k not in ("layers", "anchor_kl", "elapsed_seconds")}
    first = shards[0]
    for shard in shards[1:]:
        # Every shard recomputes the anchor KL on its own GPU; allow run-to-run BF16 reduction noise.
        assert settings(shard) == settings(first) and abs(shard["anchor_kl"] - first["anchor_kl"]) <= 1e-4 * max(1.0, abs(first["anchor_kl"])), (
            "shards were measured with different settings"
        )
    rows = sorted((row for shard in shards for row in shard["layers"]), key=lambda row: row["layer"])
    elapsed = sum(shard["elapsed_seconds"] for shard in shards)
    write_json(args.output / "probes.json", {**first, "layers": rows, "elapsed_seconds": elapsed})
    log.info("merged %d layers into %s", len(rows), args.output / "probes.json")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local Hugging Face model directory")
    parser.add_argument("--windows", type=Path, required=True, help="profile windows (input_ids) from calibration.windows")
    parser.add_argument("--banks", nargs="+", required=True, metavar="RANK=DIR", help="uniform banks of the anchor and both probe ranks")
    parser.add_argument("--anchor-rank", type=int, required=True)
    parser.add_argument("--low-rank", type=int, required=True)
    parser.add_argument("--high-rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-count", type=int, help="leading windows to use; default: all")
    parser.add_argument("--sequence-length", type=int, help="leading tokens per window; default: the stored length")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer-shard-index", type=int, default=0)
    parser.add_argument("--layer-shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="combine the layer shards into probes.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("none", "auto", "balanced"), default="none")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args(argv)
    assert args.low_rank < args.anchor_rank < args.high_rank, "probes must bracket the anchor rank"
    banks = parse_banks(args.banks)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args, banks)

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model_hash = sha256(args.model / "config.json")
    windows = load_file(str(args.windows))["input_ids"]
    windows = windows[: args.window_count or len(windows), : args.sequence_length or windows.shape[1]]
    model = load_model(args.model, dtype=args.dtype, device=args.device, device_map=args.device_map)
    natives, modules = bank_modules(model, banks)
    layers = list(natives)[args.layer_shard_index :: args.layer_shard_count]
    started = time.monotonic()
    anchor_kl, positions, rows = measure(
        model, windows, natives, modules, anchor=args.anchor_rank, low=args.low_rank, high=args.high_rank,
        layers=layers, batch_size=args.batch_size,
    )
    write_json(shard_path(args.output, args.layer_shard_index, args.layer_shard_count), {
        "format": PROBES_FORMAT,
        "model": str(args.model.resolve()),
        "model_config_sha256": model_hash,
        "banks": {str(rank): str(directory) for rank, directory in banks.items()},
        "anchor_rank": args.anchor_rank,
        "low_rank": args.low_rank,
        "high_rank": args.high_rank,
        "windows": len(windows),
        "sequence_length": int(windows.shape[1]),
        "positions": positions,
        "dtype": args.dtype,
        "anchor_kl": anchor_kl,
        "layers": rows,
        "elapsed_seconds": time.monotonic() - started,
    })
    for row in rows:
        log.info("layer %d: anchor %.6g, low %+.6g, high %+.6g", row["layer"], anchor_kl, row["delta_low"], row["delta_high"])


if __name__ == "__main__":
    main()
