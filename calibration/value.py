"""Capture the attention output projection's input covariances for core.fit_value.

    python -m calibration.value --model MODEL --windows WINDOWS/windows.safetensors --output SNAPSHOTS

Every attention layer gets the FP64 Gram of its o_proj input over the fit windows and over
the held-out windows, normalized by row count, plus the dense o_proj weight, in the
`basisserve.attention_o_proj_covariances.v1` layout. For large models split the windows with
--window-shard-index / --window-shard-count and rerun with --merge afterwards.
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import torch

from calibration.common import add_model_arguments, add_window_arguments, load_teacher, read_windows, setup
from core import snapshots
from core.factor_io import parse_layers
from core.model_layout import ValueGeometry
from core.router_io import load_record, save_record, save_tensors, write_json

log = logging.getLogger("basiskv.value")
CHUNK_ROWS = 2048


def write_layer(output, layer, tensors):
    payload = {f"{split}_covariance": tensors[f"{split}_sum"] / int(tensors[f"{split}_count"]) for split in ("fit", "heldout")}
    payload["weight"] = tensors["weight"]
    save_tensors(output / f"layer_{layer:03d}.safetensors", payload)


def write_manifest(args, geometry, layers, split_ids):
    write_json(args.output / "manifest.json", dict(
        format=snapshots.COVARIANCE_FORMAT, status="complete", layers=list(layers),
        model=dict(path=str(args.model), model_type=geometry.model_type, num_hidden_layers=geometry.num_layers,
                   num_attention_heads=geometry.num_heads, num_key_value_heads=geometry.num_kv_heads,
                   head_dim=geometry.head_dim, hidden_size=geometry.hidden_size),
        artifacts={str(layer): dict(file=f"layer_{layer:03d}.safetensors") for layer in layers},
        calibration=dict(fit_windows=len(split_ids["fit"]), heldout_windows=len(split_ids["heldout"]),
                         positions_per_window=args.sequence_length)))


@torch.inference_mode()
def capture(args, windows, split_ids):
    model, _, geometry, attention, _ = load_teacher(args)
    assert geometry.model_type != "qwen3_5", "Qwen3.5 requires separately fitted gated-Value banks"
    ids = {split: values[args.window_shard_index :: args.window_shard_count] for split, values in split_ids.items()}
    width = geometry.num_heads * geometry.head_dim
    grams = {layer: {} for layer in attention}
    active = {}

    def accumulate(layer):
        def hook(module, positional):
            rows = positional[0].reshape(-1, width)
            if active["split"] not in grams[layer]:
                grams[layer][active["split"]] = torch.zeros(width, width, device=rows.device, dtype=torch.float64)
            gram = grams[layer][active["split"]]
            for chunk in rows.split(CHUNK_ROWS):
                gram.addmm_(chunk.double().T, chunk.double())
        return hook

    handles = [module.o_proj.register_forward_pre_hook(accumulate(layer)) for layer, module in attention.items()]
    device = model.get_input_embeddings().weight.device
    for split, indices in ids.items():
        for index in indices:
            active["split"] = split
            started = time.monotonic()
            output = model.model(input_ids=windows[index : index + 1].to(device, torch.long), use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all(), "non-finite teacher output"
            del output
            log.info("%s window %d: %.1fs", split, index, time.monotonic() - started)
        for layer in attention:  # one split's Grams stay on the GPU at a time: 36 x 4096^2 FP64 is 4.8 GB on an 8B model
            grams[layer][split] = grams[layer][split].cpu() if split in grams[layer] else torch.zeros(width, width, dtype=torch.float64)
    for handle in handles:
        handle.remove()
    for layer, module in attention.items():
        tensors = {"weight": module.o_proj.weight.detach().cpu().contiguous()}
        for split, indices in ids.items():
            tensors[f"{split}_sum"] = grams[layer][split]
            tensors[f"{split}_count"] = torch.tensor(len(indices) * args.sequence_length)
        if args.window_shard_count > 1:
            save_record(args.output / "parts" / f"layer_{layer:03d}_s{args.window_shard_index:03d}.safetensors", tensors, dict(
                layer=layer, ids=ids, window_shard_count=args.window_shard_count, sequence_length=args.sequence_length,
                geometry=geometry.to_dict()))
        else:
            write_layer(args.output, layer, tensors)
    if args.window_shard_count == 1:
        write_manifest(args, geometry, tuple(attention), split_ids)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merge(args, split_ids):
    parts = sorted((args.output / "parts").glob("layer_*_s000.safetensors"))
    layers = [int(path.stem.split("_")[1]) for path in parts]
    geometry = None
    for layer in layers:
        combined, seen = {}, {split: [] for split in split_ids}
        for shard in range(args.window_shard_count):
            tensors, metadata = load_record(args.output / "parts" / f"layer_{layer:03d}_s{shard:03d}.safetensors")
            assert metadata["window_shard_count"] == args.window_shard_count and metadata["sequence_length"] == args.sequence_length, (
                f"layer {layer} shard {shard} comes from another capture")
            geometry = ValueGeometry.from_dict(metadata["geometry"])
            for split, indices in metadata["ids"].items():
                seen[split].extend(indices)
            for key, value in tensors.items():
                combined[key] = value if key == "weight" else combined.get(key, 0) + value
        assert all(sorted(seen[split]) == split_ids[split] for split in split_ids), f"layer {layer}: missing or repeated windows"
        write_layer(args.output, layer, combined)
        log.info("merged layer %d from %d shards", layer, args.window_shard_count)
    assert layers == list(parse_layers(args.layers, geometry)), "the shards do not cover --layers"
    write_manifest(args, geometry, layers, split_ids)


def main(argv=None):
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    add_window_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.model, args.output = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    windows, split_ids = read_windows(args)
    assert split_ids["heldout"], "Value fitting needs held-out windows for the error curves"
    if args.merge:
        merge(args, split_ids)
    else:
        capture(args, windows, split_ids)


if __name__ == "__main__":
    main()
