"""Fit the affine Value-to-Key base and the Page-Fisher residual router from saved statistics.

    python -m core.fit_router base --statistics-dir CAPTURE --output-dir CAPTURE/base --base-rank 16
    python -m core.fit_router fit --statistics-dir CAPTURE --output-dir ROUTER_BANK --residual-rank 16

`base` reads CAPTURE/moments; `fit` reads CAPTURE/base and CAPTURE/fisher.
Neither stage loads model weights or a tokenizer.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from core.page_fisher import fisher_loss, fit_router
from core.router import base_mse, fit_base
from core.router_io import layer_file, load_fisher_statistics, load_record, save_record

log = logging.getLogger("basiskv.fit_router")
MOMENT_NAMES = ("count", "sum_v", "sum_k", "vv", "vk", "kk")


def selected_layers(directory, specification, shard, shards):
    available = sorted(int(path.stem.removeprefix("layer_")) for path in directory.glob("layer_*.safetensors"))
    layers = available if specification == "all" else sorted(int(piece) for piece in specification.split(","))
    assert layers and set(layers) <= set(available), f"no layer statistics for {specification} under {directory}"
    return layers[shard::shards]


def completed(path, config, resume):
    """True when a verified output with this configuration already exists."""
    if not path.exists() and not path.with_suffix(".json").exists():
        return False
    assert resume, f"existing output: {path}; use --resume to keep it"
    _, record = load_record(path)
    assert record["fit_config"] == config, f"fit inputs or options changed: {path}"
    log.info("layer %d: resumed from verified output", record["layer"])
    return True


def fit_bases(args, layers):
    for layer in layers:
        payload, record = load_record(layer_file(args.statistics_dir, "moments", layer))
        encoder = payload["encoder"]
        assert 0 <= args.base_rank <= min(encoder.shape[1:]), "base rank exceeds the Value code width"
        moments = {
            split: {name: payload[f"{split}_{name}"] for name in MOMENT_NAMES} for split in ("fit", "heldout")
        }
        path = args.output_dir / f"layer_{layer:03d}.safetensors"
        config = {"stage": "base", "base_rank": args.base_rank}
        if completed(path, config, args.resume):
            continue
        started = time.monotonic()
        maps = fit_base(moments["fit"], encoder, args.base_rank)
        tensors = {name: torch.stack([getattr(item, name) for item in maps]) for name in ("left", "right", "bias")}
        tensors["encoder"] = encoder
        assert all(torch.isfinite(value).all() for value in tensors.values()), "non-finite base"
        metrics = {split: base_mse(moment, encoder, maps) for split, moment in moments.items() if int(moment["count"])}
        save_record(path, tensors, {
            "layer": layer,
            "protocol": record["protocol"],
            "fit_config": config,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
        })
        log.info("layer %d: base rank %d, relative MSE %s", layer, args.base_rank,
                 {split: round(value["relative_mse"], 6) for split, value in metrics.items()})


def fit_residuals(args, layers):
    device = torch.device(args.device)
    for layer in layers:
        base, record, statistics = load_fisher_statistics(args.statistics_dir, layer)
        base_rank, dim = base["left"].shape[-1], base["encoder"].shape[1]
        assert 0 <= args.residual_rank <= dim, "residual rank exceeds the Key width"
        config = {
            "stage": "fit",
            "base_rank": base_rank,
            "residual_rank": args.residual_rank,
            "sweeps": args.sweeps,
            "pcg_iterations": args.pcg_iterations,
            "pcg_tolerance": args.pcg_tolerance,
            "relative_damping": args.relative_damping,
            "device": str(device),
            "allow_tf32": args.allow_tf32,
        }
        path = args.output_dir / f"layer_{layer:03d}.safetensors"
        if completed(path, config, args.resume):
            continue
        started = time.monotonic()
        fit = statistics["fit"].to(device)
        heldout = statistics["heldout"].to(device) if "heldout" in statistics else fit
        encoder, query, info = fit_router(
            fit, args.residual_rank, sweeps=args.sweeps, relative_damping=args.relative_damping,
            cg_iterations=args.pcg_iterations, cg_tolerance=args.pcg_tolerance,
        )
        losses = {
            "fit_page_fisher_nmse": fisher_loss(fit, encoder, query) / fit.teacher_energy,
            "heldout_page_fisher_nmse": fisher_loss(heldout, encoder, query) / heldout.teacher_energy,
            **info,
        }
        suffix = f"b{base_rank}_r{args.residual_rank}"
        tensors = {f"base_{name}_b{base_rank}": base[name].float() for name in ("left", "right", "bias")}
        tensors[f"residual_encoder_{suffix}"] = encoder.cpu().float()
        tensors[f"residual_query_{suffix}"] = query.cpu().float()
        assert all(torch.isfinite(value).all() for value in tensors.values()), "non-finite router"
        heldout_scope = "held-out diagnostic windows" if "heldout" in statistics else "in-sample: no diagnostic windows"
        save_record(path, tensors, {
            "layer": layer,
            "protocol": {**record["protocol"], "residual_rank": args.residual_rank, "heldout_scope": heldout_scope},
            "fit_config": config,
            "v_rank": base["encoder"].shape[-1],
            "sweeps": args.sweeps,
            "pcg_iterations": args.pcg_iterations,
            "losses": losses,
            "elapsed_seconds": time.monotonic() - started,
        })
        log.info("layer %d: fit nmse %.6g, held-out nmse %.6g (%s)", layer,
                 losses["fit_page_fisher_nmse"], losses["heldout_page_fisher_nmse"], heldout_scope)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("base", "fit"):
        stage = stages.add_parser(name)
        stage.add_argument("--statistics-dir", type=Path, required=True)
        stage.add_argument("--output-dir", type=Path, required=True)
        stage.add_argument("--layers", default="all")
        stage.add_argument("--layer-shard-index", type=int, default=0)
        stage.add_argument("--layer-shard-count", type=int, default=1)
        stage.add_argument("--torch-num-threads", type=int, default=1)
        stage.add_argument("--resume", action="store_true")
        if name == "base":
            stage.add_argument("--base-rank", type=int, required=True)
        else:
            stage.add_argument("--residual-rank", type=int, required=True)
            stage.add_argument("--device", default="cuda:0")
            stage.add_argument("--sweeps", type=int, default=40)
            stage.add_argument("--pcg-iterations", type=int, default=100)
            stage.add_argument("--pcg-tolerance", type=float, default=1e-5)
            stage.add_argument("--relative-damping", type=float, default=1e-5)
            stage.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = getattr(args, "allow_tf32", False)
    args.statistics_dir = args.statistics_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    source = "moments" if args.stage == "base" else "base"
    layers = selected_layers(args.statistics_dir / source, args.layers, args.layer_shard_index, args.layer_shard_count)
    assert args.output_dir != args.statistics_dir / source, "output would overwrite the input statistics"
    if args.stage == "base":
        fit_bases(args, layers)
    else:
        fit_residuals(args, layers)


if __name__ == "__main__":
    main()
