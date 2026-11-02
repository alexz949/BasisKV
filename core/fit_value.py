"""Fit V/O joint ALS factors from a calibration snapshot; merge fitted layer shards.

    python -m core.fit_value fit-shard --snapshot-dir S --output-dir F --rank 96
    python -m core.fit_value merge --output-dir F

The snapshot (`calibration.value`) carries the fit and held-out
covariances of every layer; held-out error is reported, never fitted.
"""

import argparse
import logging
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from core import snapshots
from core.als import damp_covariance, evaluate_loss, fit_joint, initialize_weighted_svd, quadratic_from_target
from core.factor_io import FORMAT, cg_summary, merge, parse_layers, read_layer, write_json, write_layer
from core.model_layout import ValueGeometry

log = logging.getLogger("basiskv.fit_value")


@dataclass(frozen=True)
class FitSettings:
    cache_rank: int
    encoder_sweeps: int
    cg_iterations: int
    work_dtype: str
    factor_dtype: str
    allow_tf32: bool
    covariance_damping: float

    @classmethod
    def from_args(cls, args):
        return cls(**{field: getattr(args, field) for field in cls.__dataclass_fields__})

    @property
    def torch_work_dtype(self):
        return getattr(torch, self.work_dtype)

    @property
    def torch_factor_dtype(self):
        return getattr(torch, self.factor_dtype)


def fit_config(settings, geometry, manifest):
    """Solver settings, model identity, and data partition shared by every layer record.

    It is also the resume and merge consistency key; `cache_rank_per_head` is the
    one entry that rank allocation lets differ between the layers of a bank.
    """
    assert 0 < settings.cache_rank <= geometry.head_dim, "rank is outside the attention head width"
    model, calibration = manifest["model"], manifest["calibration"]
    return {
        **{key: value for key, value in asdict(settings).items() if key != "cache_rank"},
        "cache_rank_per_head": settings.cache_rank,
        "model": str(Path(model["path"]).expanduser().resolve()),
        "geometry": geometry.to_dict(),
        "fit_windows": int(calibration["fit_windows"]),
        "heldout_windows": int(calibration["heldout_windows"]),
        "positions_per_window": int(calibration["positions_per_window"]),
    }


def _exact_full_rank(weight, geometry, dtype):
    """Identity encoders and the dense o_proj as decoders: the rank == head_dim endpoint."""
    encoder = torch.eye(geometry.head_dim, dtype=dtype).expand(geometry.num_kv_heads, -1, -1)
    decoder = snapshots.head_targets(weight, geometry, torch.device("cpu")).to(dtype)
    return encoder, decoder


@torch.no_grad()
def fit_layer(layer, *, snapshot_dir, manifest, output_dir, config, settings, geometry, device, resume):
    started = time.perf_counter()
    if resume:
        prior = read_layer(output_dir, layer, config)
        if prior is not None:
            log.info("layer %d: resumed from verified output", layer)
            return prior
    payload, source = snapshots.load_layer(snapshot_dir, manifest, layer)
    factor_dtype = settings.torch_factor_dtype
    record = {"fit_config": config, "source_snapshot": source}

    if settings.cache_rank == geometry.head_dim:
        encoder, decoder = _exact_full_rank(payload["weight"], geometry, factor_dtype)
        zero = {"relative_mse": 0.0, "factor_dtype_relative_mse": 0.0}
        record.update(fit=zero, heldout=zero, covariance=None, initialization=[])
        record["solver"] = {"method": "exact_full_rank", "sweeps": 0}
    else:
        fit_raw, fit_repair = snapshots.stored_covariance(payload["fit_covariance"], geometry, device)
        heldout_raw, heldout_repair = snapshots.stored_covariance(payload["heldout_covariance"], geometry, device)
        fit_covariance, damping = damp_covariance(fit_raw, settings.covariance_damping)
        target = snapshots.head_targets(payload["weight"], geometry, device)
        fit_objective = quadratic_from_target(fit_covariance, target)
        heldout_objective = quadratic_from_target(heldout_raw, target)
        del fit_raw, heldout_raw, fit_covariance
        mapping = geometry.head_to_kv_group(device)
        work_dtype = settings.torch_work_dtype
        initial, initialization = initialize_weighted_svd(
            fit_objective.covariance.to(work_dtype), target.to(work_dtype), mapping, settings.cache_rank
        )
        encoder, decoder, info = fit_joint(
            fit_objective, initial, mapping,
            sweeps=settings.encoder_sweeps, cg_iterations=settings.cg_iterations, work_dtype=work_dtype,
        )
        fitted = (encoder.to(device, work_dtype), decoder.to(device, work_dtype))
        encoder, decoder = encoder.to(factor_dtype), decoder.to(factor_dtype)
        stored = (encoder.to(device, work_dtype), decoder.to(device, work_dtype))

        def losses(objective):
            return {
                "relative_mse": evaluate_loss(objective, *fitted, mapping)[1],
                "factor_dtype_relative_mse": evaluate_loss(objective, *stored, mapping)[1],
            }

        record["fit"] = losses(fit_objective)
        record["heldout"] = losses(heldout_objective)
        record["covariance"] = {
            "fit_absolute_trace_damping": damping,
            "stored_gram_psd_repair": {"fit": fit_repair, "heldout": heldout_repair},
        }
        record["initialization"] = initialization
        record["solver"] = {
            "method": "als",
            "initial_loss": info["initial_loss"],
            "decoder_only_loss": info["decoder_only_loss"],
            "final_loss": info["final_loss"],
            "sweeps": info["sweeps"],
            "cg": cg_summary(info["updates"], settings.cg_iterations),
        }
    record["elapsed_seconds"] = time.perf_counter() - started
    complete = write_layer(output_dir, layer, encoder, decoder, record)
    log.info(
        "layer %d: %s heldout=%.8g seconds=%.2f",
        layer, record["solver"]["method"], record["heldout"]["factor_dtype_relative_mse"], record["elapsed_seconds"],
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return complete


def fit_shard(args):
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    manifest = snapshots.read_manifest(snapshot_dir)
    geometry = ValueGeometry.from_manifest(manifest)
    settings = FitSettings.from_args(args)
    config = fit_config(settings, geometry, manifest)
    requested = parse_layers(args.layers, geometry)
    layers = requested[args.layer_shard_index :: args.layer_shard_count]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for layer in layers:
        fit_layer(
            layer, snapshot_dir=snapshot_dir, manifest=manifest, output_dir=output_dir, config=config,
            settings=settings, geometry=geometry, device=device, resume=args.resume,
        )
    cuda = device.type == "cuda"
    write_json(output_dir / f"shard_{args.layer_shard_index:02d}.json", {
        "format": FORMAT,
        "status": "shard_complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "layers": list(layers),
        "fit_config": config,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if cuda else None,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if cuda else None,
            "torch_num_threads": torch.get_num_threads(),
        },
    })
    log.info("shard %d/%d complete: layers %s", args.layer_shard_index, args.layer_shard_count, list(layers))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit-shard", help="fit the layers of one shard")
    fit.add_argument("--snapshot-dir", required=True, help="calibration.value output")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--layers", default="all", help="'all', or indices and ranges such as 0-3,7")
    fit.add_argument("--rank", dest="cache_rank", type=int, required=True, help="retained V dimensions per KV head")
    fit.add_argument("--encoder-sweeps", type=int, default=6, help="encoder/decoder ALS sweeps")
    fit.add_argument("--cg-iterations", type=int, default=16, help="fixed CG iterations per encoder update")
    fit.add_argument("--covariance-damping", type=float, default=1e-7, help="relative trace damping of the fit covariance")
    fit.add_argument("--work-dtype", choices=("float32", "float64"), default="float32")
    fit.add_argument("--factor-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    fit.add_argument("--allow-tf32", action="store_true")
    fit.add_argument("--layer-shard-index", type=int, default=0)
    fit.add_argument("--layer-shard-count", type=int, default=1)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true", help="keep verified layer outputs already in --output-dir")

    merge_parser = commands.add_parser("merge", help="collect fitted layers into results.json")
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--layers", default="all")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    args = parse_args()
    if args.command == "fit-shard":
        fit_shard(args)
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        geometry = ValueGeometry.from_factors(output_dir)
        payload = merge(output_dir, parse_layers(args.layers, geometry), geometry)
        log.info("merged %d layers into %s", len(payload["records"]), output_dir / "results.json")


if __name__ == "__main__":
    main()
