"""Table 7: the Group-SVD baseline, an activation-weighted SVD per physical KV group.

Each group's encoder is the leading subspace of its own pooled attention-output
covariance (the same initialization BasisKV starts from); each query head's
decoder is the closed-form least-squares fit to that head's output under the
group covariance alone. Unlike BasisKV there is no cross-head coupling and no
joint refinement. The output has the `basiskv.value_factors.v1` layout, so
`eval.ppl` and `eval.mcq` evaluate it through `--value-factors`.

    python -m appendix.group_svd --snapshot-dir SNAPSHOTS --output-dir FACTORS --rank 96
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from core import snapshots
from core.als import evaluate_loss, initialize_weighted_svd, quadratic_from_target
from core.factor_io import merge, parse_layers, write_layer
from core.model_layout import ValueGeometry

log = logging.getLogger("basiskv.group_svd")


def group_decoders(covariance, target, encoder, mapping, jitter):
    """Per-head decoders D_h = (E^T S_g E)^-1 E^T S_g W_h with S_g the pooled group covariance."""
    decoder = target.new_empty(target.shape[0], encoder.shape[-1], target.shape[-1])
    for group, e in enumerate(encoder):
        heads = torch.nonzero(mapping == group).flatten()
        pooled = covariance[heads, heads].mean(dim=0)
        pooled = 0.5 * (pooled + pooled.T) + jitter[group] * torch.eye(pooled.shape[0], device=pooled.device, dtype=pooled.dtype)
        normal = e.T @ pooled @ e
        rhs = torch.einsum("ri,hio->hro", e.T @ pooled, target[heads])
        decoder[heads] = torch.linalg.solve(normal, rhs)
    return decoder


@torch.no_grad()
def build(args):
    device = torch.device(args.device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = snapshots.read_manifest(snapshot_dir)
    geometry = ValueGeometry.from_manifest(manifest)
    assert 0 < args.rank <= geometry.head_dim, "rank is outside the attention head width"
    layers = parse_layers(args.layers, geometry)
    mapping = geometry.head_to_kv_group(device)
    factor_dtype = getattr(torch, args.factor_dtype)
    model, calibration = manifest["model"], manifest["calibration"]
    config = {
        "method": "group_svd",
        "encoder_sweeps": 0,
        "cg_iterations": 0,
        "work_dtype": "float64",
        "factor_dtype": args.factor_dtype,
        "allow_tf32": False,
        "covariance_damping": 0.0,
        "cache_rank_per_head": args.rank,
        "model": str(Path(model["path"]).expanduser().resolve()),
        "geometry": geometry.to_dict(),
        "fit_windows": int(calibration["fit_windows"]),
        "heldout_windows": int(calibration["heldout_windows"]),
        "positions_per_window": int(calibration["positions_per_window"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        started = time.perf_counter()
        payload, source = snapshots.load_layer(snapshot_dir, manifest, layer)
        fit, _ = snapshots.stored_covariance(payload["fit_covariance"], geometry, device)
        heldout, _ = snapshots.stored_covariance(payload["heldout_covariance"], geometry, device)
        target = snapshots.head_targets(payload["weight"], geometry, device)
        encoder, initialization = initialize_weighted_svd(fit, target, mapping, args.rank)
        jitter = [item["absolute_jitter"] for item in initialization]
        decoder = group_decoders(fit, target, encoder, mapping, jitter)
        fit_objective, heldout_objective = quadratic_from_target(fit, target), quadratic_from_target(heldout, target)
        stored = (encoder.to(factor_dtype), decoder.to(factor_dtype))
        restored = tuple(value.to(device, torch.float64) for value in stored)

        def losses(objective):
            return {
                "relative_mse": evaluate_loss(objective, encoder, decoder, mapping)[1],
                "factor_dtype_relative_mse": evaluate_loss(objective, *restored, mapping)[1],
            }

        record = {
            "fit_config": config,
            "source_snapshot": source,
            "fit": losses(fit_objective),
            "heldout": losses(heldout_objective),
            "covariance": None,
            "initialization": initialization,
            "solver": {"method": "group_svd", "sweeps": 0},
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_layer(output_dir, layer, *stored, record)
        log.info("layer %d: heldout=%.8g seconds=%.2f", layer, record["heldout"]["factor_dtype_relative_mse"],
                 record["elapsed_seconds"])
    merge(output_dir, layers, geometry)
    log.info("wrote %d layers to %s", len(layers), output_dir)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", required=True, help="calibration.value output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True, help="retained V dimensions per KV head")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--factor-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
