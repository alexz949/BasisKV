"""Build dense and per-head weight-SVD reference checkpoints."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping
import torch
from torch import Tensor
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from others.weight_svd import factorize_weight_svd_projection
from others import model_support

EXPERIMENT_PROFILES = {
    "qwen3_8b": {
        "format": "basisserve.qwen3_8b.iclr_v_factors.v1",
        "run_id_prefix": "Q3-8B",
    },
    "llama31_8b": {
        "format": "basisserve.llama31_8b.iclr_v_factors.v1",
        "run_id_prefix": "L31-8B",
    },
    "llama2_7b": {
        "format": "basisserve.llama2_7b.iclr_v_factors.v1",
        "run_id_prefix": "L2-7B",
    },
    "qwen3_32b": {
        "format": "basisserve.qwen3_32b.iclr_v_factors.v1",
        "run_id_prefix": "Q3-32B",
    },
    "llama31_70b": {
        "format": "basisserve.llama31_70b.iclr_v_factors.v1",
        "run_id_prefix": "L31-70B",
    },
}
FORMAT = str(EXPERIMENT_PROFILES["qwen3_8b"]["format"])
PROFILE = "qwen3_8b"
RUN_ID_PREFIX = str(EXPERIMENT_PROFILES["qwen3_8b"]["run_id_prefix"])
SUPPORTED_EQUIVALENT_RANKS = (64, 80, 96)
SUPPORTED_HEAD_GROUP_SIZES = (1,)


def activate_experiment_profile(name: str) -> None:
    experiment = EXPERIMENT_PROFILES[name]
    global FORMAT, PROFILE, RUN_ID_PREFIX
    FORMAT = str(experiment["format"])
    PROFILE = name
    RUN_ID_PREFIX = str(experiment["run_id_prefix"])
    model_support.activate_model_profile(name)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def run_id(method, equivalent_rank, head_group_size):
    assert method in ("dense", "weight-svd")
    return (
        f"{RUN_ID_PREFIX}-Dense"
        if method == "dense"
        else f"{RUN_ID_PREFIX}-SVD-R{equivalent_rank}"
    )


def _dense_manifest(model_metadata: Mapping[str, Any]) -> dict[str, Any]:
    dense_rank_sum = (
        model_support.NUM_LAYERS * model_support.NUM_KV_HEADS * model_support.HEAD_DIM
    )
    return {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id("dense", None, 1),
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "dense_kv_cache",
            "key_cache": "dense",
            "value_cache": "dense",
            "projection": None,
            "method": "dense",
            "method_label": "Dense",
            "num_query_heads": model_support.NUM_QUERY_HEADS,
            "num_physical_kv_heads": model_support.NUM_KV_HEADS,
            "head_dim": model_support.HEAD_DIM,
            "rank_sum_across_layers": dense_rank_sum,
            "dense_rank_sum_across_layers": dense_rank_sum,
            "realized_retained_v_ratio": 1.0,
            "realized_v_cache_compression_ratio": 0.0,
        },
        "artifact": None,
        "layers": [],
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> int:
    model_support.activate_model_profile(PROFILE)
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not _check(
        not output_dir.exists(), f"output directory already exists: {output_dir}"
    ):
        return 2
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model_support._validate_config(config)
    model_metadata = model_support._model_metadata(model_path)
    if args.method == "dense":
        output_dir.mkdir(parents=True)
        model_support._atomic_json(
            output_dir / "manifest.json", _dense_manifest(model_metadata)
        )
        print(f"[Result] dense reference={output_dir}", flush=True)
        return 0
    equivalent_rank = int(args.equivalent_rank or 0)
    valid = all(
        (
            _check(
                equivalent_rank in SUPPORTED_EQUIVALENT_RANKS,
                f"equivalent rank must be one of {SUPPORTED_EQUIVALENT_RANKS}",
            ),
            _check(
                args.head_group_size in SUPPORTED_HEAD_GROUP_SIZES,
                f"head group size must be one of {SUPPORTED_HEAD_GROUP_SIZES}",
            ),
            _check(
                args.method != "weight-svd" or args.head_group_size == 1,
                "Weight-SVD is defined per physical V head",
            ),
        )
    )
    if not valid:
        return 2
    layer_ranks = [
        [equivalent_rank] * model_support.NUM_KV_HEADS
        for _ in range(model_support.NUM_LAYERS)
    ]
    rank_sum = sum((sum(ranks) for ranks in layer_ranks))
    total_rank = (
        model_support.NUM_LAYERS * model_support.NUM_KV_HEADS * model_support.HEAD_DIM
    )
    factor_payload: dict[str, Tensor] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index, ranks in enumerate(layer_ranks):
        layer_started = time.perf_counter()
        tensor_name = f"model.layers.{layer_index}.self_attn.v_proj.weight"
        dense = model_support._load_indexed_tensor(model_path, tensor_name)
        expected_shape = (
            model_support.NUM_KV_HEADS * model_support.HEAD_DIM,
            model_support.HIDDEN_SIZE,
        )
        if not _check(
            tuple(dense.shape) == expected_shape,
            f"unexpected v_proj shape at layer {layer_index}: {tuple(dense.shape)}",
        ):
            return 2
        (writer, decoder, diagnostics) = factorize_weight_svd_projection(
            dense, ranks=ranks, output_dtype=torch.bfloat16
        )
        factor_payload[f"layers.{layer_index}.v_writer.weight"] = writer
        factor_payload[f"layers.{layer_index}.v_decoder.weight"] = decoder
        records.append(
            {
                "layer": layer_index,
                "ranks": list(ranks),
                "source_tensor": tensor_name,
                "source_dtype": str(dense.dtype),
                "writer_shape": list(writer.shape),
                "decoder_shape": list(decoder.shape),
                "factor_dtype": str(writer.dtype),
                **diagnostics,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
        )
        metric = diagnostics["relative_frobenius_error"]
        print(
            f"[{args.method}] layer={layer_index}/{model_support.NUM_LAYERS - 1} group_rank={ranks[0]} error={metric:.6f}",
            flush=True,
        )
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "v_factors.safetensors"
    model_support._atomic_safetensors(artifact_path, factor_payload)
    group_count = model_support.NUM_KV_HEADS // args.head_group_size
    method_label = "Weight-SVD"
    realized_retained_ratio = rank_sum / total_rank
    compression = {
        "target": "value_cache_only",
        "key_cache": "dense",
        "value_cache": "low_rank",
        "projection": "v",
        "method": args.method,
        "method_label": method_label,
        "allocation": "uniform_per_physical_head",
        "equivalent_rank_target": equivalent_rank,
        "nominal_retained_v_ratio": equivalent_rank / model_support.HEAD_DIM,
        "nominal_v_cache_compression_ratio": 1.0
        - equivalent_rank / model_support.HEAD_DIM,
        "rank_block_size": None,
        "num_query_heads": model_support.NUM_QUERY_HEADS,
        "num_physical_kv_heads": model_support.NUM_KV_HEADS,
        "head_dim": model_support.HEAD_DIM,
        "head_group_size": args.head_group_size,
        "groups": group_count,
        "group_width": args.head_group_size * model_support.HEAD_DIM,
        "nominal_group_rank": args.head_group_size * equivalent_rank,
        "layer_ranks": layer_ranks,
        "rank_sum_across_layers": rank_sum,
        "dense_rank_sum_across_layers": total_rank,
        "realized_retained_v_ratio": realized_retained_ratio,
        "realized_v_cache_compression_ratio": 1.0 - realized_retained_ratio,
        "factorization_work_device": "cpu",
        "factorization_work_dtype": "torch.float32",
        "stored_factor_dtype": "torch.bfloat16",
    }
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id(args.method, equivalent_rank, args.head_group_size),
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": compression,
        "artifact": {
            "file": artifact_path.name,
            "sha256": model_support._sha256(artifact_path),
            "tensor_count": len(factor_payload),
            "bytes": artifact_path.stat().st_size,
        },
        "layers": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    model_support._atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[Result] run_id={manifest['run_id']} realized_retained_v_ratio={realized_retained_ratio:.9f} artifact={artifact_path}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dense", "weight-svd"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--equivalent-rank", type=int)
    parser.add_argument("--head-group-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
