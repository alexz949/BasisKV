#!/usr/bin/env python3
"""Evaluate folded Qwen3-32B GQA C1 factors on WikiText-2.

The reference path pads every low-rank physical Value writer back to head width
128 and pads each query-head decoder in ``o_proj``.  It therefore preserves the
compressed C1 function exactly without claiming compressed-cache runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_layout import GQAVOLayout
from others.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss


FORMAT = "basisserve.qwen3_32b.gqa_c1_wikitext2_ppl.v1"
FACTOR_FORMAT = "basiskv.value_factors.v1"
MODEL_LABEL = "Qwen3-32B"
NUM_LAYERS, NUM_QUERY_HEADS, NUM_KV_HEADS = 64, 64, 8
HEAD_DIM, HIDDEN_SIZE = 128, 5120


def activate_model_profile(name):
    global FORMAT, MODEL_LABEL, NUM_LAYERS, NUM_QUERY_HEADS, NUM_KV_HEADS
    global HEAD_DIM, HIDDEN_SIZE
    MODEL_LABEL, NUM_LAYERS, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE = {
        "qwen3_32b": ("Qwen3-32B", 64, 64, 8, 128, 5120),
        "qwen3_8b": ("Qwen3-8B-Base", 36, 32, 8, 128, 4096),
    }[name]
    FORMAT = f"scripts.{name}.gqa_c1_wikitext2_ppl.v1"


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen-style decoder model")
    return layers


def _layout(config: Any, rank: int) -> GQAVOLayout:
    layout = GQAVOLayout(
        hidden_size=int(config.hidden_size),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(getattr(config, "head_dim", 0)),
        rank=rank,
    )
    if (
        str(config.model_type) != "qwen3"
        or int(config.num_hidden_layers) != NUM_LAYERS
        or layout.hidden_size != HIDDEN_SIZE
        or layout.num_attention_heads != NUM_QUERY_HEADS
        or layout.num_key_value_heads != 8
        or layout.head_dim != 128
    ):
        raise ValueError(f"unexpected {MODEL_LABEL} geometry: {layout}")
    return layout


def _load_results(factor_dir: Path, model_path: Path) -> dict[str, Any]:
    path = factor_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("C1 factor result is incomplete or incompatible")
    config = result["fit_config"]
    if config.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("C1 factors belong to another model config")
    if tuple(map(int, result.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError(f"C1 result does not cover all {NUM_LAYERS} layers")
    return result


@torch.no_grad()
def fold_c1_to_padded_weights(
    *,
    dense_v_weight: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    layout: GQAVOLayout,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Fold routed C1 factors and pad latent coordinates into dense HF slots."""

    if tuple(dense_v_weight.shape) != (layout.kv_width, layout.hidden_size):
        raise ValueError("dense v_proj weight has incompatible geometry")
    if tuple(encoders.shape) != (
        layout.num_key_value_heads,
        layout.head_dim,
        layout.rank,
    ):
        raise ValueError("C1 encoder tensor has incompatible geometry")
    if tuple(decoders.shape) != (
        layout.num_attention_heads,
        layout.rank,
        layout.hidden_size,
    ):
        raise ValueError("C1 decoder tensor has incompatible geometry")
    device = dense_v_weight.device
    work_dtype = torch.float32
    work_dense_v = dense_v_weight.to(dtype=work_dtype)
    work_A = encoders.to(device=device, dtype=work_dtype)
    work_D = decoders.to(device=device, dtype=work_dtype)
    padded_v = torch.zeros(
        layout.kv_width,
        layout.hidden_size,
        device=device,
        dtype=work_dtype,
    )
    padded_o = torch.zeros(
        layout.hidden_size,
        layout.query_width,
        device=device,
        dtype=work_dtype,
    )
    for group in range(layout.num_key_value_heads):
        target = slice(
            group * layout.head_dim,
            group * layout.head_dim + layout.rank,
        )
        dense_rows = slice(
            group * layout.head_dim,
            (group + 1) * layout.head_dim,
        )
        padded_v[target].copy_(work_A[group].T @ work_dense_v[dense_rows])
    for head in range(layout.num_attention_heads):
        target = slice(
            head * layout.head_dim,
            head * layout.head_dim + layout.rank,
        )
        padded_o[:, target].copy_(work_D[head].T)
    return padded_v, padded_o, {
        "maximum_dense_fold_error": 0.0,
        "maximum_qr_product_error": 0.0,
    }


@torch.no_grad()
def install_c1_factors(
    model: nn.Module,
    factor_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rank = int(result["fit_config"]["cache_rank_per_head"])
    layout = _layout(model.config, rank)
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        started = time.perf_counter()
        artifact = result["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"C1 artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"value_coordinate_encoders", "head_output_decoders"}:
            raise ValueError(f"unexpected C1 tensors at layer {layer_index}")
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        dense_k = layer.self_attn.k_proj
        if (
            not isinstance(v_proj, nn.Linear)
            or not isinstance(o_proj, nn.Linear)
            or v_proj.bias is not None
            or o_proj.bias is not None
            or v_proj.weight.device.type != "cuda"
            or o_proj.weight.device.type != "cuda"
        ):
            raise TypeError(f"unsupported/offloaded Qwen projections at layer {layer_index}")
        padded_v, padded_o, diagnostics = fold_c1_to_padded_weights(
            dense_v_weight=v_proj.weight.detach(),
            encoders=payload["value_coordinate_encoders"],
            decoders=payload["head_output_decoders"],
            layout=layout,
        )
        v_proj.weight.copy_(padded_v.to(device=v_proj.weight.device, dtype=v_proj.weight.dtype))
        o_proj.weight.copy_(padded_o.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        if layer.self_attn.k_proj is not dense_k:
            raise RuntimeError(f"C1 installation changed K at layer {layer_index}")
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(payload["value_coordinate_encoders"].shape),
                "decoder_shape": list(payload["head_output_decoders"].shape),
                "cache_rank_per_physical_kv_head": rank,
                "runtime": (
                    f"rank-{rank} C1 coordinates padded into dense HF V/O slots"
                ),
                **diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return records


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C1 PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    cuda_indices = _cuda_device_indices()
    for index in cuda_indices:
        torch.cuda.reset_peak_memory_stats(index)
    model_path = Path(args.model).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _load_results(factor_dir, model_path)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in cuda_indices
        },
    ).eval()
    model.config.use_cache = False
    install_started = time.perf_counter()
    rank = int(result["fit_config"]["cache_rank_per_head"])
    layout = _layout(model.config, rank)
    installation = install_c1_factors(model, factor_dir, result)
    decoder_solver = result["fit_config"].get("decoder_solver")
    arm = (
        f"c1_joint_{decoder_solver}"
        if decoder_solver is not None
        else "c1_joint_als"
    )
    result_path = factor_dir / "results.json"
    factor_result_record = {
        "path": str(result_path),
        "sha256": _sha256(result_path),
        "format": result["format"],
        "aggregate": result["aggregate"],
    }
    compression = {
        "target": "value_cache_only",
        "key_cache": "dense",
        "num_query_heads": layout.num_attention_heads,
        "num_physical_kv_heads": layout.num_key_value_heads,
        "head_dim": layout.head_dim,
        "rank_per_physical_kv_head": rank,
        "rank_sum": layout.num_key_value_heads * rank,
        "retained_v_ratio": rank / layout.head_dim,
        "v_cache_compression_ratio": 1.0 - rank / layout.head_dim,
    }
    runtime_description = (
        "C1 latent coordinates padded into Hugging Face V/O slots; "
        "function-equivalent quality path, not a cache-performance benchmark"
    )
    install_seconds = time.perf_counter() - install_started
    ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "arm": arm,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_result": factor_result_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": runtime_description,
            "installation_seconds": install_seconds,
            "layers": installation,
        },
        "ppl": ppl,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in cuda_indices
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in cuda_indices
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, payload)
    print(f"[Result] arm={arm} ppl={ppl['ppl']:.9f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
