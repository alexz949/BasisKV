"""Shared model metadata, tensor IO and C4 window preparation."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Any, Mapping
from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer

WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"
MODEL_PROFILES = {
    "llama2_7b": {
        "label": "Llama-2-7B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-2-7b-hf",
        "revision": "01c7f73d771dfac7d292323805ebc428287df4f9",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "format_slug": "llama2_7b",
    },
    "llama31_8b": {
        "label": "Llama-3.1-8B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-3.1-8B",
        "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "llama31_8b",
    },
    "qwen3_8b": {
        "label": "Qwen3-8B-Base",
        "model_type": "qwen3",
        "repo": "Qwen/Qwen3-8B-Base",
        "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        "num_layers": 36,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "qwen3_8b",
    },
    "qwen3_32b": {
        "label": "Qwen3-32B-Base",
        "model_type": "qwen3",
        "repo": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "num_layers": 64,
        "hidden_size": 5120,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "qwen3_32b",
    },
    "qwen35_9b": {
        "label": "Qwen3.5-9B",
        "model_type": "qwen3_5_text",
        "repo": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "format_slug": "qwen35_9b",
    },
    "llama31_70b": {
        "label": "Llama-3.1-70B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-3.1-70B",
        "revision": "349b2ddb53ce8f2849a6c168a81980ab25258dac",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "llama31_70b",
    },
}
MODEL_LABEL = ""
MODEL_TYPE = ""
MODEL_REPO = ""
MODEL_REVISION = ""
NUM_LAYERS = 0
HIDDEN_SIZE = 0
NUM_QUERY_HEADS = 0
NUM_KV_HEADS = 0
HEAD_DIM = 0
SEQUENCE_LENGTH = 2048
CALIBRATION_SAMPLES = 128


def activate_model_profile(name: str) -> None:
    profile = MODEL_PROFILES[name]
    global MODEL_LABEL, MODEL_TYPE, MODEL_REPO, MODEL_REVISION
    global NUM_LAYERS, HIDDEN_SIZE, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM
    MODEL_LABEL = str(profile["label"])
    MODEL_TYPE = str(profile["model_type"])
    MODEL_REPO = str(profile["repo"])
    MODEL_REVISION = str(profile["revision"])
    NUM_LAYERS = int(profile["num_layers"])
    HIDDEN_SIZE = int(profile["hidden_size"])
    NUM_QUERY_HEADS = int(profile["num_query_heads"])
    NUM_KV_HEADS = int(profile["num_kv_heads"])
    HEAD_DIM = int(profile["head_dim"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _hf_download_revision(model_path: Path) -> str | None:
    metadata = (
        model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    )
    if metadata.is_file():
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        return first_line or None
    if model_path.parent.name == "snapshots":
        revision = model_path.name.lower()
        if len(revision) == 40 and all(
            (character in "0123456789abcdef" for character in revision)
        ):
            return revision
    return None


def _model_metadata(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    assert not (not config_path.is_file() or not index_path.is_file()), (
        "model must contain config.json and model.safetensors.index.json"
    )
    revision = _hf_download_revision(model_path)
    assert not revision != MODEL_REVISION, (
        f"expected pinned {MODEL_LABEL} revision {MODEL_REVISION}, found {revision!r}"
    )
    return {
        "path": str(model_path),
        "huggingface_repo": MODEL_REPO,
        "revision": revision,
        "config_sha256": _sha256(config_path),
        "safetensors_index_sha256": _sha256(index_path),
    }


def _validate_config(config: Any) -> None:
    text_config = getattr(config, "text_config", config)
    head_dim = int(
        getattr(text_config, "head_dim", 0)
        or text_config.hidden_size // text_config.num_attention_heads
    )
    observed = (
        str(text_config.model_type),
        int(text_config.num_hidden_layers),
        int(text_config.hidden_size),
        int(text_config.num_attention_heads),
        int(text_config.num_key_value_heads),
        head_dim,
    )
    expected = (
        MODEL_TYPE,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    assert not observed != expected, (
        f"expected {MODEL_LABEL} GQA geometry {expected}, found {observed}"
    )


def _document_id(row: Mapping[str, Any], stream_index: int) -> str:
    for key in ("id", "url", "timestamp"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    digest = hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}:stream:{stream_index}"


def _prepare_windows(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    assert not output_dir.exists(), output_dir
    _validate_config(AutoConfig.from_pretrained(str(model_path), local_files_only=True))
    model_metadata = _model_metadata(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    requested_revision = args.dataset_revision
    dataset_revision = (
        HfApi().dataset_info("allenai/c4", revision=requested_revision).sha
    )
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split=args.dataset_split,
        streaming=True,
        revision=dataset_revision,
        cache_dir=args.dataset_cache,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    generator = random.Random(args.seed)
    seen_ids: set[str] = set()
    windows: list[Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stream_index, row in enumerate(dataset):
        document_id = _document_id(row, stream_index)
        if document_id in seen_ids:
            continue
        input_ids = tokenizer(
            str(row["text"]), return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if input_ids.numel() < args.sequence_length:
            continue
        token_start = generator.randint(0, input_ids.numel() - args.sequence_length)
        window = input_ids[token_start : token_start + args.sequence_length].to(
            torch.int32
        )
        windows.append(window.cpu().contiguous())
        records.append(
            {
                "sample_index": len(windows) - 1,
                "document_id": document_id,
                "stream_index": stream_index,
                "token_start": token_start,
                "document_token_count": int(input_ids.numel()),
                "input_ids_sha256": _tensor_sha256(window),
            }
        )
        seen_ids.add(document_id)
        if len(windows) == args.samples:
            break
    assert not len(windows) != args.samples, (
        f"C4 stream yielded only {len(windows)} usable unique documents"
    )
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "windows.safetensors"
    stacked = torch.stack(windows)
    _atomic_safetensors(artifact_path, {"input_ids": stacked})
    manifest = {
        "format": WINDOWS_FORMAT,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer)},
        "dataset": {
            "repo": "allenai/c4",
            "config": "en",
            "split": args.dataset_split,
            "requested_revision": requested_revision,
            "resolved_revision": dataset_revision,
            "streaming": True,
            "shuffle_buffer": args.shuffle_buffer,
        },
        "sampling": {
            "seed": args.seed,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "document_disjoint": len(seen_ids) == len(windows),
            "add_special_tokens": False,
            "within_document_start": "uniform integer",
        },
        "records": records,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensor": "input_ids",
            "dtype": str(stacked.dtype),
            "shape": list(stacked.shape),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[C4] wrote {artifact_path} shape={tuple(stacked.shape)}", flush=True)


def _load_indexed_tensor(model_path: Path, tensor_name: str) -> Tensor:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"].get(tensor_name)
    assert not shard_name is None, f"tensor absent from model index: {tensor_name}"
    with safe_open(
        str(model_path / shard_name), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(tensor_name)


activate_model_profile("llama31_8b")
