"""Quality evaluation helpers for dense, C1 and weight-SVD checkpoints."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from safetensors.torch import load_file
from torch import Tensor, nn

SUPPORTED_CHECKPOINT_FORMATS = {
    f"scripts.{profile}.iclr_v_factors.v1"
    for profile in ("llama2_7b", "llama31_8b", "llama31_70b", "qwen3_8b", "qwen3_32b")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    assert not layers is None, (
        f"expected a Llama/Qwen-style decoder, found {type(model).__name__}"
    )
    return layers


def _load_checkpoint(
    checkpoint_dir: Path, model_path: Path
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    manifest_path = checkpoint_dir / "manifest.json"
    assert not not manifest_path.is_file(), manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest.get("format") not in SUPPORTED_CHECKPOINT_FORMATS, (
        f"unsupported Weight-SVD checkpoint format: {manifest.get('format')!r}"
    )
    assert not manifest.get("status") != "complete", (
        "Weight-SVD checkpoint is not marked complete"
    )
    assert not manifest["model"]["config_sha256"] != _sha256(
        model_path / "config.json"
    ), "Weight-SVD checkpoint belongs to another model config"
    artifact_path = checkpoint_dir / manifest["artifact"]["file"]
    assert not _sha256(artifact_path) != manifest["artifact"]["sha256"], (
        "Weight-SVD factor artifact hash changed"
    )
    payload = load_file(str(artifact_path), device="cpu")
    assert not len(payload) != int(manifest["artifact"]["tensor_count"]), (
        "Weight-SVD tensor count differs from the manifest"
    )
    return (manifest, payload)
