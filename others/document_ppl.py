"""Document-level C4 loss and window loading shared by quality evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F


WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _paired(values):
    checked = [float(value) for value in values]
    mean = statistics.fmean(checked)
    standard_error = (
        statistics.stdev(checked) / math.sqrt(len(checked))
        if len(checked) > 1
        else 0.0
    )
    return {
        "values": checked,
        "mean": mean,
        "paired_standard_error": standard_error,
        "one_standard_error_ucb": mean + standard_error,
    }



def _load_windows(path: Path, *, model_path: Path) -> tuple[Tensor, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WINDOWS_FORMAT:
        raise ValueError("incompatible C4 window manifest")
    if _file_sha256(resolved) != manifest["artifact"]["sha256"]:
        raise ValueError("C4-validation window artifact hash mismatch")
    if manifest["model"]["config_sha256"] != _file_sha256(model_path / "config.json"):
        raise ValueError("C4-validation windows belong to another model config")
    if manifest["dataset"]["split"] != "validation":
        raise ValueError("formal C4 PPL requires the validation split")
    if manifest["sampling"]["samples"] != 128:
        raise ValueError("formal C4 PPL requires exactly 128 documents")
    if manifest["sampling"]["sequence_length"] != 2048:
        raise ValueError("formal C4 PPL requires complete 2048-token windows")
    records = manifest.get("records", ())
    document_ids = [str(row["document_id"]) for row in records]
    if len(records) != 128 or len(document_ids) != len(set(document_ids)):
        raise ValueError("C4 PPL documents are missing or duplicated")
    payload = load_file(str(resolved), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("C4 PPL bank must contain only input_ids")
    input_ids = payload["input_ids"].to(torch.long)
    if tuple(input_ids.shape) != (128, 2048):
        raise ValueError("C4 PPL tensor has an incompatible shape")
    provenance = {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "dataset": manifest["dataset"],
        "sampling": manifest["sampling"],
        "document_ids": document_ids,
    }
    return input_ids.contiguous(), provenance



def _document_nll_from_logits(logits: Tensor, input_ids: Tensor) -> Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("C4 PPL expects batched logits and token ids")
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = input_ids[:, 1:].to(shift_logits.device).contiguous()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(len(input_ids), -1)
    return losses.mean(dim=1)



@torch.inference_mode()
def _evaluate_document_ppl(
    model: nn.Module,
    sequences: Tensor,
    *,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    input_device = model.get_input_embeddings().weight.device
    document_nll = []
    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(input_device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        values = _document_nll_from_logits(logits, input_ids)
        if not bool(torch.isfinite(values).all().cpu()):
            raise FloatingPointError(f"non-finite C4 NLL for {label} at document {start}")
        document_nll.extend(map(float, values.cpu().tolist()))
        stop = min(start + batch_size, len(sequences))
        if stop % 16 == 0 or stop == len(sequences):
            print(f"[C4 PPL] {label} documents={stop}/{len(sequences)}", flush=True)
        del input_ids, logits, values
    paired = _paired(document_nll)
    token_count = len(sequences) * (int(sequences.shape[1]) - 1)
    nll_sum = sum(document_nll) * (int(sequences.shape[1]) - 1)
    return {
        "documents": len(sequences),
        "tokens": token_count,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / token_count,
        "ppl": math.exp(nll_sum / token_count),
        "document_nll": paired,
        "loss_dtype": "float32",
    }
