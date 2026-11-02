"""Router statistics and factor records on disk.

Capture writes one record per layer and stage under `ROOT/<stage>/`: moments
for the base fit, the fitted base, and per-window packed Fisher statistics
under `ROOT/fisher/lNNN/wNNN`. A record is a safetensors file plus a JSON file
of the same stem. The JSON and tensor helpers at the end are shared by the
calibration and evaluation launchers.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core.factor_io import read_layer
from core.page_fisher import FisherStatistics, pack_grams, unpack_grams


def layer_file(root, stage, layer):
    return Path(root) / stage / f"layer_{layer:03d}.safetensors"


def save_record(path, tensors, metadata):
    save_tensors(path, {key: value.cpu().contiguous() for key, value in tensors.items()})
    write_json(path.with_suffix(".json"), {"status": "complete", **metadata})


def load_record(path):
    """(tensors, metadata) of a completed record."""
    metadata = read_json(path.with_suffix(".json"))
    assert metadata["status"] == "complete", f"incomplete record: {path}"
    return load_file(str(path)), metadata


def encoder_for(value_factors, layer, geometry):
    """The layer's Value encoder from a core.fit_value bank, or the identity for dense Values."""
    if value_factors is None:
        return torch.eye(geometry.head_dim, dtype=torch.float32).expand(geometry.num_kv_heads, -1, -1).clone()
    record = read_layer(Path(value_factors), layer, None)
    assert record is not None, f"missing Value factors at layer {layer}"
    return load_file(str(Path(value_factors) / record["artifact"]["file"]))["value_coordinate_encoders"]


def pack_statistics(stats):
    """One window's statistics as a record payload; only floating-point antisymmetry is removed."""
    grams = 0.5 * (stats.grams + stats.grams.mT)
    return {
        "queries": stats.queries.cpu(),
        "packed": pack_grams(grams).cpu(),
        "teacher_energy": torch.tensor(stats.teacher_energy, dtype=torch.float64),
    }


def unpack_statistics(payloads, groups):
    """Join window payloads of one layer in query-major, window-minor order."""
    heads, _, dim = payloads[0]["queries"].shape
    queries = torch.stack([payload["queries"] for payload in payloads], dim=2)
    grams = torch.stack([unpack_grams(payload["packed"], dim) for payload in payloads], dim=2)
    return FisherStatistics(
        queries=queries.reshape(heads, -1, dim),
        grams=grams.reshape(heads, -1, dim, dim),
        head_to_kv_group=torch.arange(heads) // (heads // groups),
        scaling=dim**-0.5,
        teacher_energy=sum(float(payload["teacher_energy"]) for payload in payloads),
    )


def load_fisher_statistics(root, layer):
    """Base record and the fit (and held-out, when captured) Fisher statistics of one layer."""
    root = Path(root)
    base, record = load_record(layer_file(root, "base", layer))
    protocol = record["protocol"]
    groups = base["encoder"].shape[0]
    fit_ids, heldout_ids = list(protocol["fit_ids"]), list(protocol["diagnostic_ids"])
    assert fit_ids and not set(fit_ids) & set(heldout_ids), "fit and held-out windows must be disjoint"
    statistics = {}
    for split, indices in (("fit", fit_ids), ("heldout", heldout_ids)):
        payloads = []
        for index in indices:
            payload, metadata = load_record(root / "fisher" / f"l{layer:03d}" / f"w{index:03d}.safetensors")
            assert metadata["protocol"] == protocol, f"window {index} of layer {layer} comes from another capture"
            assert all(torch.isfinite(value).all() for value in payload.values()), f"non-finite statistics at window {index}"
            payloads.append(payload)
        if payloads:
            statistics[split] = unpack_statistics(payloads, groups)
            assert statistics[split].teacher_energy > 0, f"{split} statistics have zero residual energy"
    return base, record, statistics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic write through a .partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def save_tensors(path, tensors):
    """Atomic safetensors write through a .partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(partial))
    partial.replace(path)
