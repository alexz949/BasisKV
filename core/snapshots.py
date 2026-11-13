"""Calibration snapshots: manifests, per-layer payloads, and covariance blocks.

`calibration.value` writes one covariance snapshot per model: a
manifest plus, per attention layer, the normalized fit and held-out Gram
matrices of the output-projection input and the dense o_proj weight.
"""

import json

import torch
from safetensors.torch import load_file

COVARIANCE_FORMAT = "basiskv.attention_o_proj_covariances.v1"


def read_manifest(snapshot_dir):
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("format") == COVARIANCE_FORMAT, f"incompatible snapshot format in {snapshot_dir}"
    return manifest


def load_layer(snapshot_dir, manifest, layer):
    """Layer payload (weight, fit_covariance, heldout_covariance) plus its source record."""
    record = manifest["artifacts"][str(layer)]
    path = snapshot_dir / record["file"]
    payload = {key: value.contiguous() for key, value in load_file(str(path), device="cpu").items()}
    return payload, {"path": str(path), **record}


def head_targets(weight, geometry, device):
    """o_proj weight as per-head targets W[h] of shape [d, o]."""
    return (
        weight.to(device=device, dtype=torch.float64)
        .transpose(0, 1)
        .reshape(geometry.num_heads, geometry.head_dim, geometry.hidden_size)
        .contiguous()
    )


def covariance_blocks(gram, geometry):
    """[H d, H d] Gram matrix as symmetric [H, H, d, d] blocks."""
    heads, dim = geometry.num_heads, geometry.head_dim
    blocks = gram.reshape(heads, dim, heads, dim).permute(0, 2, 1, 3).contiguous()
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


@torch.no_grad()
def stored_covariance(gram, geometry, device):
    """Stored Gram matrix as blocks, with its finite-precision negative spectrum removed."""
    work = gram.to(device=device, dtype=torch.float64)
    work = 0.5 * (work + work.T)
    _, info = torch.linalg.cholesky_ex(work, check_errors=False)
    if int(info) == 0:
        return covariance_blocks(work, geometry), {"applied": False, "negative_eigenvalue_count": 0}
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    negative = eigenvalues < 0
    repaired = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T
    negative_norm = torch.linalg.vector_norm(eigenvalues[negative])
    total_norm = torch.linalg.vector_norm(eigenvalues).clamp_min(torch.finfo(torch.float64).tiny)
    diagnostics = {
        "applied": True,
        "method": "fp64_eigenvalue_clip",
        "minimum_eigenvalue_before": float(eigenvalues[0]),
        "maximum_eigenvalue_before": float(eigenvalues[-1]),
        "negative_eigenvalue_count": int(negative.sum()),
        "relative_spectral_correction_frobenius": float(negative_norm / total_norm),
    }
    return covariance_blocks(0.5 * (repaired + repaired.T), geometry), diagnostics
