"""Affine Value-to-Key base: fitting from moments, and the feature transforms it feeds.

The base predicts each KV group's pre-RoPE Key from its stored Value codes
with one affine map `codes @ left @ right + bias` of bounded rank. The
residual router (page_fisher) then models what the base misses.
"""

from typing import NamedTuple

import torch


class AffineMap(NamedTuple):
    left: torch.Tensor  # [input, rank]
    right: torch.Tensor  # [rank, output]
    bias: torch.Tensor  # [output]


def _inverse_square_root(covariance):
    """Moore-Penrose inverse square root of a symmetric PSD matrix."""
    covariance = 0.5 * (covariance + covariance.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    cutoff = torch.finfo(covariance.dtype).eps * covariance.shape[0] * eigenvalues.abs().amax()
    tiny = torch.finfo(covariance.dtype).tiny
    roots = torch.where(eigenvalues > cutoff, eigenvalues.clamp_min(tiny).rsqrt(), torch.zeros_like(eigenvalues))
    return (eigenvectors * roots) @ eigenvectors.mT


def fit_affine_map(*, row_count, input_sum, target_sum, input_gram, input_target_gram, rank, fit_bias=True):
    """Exact rank-constrained least squares (reduced-rank regression) from row moments.

    The centered input covariance is whitened with its inverse square root;
    truncating the SVD of the whitened cross moment gives the optimal map of
    the given rank. With `fit_bias=False` the raw moments are used and the bias
    is zero.
    """
    count = float(row_count)
    mean_input, mean_target = input_sum / count, target_sum / count
    covariance, cross = input_gram, input_target_gram
    if fit_bias:
        covariance = covariance - count * torch.outer(mean_input, mean_input)
        cross = cross - count * torch.outer(mean_input, mean_target)
    inverse_root = _inverse_square_root(covariance)
    u, s, vt = torch.linalg.svd(inverse_root @ cross, full_matrices=False)
    rank = min(rank, s.numel())
    left = inverse_root @ u[:, :rank]
    right = s[:rank, None] * vt[:rank]
    bias = mean_target - mean_input @ left @ right if fit_bias else torch.zeros_like(mean_target)
    return AffineMap(left, right, bias)


def fit_base(moments, encoder, rank):
    """One affine map per KV group from Value codes to pre-RoPE Key, in FP64."""
    encoder = encoder.cpu().double()
    maps = []
    for group, f in enumerate(encoder):
        if rank == 0:  # pure residual routing: zero predictor and zero intercept
            left = f.new_zeros(f.shape[-1], 0)
            right = f.new_zeros(0, moments["sum_k"].shape[-1])
            maps.append(AffineMap(left, right, torch.zeros_like(moments["sum_k"][group])))
            continue
        maps.append(fit_affine_map(
            row_count=int(moments["count"]),
            input_sum=moments["sum_v"][group] @ f,
            target_sum=moments["sum_k"][group],
            input_gram=f.mT @ moments["vv"][group] @ f,
            input_target_gram=f.mT @ moments["vk"][group],
            rank=rank,
        ))
    return tuple(maps)


def base_mse(moments, encoder, maps):
    """Squared pre-RoPE Key error of the base over the moments, and its ratio to the Key energy."""
    error = 0.0
    for group, base in enumerate(maps):
        weight = encoder[group].cpu().double() @ base.left.double() @ base.right.double()
        bias = base.bias.double()
        error += float(
            torch.trace(weight.mT @ moments["vv"][group] @ weight)
            - 2 * (weight * moments["vk"][group]).sum()
            + torch.trace(moments["kk"][group])
            + 2 * ((moments["sum_v"][group] @ weight - moments["sum_k"][group]) * bias).sum()
            + int(moments["count"]) * bias.square().sum()
        )
    energy = float(moments["kk"].diagonal(dim1=-2, dim2=-1).sum())
    assert energy > 0 and error >= -1e-8 * energy, "base error is materially negative"
    error = max(error, 0.0)
    return {"squared_error": error, "key_energy": energy, "relative_mse": error / energy}


def restore_base(payload):
    """Per-group maps from a saved base record (`left`, `right`, `bias` stacked over groups)."""
    return tuple(
        AffineMap(payload["left"][group], payload["right"][group], payload["bias"][group])
        for group in range(len(payload["left"]))
    )


def stack_base(maps, device):
    """(left, right, bias) stacked over groups in FP32."""
    return tuple(
        torch.stack([getattr(item, name) for item in maps]).to(device=device, dtype=torch.float32)
        for name in ("left", "right", "bias")
    )


def apply_base(codes, factors):
    left, right, bias = factors
    return torch.einsum("tgi,gir,grd->tgd", codes, left, right) + bias


def align_value_basis(base_left, base_right):
    """Orthogonal Q and the predictor in its first base-rank coordinates (Section 4.1).

    If L = Q_b R, then c L B = (c Q)[:, :base_rank] (R B).
    Fold Q into the Value encoder and Q^T into every head's output decoder.
    """
    rank = base_left.shape[-1]
    rotation, triangular = torch.linalg.qr(base_left, mode="complete")
    return rotation, triangular[..., :rank, :] @ base_right


def value_codes(dense_value, encoder):
    return torch.einsum("tgd,gdr->tgr", dense_value, encoder)


def pre_rope_rows(post_rope, cos, sin):
    return invert_rotary(post_rope.permute(1, 0, 2).unsqueeze(0), cos, sin)[0].permute(1, 0, 2)


def post_rope_rows(pre_rope, cos, sin):
    return apply_rotary(pre_rope.permute(1, 0, 2).unsqueeze(0), cos, sin)[0].permute(1, 0, 2)


def rotate_half(values):
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def apply_rotary(value, cos, sin):
    """RoPE in the dtype of `value` on the configured prefix; Qwen3.5 leaves the suffix unchanged.

    Calibration passes FP32 rows; inference uses the cache dtype.
    """
    width = cos.shape[-1]
    cosine, sine = cos.to(device=value.device, dtype=value.dtype).unsqueeze(1), sin.to(device=value.device, dtype=value.dtype).unsqueeze(1)
    prefix = value[..., :width]
    rotated = prefix * cosine + rotate_half(prefix) * sine
    return torch.cat((rotated, value[..., width:]), -1)


def invert_rotary(value, cos, sin):
    return apply_rotary(value, cos, -sin)


def conditional_routing_query_projector(residual_query_projector):
    """[I, U_residual] so that one dot product with a [base K, residual code] sidecar scores a query."""
    heads, dim, _ = residual_query_projector.shape
    identity = torch.eye(dim, device=residual_query_projector.device, dtype=residual_query_projector.dtype)
    return torch.cat((identity.expand(heads, dim, dim), residual_query_projector), dim=-1).contiguous()


def build_conditional_routing_sidecar(
    codes, exact_post_rope_key, *, base_left, base_right, base_bias, residual_encoder, cos, sin
):
    """[predicted post-RoPE K, residual code] for cache-major [batch, KV heads, tokens, width] inputs.

    The base acts on the stored Value codes before RoPE; the residual code is
    the encoder applied to what the rotated prediction misses.
    """
    like = dict(device=codes.device, dtype=codes.dtype)
    predicted = torch.einsum("bhtv,hvr,hrd->bhtd", codes, base_left.to(**like), base_right.to(**like))
    predicted = predicted + base_bias.to(**like)[None, :, None, :]
    if cos is not None:
        predicted = apply_rotary(predicted, cos, sin)
    residual_code = torch.einsum("bhtd,hdr->bhtr", exact_post_rope_key - predicted, residual_encoder.to(**like))
    return torch.cat((predicted, residual_code), dim=-1).contiguous()
