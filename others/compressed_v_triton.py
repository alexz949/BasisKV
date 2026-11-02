"""Dense-Key, compact-Value Triton prefill and single-token decode.

Both kernels stream attention without expanding grouped K/V heads or storing
the full score matrix. Historical resident-CUDA experiments live in others.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _compressed_v_gqa_decode_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    valid_sequence_length_ptr,
    valid_sequence_length_value,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_feature,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_feature,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_feature,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    scale: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    BLOCK_SEQUENCE: tl.constexpr,
    CACHE_LENGTH: tl.constexpr,
    VALID_LENGTH_IS_POINTER: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV

    qk_offsets = tl.arange(0, BLOCK_QK)
    value_offsets = tl.arange(0, BLOCK_VALUE)
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + 0 * query_stride_token
        + qk_offsets * query_stride_feature,
        mask=qk_offsets < QK_DIM,
        other=0.0,
    ).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_VALUE,), dtype=tl.float32)

    if VALID_LENGTH_IS_POINTER:
        valid_sequence_length = tl.load(valid_sequence_length_ptr)
    else:
        valid_sequence_length = valid_sequence_length_value

    for sequence_start in range(0, CACHE_LENGTH, BLOCK_SEQUENCE):
        sequence_offsets = sequence_start + tl.arange(0, BLOCK_SEQUENCE)
        sequence_mask = sequence_offsets < valid_sequence_length
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + sequence_offsets[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=sequence_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(sequence_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)

        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + sequence_offsets[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=sequence_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
        running_max = next_max

    output = accumulator / running_sum
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + 0 * output_stride_token
        + value_offsets * output_stride_feature,
        output,
        mask=value_offsets < VALUE_DIM,
    )


@triton.jit
def _compressed_v_gqa_prefill_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    sequence_length,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_feature,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_feature,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_feature,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    scale: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_index = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    qk_offsets = tl.arange(0, BLOCK_QK)
    value_offsets = tl.arange(0, BLOCK_VALUE)
    query_mask = query_offsets < sequence_length
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + query_offsets[:, None] * query_stride_token
        + qk_offsets[None, :] * query_stride_feature,
        mask=query_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_VALUE), dtype=tl.float32)
    key_stop = tl.minimum(sequence_length, (query_block + 1) * BLOCK_M)

    for key_start in range(0, key_stop, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < sequence_length
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + key_offsets[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=key_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale
        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (query_offsets[:, None] >= key_offsets[None, :])
        )
        scores = tl.where(causal_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        next_max = tl.where(query_mask, next_max, 0.0)
        previous_scale = tl.where(
            query_mask,
            tl.exp(running_max - next_max),
            0.0,
        )
        probabilities = tl.where(
            causal_mask,
            tl.exp(scores - next_max[:, None]),
            0.0,
        )

        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + key_offsets[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=key_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        )
        accumulator = accumulator * previous_scale[:, None] + tl.dot(
            probabilities.to(values.dtype),
            values,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = next_max

    output = accumulator / running_sum[:, None]
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + query_offsets[:, None] * output_stride_token
        + value_offsets[None, :] * output_stride_feature,
        output,
        mask=query_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
    )


def _validate_inputs(query: Tensor, key: Tensor, value: Tensor) -> tuple[int, ...]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("compressed-V attention Q/K/V must be rank-4 tensors")
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    key_batch, kv_heads, sequence_length, key_dim = map(int, key.shape)
    value_batch, value_heads, value_sequence, value_dim = map(int, value.shape)
    if batch != key_batch or batch != value_batch:
        raise ValueError("compressed-V attention Q/K/V batch dimensions differ")
    if kv_heads != value_heads or sequence_length != value_sequence:
        raise ValueError(
            "compressed-V attention K/V head or sequence dimensions differ"
        )
    if qk_dim != key_dim:
        raise ValueError("compressed-V attention Q/K head dimensions differ")
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    assert sequence_length > 0 and 0 < value_dim <= 256, (
        "cache sequence must be positive and compressed Value width must lie in [1, 256]"
    )
    if qk_dim <= 0 or qk_dim > 256:
        raise ValueError("Q/K head dimension must lie in [1, 256]")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("compressed-V decode Q/K/V dtypes differ")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("compressed-V attention supports FP16 and BF16")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("compressed-V attention requires CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("compressed-V attention Q/K/V devices differ")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("compressed-V attention feature dimensions must be contiguous")
    return (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    )


def compressed_v_decode_attention_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    valid_sequence_length: Tensor | None = None,
) -> Tensor:
    """Run online-softmax GQA with a one-token query and compact Values.

    Shapes are ``query=[B,Hq,1,D]``, ``key=[B,Hkv,S,D]``, and
    ``value=[B,Hkv,S,R]``.  The result is ``[B,Hq,1,R]``. When supplied,
    ``valid_sequence_length`` is a CUDA scalar read by the kernel so one
    fixed-cache launch shape can be replayed at changing decode positions.
    """

    (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    ) = _validate_inputs(query, key, value)
    if query_tokens != 1:
        raise ValueError("compressed-V Triton decode accepts one query token")
    if valid_sequence_length is not None:
        if (
            valid_sequence_length.ndim != 0
            or valid_sequence_length.dtype not in (torch.int32, torch.int64)
            or not valid_sequence_length.is_cuda
            or valid_sequence_length.device != query.device
        ):
            raise ValueError(
                "valid sequence length must be a CUDA int32/int64 scalar on the QKV device"
            )
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

    output = torch.empty(
        batch,
        query_heads,
        1,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    block_qk = triton.next_power_of_2(qk_dim)
    block_value = triton.next_power_of_2(value_dim)
    block_sequence = 64 if value_dim <= 64 else 32
    with torch.cuda.device(query.device):
        _compressed_v_gqa_decode_kernel[(batch * query_heads,)](
            query,
            key,
            value,
            output,
            key if valid_sequence_length is None else valid_sequence_length,
            sequence_length,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            scale=selected_scale,
            QUERY_HEADS=query_heads,
            HEADS_PER_KV=query_heads // kv_heads,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            BLOCK_QK=block_qk,
            BLOCK_VALUE=block_value,
            BLOCK_SEQUENCE=block_sequence,
            CACHE_LENGTH=sequence_length,
            VALID_LENGTH_IS_POINTER=valid_sequence_length is not None,
            num_warps=4,
            num_stages=2,
        )
    return output


def compressed_v_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Run fused causal GQA for a complete prompt with compact Values.

    Shapes are ``query=[B,Hq,S,D]``, ``key=[B,Hkv,S,D]``, and
    ``value=[B,Hkv,S,R]``. This deliberately accepts only the initial complete
    prefill. Chunked prefill would require a causal offset and is outside the
    benchmark protocol.
    """

    (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    ) = _validate_inputs(query, key, value)
    if query_tokens <= 1 or query_tokens != sequence_length:
        raise ValueError(
            "compressed-V prefill requires matching Q/K lengths greater than one"
        )
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")
    output = torch.empty(
        batch,
        query_heads,
        sequence_length,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    block_m = 32
    block_n = 64
    block_qk = triton.next_power_of_2(qk_dim)
    block_value = triton.next_power_of_2(value_dim)
    with torch.cuda.device(query.device):
        _compressed_v_gqa_prefill_kernel[
            (triton.cdiv(sequence_length, block_m), batch * query_heads)
        ](
            query,
            key,
            value,
            output,
            sequence_length,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            scale=selected_scale,
            QUERY_HEADS=query_heads,
            HEADS_PER_KV=query_heads // kv_heads,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_QK=block_qk,
            BLOCK_VALUE=block_value,
            num_warps=4,
            num_stages=2,
        )
    return output


__all__ = [
    "compressed_v_decode_attention_triton",
    "compressed_v_prefill_attention",
]
