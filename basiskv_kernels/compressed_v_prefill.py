"""Fused causal GQA prefill with dense Keys and compact Values (Triton).

The kernel streams Key/Value tiles with an online softmax, so grouped K/V heads are never
expanded and the score matrix is never stored. Only the initial, complete prefill is
supported: chunked prefill would need a causal offset.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_kernel(
    query_ptr, key_ptr, value_ptr, output_ptr, sequence_length,
    query_stride_batch, query_stride_head, query_stride_token, query_stride_feature,
    key_stride_batch, key_stride_head, key_stride_token, key_stride_feature,
    value_stride_batch, value_stride_head, value_stride_token, value_stride_feature,
    output_stride_batch, output_stride_head, output_stride_token, output_stride_feature,
    scale: tl.constexpr, QUERY_HEADS: tl.constexpr, HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_QK: tl.constexpr, BLOCK_VALUE: tl.constexpr,
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
        query_ptr + batch_index * query_stride_batch + query_head * query_stride_head
        + query_offsets[:, None] * query_stride_token + qk_offsets[None, :] * query_stride_feature,
        mask=query_mask[:, None] & (qk_offsets[None, :] < QK_DIM), other=0.0)

    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_VALUE), dtype=tl.float32)
    key_stop = tl.minimum(sequence_length, (query_block + 1) * BLOCK_M)

    for key_start in range(0, key_stop, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < sequence_length
        keys = tl.load(
            key_ptr + batch_index * key_stride_batch + kv_head * key_stride_head
            + key_offsets[:, None] * key_stride_token + qk_offsets[None, :] * key_stride_feature,
            mask=key_mask[:, None] & (qk_offsets[None, :] < QK_DIM), other=0.0)
        scores = tl.dot(query, tl.trans(keys)) * scale
        causal_mask = query_mask[:, None] & key_mask[None, :] & (query_offsets[:, None] >= key_offsets[None, :])
        scores = tl.where(causal_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        next_max = tl.where(query_mask, next_max, 0.0)
        previous_scale = tl.where(query_mask, tl.exp(running_max - next_max), 0.0)
        probabilities = tl.where(causal_mask, tl.exp(scores - next_max[:, None]), 0.0)

        values = tl.load(
            value_ptr + batch_index * value_stride_batch + kv_head * value_stride_head
            + key_offsets[:, None] * value_stride_token + value_offsets[None, :] * value_stride_feature,
            mask=key_mask[:, None] & (value_offsets[None, :] < VALUE_DIM), other=0.0)
        accumulator = accumulator * previous_scale[:, None] + tl.dot(probabilities.to(values.dtype), values)
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = next_max

    output = accumulator / running_sum[:, None]
    tl.store(
        output_ptr + batch_index * output_stride_batch + query_head * output_stride_head
        + query_offsets[:, None] * output_stride_token + value_offsets[None, :] * output_stride_feature,
        output, mask=query_mask[:, None] & (value_offsets[None, :] < VALUE_DIM))


def compressed_v_prefill_attention(query, key, value, *, scale=None):
    """query [B, Hq, S, D], key [B, Hkv, S, D], value [B, Hkv, S, R] -> [B, Hq, S, R], FP16 or BF16."""
    batch, query_heads, tokens, qk_dim = query.shape
    kv_heads, value_dim = key.shape[1], value.shape[-1]
    assert tokens == key.shape[2] == value.shape[2], 'prefill covers the whole prompt: Q, K, and V must have the same length'
    output = torch.empty(batch, query_heads, tokens, value_dim, dtype=query.dtype, device=query.device)
    block_m, block_n = 32, 64
    with torch.cuda.device(query.device):
        _prefill_kernel[(triton.cdiv(tokens, block_m), batch * query_heads)](
            query, key, value, output, tokens, *query.stride(), *key.stride(), *value.stride(), *output.stride(),
            scale=qk_dim ** -0.5 if scale is None else float(scale), QUERY_HEADS=query_heads,
            HEADS_PER_KV=query_heads // kv_heads, QK_DIM=qk_dim, VALUE_DIM=value_dim, BLOCK_M=block_m,
            BLOCK_N=block_n, BLOCK_QK=triton.next_power_of_2(qk_dim), BLOCK_VALUE=triton.next_power_of_2(value_dim),
            num_warps=4, num_stages=2)
    return output
