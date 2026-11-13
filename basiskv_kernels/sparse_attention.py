"""Split-K decode with token selections shared by a query head or KV group.

Q: [batch, query_heads, 1, head_dim]. K/V: [batch, kv_heads, capacity, width].
Selected token IDs always address V. Optional key_slots address a persistent K
cache independently. Workspace is (partial FP32 outputs, FP32 LSE, final output).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_decode(
    Q, K, V, IDS, SLOTS, PARTIAL, LSE,
    QS0: tl.constexpr, QS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
    HEADS: tl.constexpr, GROUP: tl.constexpr, SHARED_IDS: tl.constexpr,
    HAS_SLOTS: tl.constexpr, KEY_CAPACITY: tl.constexpr, VALUE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, SELECTED: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr, SPLITS: tl.constexpr,
    PER_SPLIT: tl.constexpr, SCALE: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head = row // HEADS, row % HEADS
    kv_head = head // GROUP
    selection_head = kv_head if SHARED_IDS else head
    dimensions, value_dimensions = tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_V)
    query = tl.load(Q + batch * QS0 + head * QS1 + dimensions,
                    dimensions < HEAD_DIM, 0).to(tl.float32)
    maximum = -float('inf')
    denominator = 0.0
    accumulator = tl.zeros((BLOCK_V,), tl.float32)

    # Each split streams disjoint selected tokens, maintaining an online softmax.
    for start in range(split * PER_SPLIT, (split + 1) * PER_SPLIT, 32):
        positions = start + tl.arange(0, 32)
        tokens = tl.load(IDS + batch * IS0 + selection_head * IS1 + positions * IS2,
                         positions < SELECTED, -1)
        if HAS_SLOTS:
            slots = tl.load(SLOTS + batch * SS0 + selection_head * SS1 + positions * SS2,
                            positions < SELECTED, -1)
        else:
            slots = tokens
        valid = ((positions < SELECTED) & (tokens >= 0) & (tokens < VALUE_CAPACITY)
                 & (slots >= 0) & (slots < KEY_CAPACITY))
        keys = tl.load(K + batch * KS0 + kv_head * KS1 + slots[:, None] * KS2 + dimensions[None, :],
                       valid[:, None] & (dimensions[None, :] < HEAD_DIM), 0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float('inf'))
        any_valid = tl.sum(valid.to(tl.int32), 0) > 0
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        next_maximum = tl.where(any_valid, next_maximum, maximum)
        correction = tl.where(any_valid, tl.exp(maximum - next_maximum), 1.0)
        probabilities = tl.where(valid, tl.exp(scores - next_maximum), 0.0)
        values = tl.load(V + batch * VS0 + kv_head * VS1 + tokens[:, None] * VS2 + value_dimensions[None, :],
                         valid[:, None] & (value_dimensions[None, :] < VALUE_DIM), 0).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * correction + tl.sum(probabilities, 0)
        maximum = next_maximum

    result = accumulator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(PARTIAL + (row * SPLITS + split) * VALUE_DIM + value_dimensions,
             result, value_dimensions < VALUE_DIM)
    tl.store(LSE + row * SPLITS + split,
             tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')))


@triton.jit
def _merge_splits(PARTIAL, LSE, OUT, HEADS: tl.constexpr, VALUE_DIM: tl.constexpr,
                  BLOCK_V: tl.constexpr, SPLITS: tl.constexpr,
                  OS0: tl.constexpr, OS1: tl.constexpr, OS3: tl.constexpr):
    row = tl.program_id(0)
    splits, dimensions = tl.arange(0, SPLITS), tl.arange(0, BLOCK_V)
    logs = tl.load(LSE + row * SPLITS + splits)
    maximum = tl.max(logs, 0)
    # Empty splits contribute zero; an entirely masked selection returns zero.
    weights = tl.where(logs > -float('inf'), tl.exp(logs - maximum), 0.0)
    total = tl.sum(weights, 0)
    weights /= tl.where(total > 0, total, 1.0)
    values = tl.load(PARTIAL + (row * SPLITS + splits[:, None]) * VALUE_DIM + dimensions[None, :],
                     dimensions[None, :] < VALUE_DIM, 0)
    output = tl.sum(values * weights[:, None], 0)
    tl.store(OUT + (row // HEADS) * OS0 + (row % HEADS) * OS1 + dimensions * OS3,
             output, dimensions < VALUE_DIM)


def sparse_attention(query, key, value, selected_ids, *, key_slots=None, workspace=None, scale):
    batch, heads, tokens, dim = query.shape
    groups, rank = key.shape[1], value.shape[-1]
    assert tokens == 1 and heads % groups == 0
    assert key.shape[:2] == value.shape[:2] == (batch, groups) and key.shape[-1] == dim
    assert query.is_cuda and query.device == key.device == value.device == selected_ids.device
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    assert selected_ids.dtype == torch.int64 and selected_ids.shape[0] == batch
    assert selected_ids.shape[1] in (groups, heads) and selected_ids.shape[2] > 0
    if key_slots is None:
        assert key.shape[2] == value.shape[2]
    else:
        assert key_slots.shape == selected_ids.shape and key_slots.dtype == torch.int64
        assert key_slots.device == query.device
    selected = selected_ids.shape[-1]
    if workspace is None:
        splits = min(16, triton.next_power_of_2(triton.cdiv(selected, 128)))
        workspace = (torch.empty(batch, heads, splits, rank, device=query.device, dtype=torch.float32),
                     torch.empty(batch, heads, splits, device=query.device, dtype=torch.float32),
                     torch.empty(batch, heads, 1, rank, device=query.device, dtype=query.dtype))
    partial, lse, output = workspace
    splits = partial.shape[2]
    assert 0 < splits <= 16 and splits & (splits - 1) == 0
    assert partial.shape == (batch, heads, splits, rank) and lse.shape == (batch, heads, splits)
    assert partial.is_contiguous() and lse.is_contiguous()
    assert partial.dtype == lse.dtype == torch.float32 and output.dtype == query.dtype
    assert partial.device == lse.device == output.device == query.device
    assert output.shape == (batch, heads, 1, rank)
    slots = selected_ids if key_slots is None else key_slots
    per_split = triton.cdiv(triton.cdiv(selected, splits), 32) * 32
    with torch.cuda.device(query.device):
        _sparse_decode[(batch * heads, splits)](
            query, key, value, selected_ids, slots, partial, lse,
            *query.stride()[:2], *key.stride()[:3], *value.stride()[:3],
            *selected_ids.stride(), *slots.stride(), HEADS=heads, GROUP=heads // groups,
            SHARED_IDS=selected_ids.shape[1] == groups, HAS_SLOTS=key_slots is not None,
            KEY_CAPACITY=key.shape[2], VALUE_CAPACITY=value.shape[2], HEAD_DIM=dim, VALUE_DIM=rank,
            SELECTED=selected, BLOCK_D=triton.next_power_of_2(dim), BLOCK_V=triton.next_power_of_2(rank),
            SPLITS=splits, PER_SPLIT=per_split, SCALE=scale, num_warps=4)
        _merge_splits[(batch * heads,)](partial, lse, output, HEADS=heads, VALUE_DIM=rank,
            BLOCK_V=triton.next_power_of_2(rank), SPLITS=splits,
            OS0=output.stride(0), OS1=output.stride(1), OS3=output.stride(3), num_warps=4)
    return output
