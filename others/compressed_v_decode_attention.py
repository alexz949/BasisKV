"""Historical resident-CUDA decode, Page64 staging, and attention references."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from others.compressed_v_triton import _validate_inputs


_CUDA_DECODE_CONTEXT_LIMITS = (128, 512, 1024, 4096, 8192)


_CUDA_DECODE_TUNING: dict[
    tuple[int, int],
    dict[int, tuple[int, ...]],
] = {
    (8, 9): {
        1: (16, 16, 32, 32, 32),
        2: (4, 16, 32, 32, 32),
        4: (4, 16, 16, 16, 16),
        8: (8, 8, 8, 32, 32),
        16: (4, 4, 16, 32, 16),
        32: (1, 8, 8, 16, 16),
        64: (1, 4, 8, 8, 8),
        128: (1, 1, 4, 4, 16),
    },
}


def select_compressed_v_decode_cuda_splits(
    *,
    capability: tuple[int, int],
    batch: int,
    sequence_length: int,
) -> int:
    """Return the offline-tuned split count for an explicitly CUDA request."""

    if batch <= 0 or sequence_length <= 0:
        raise ValueError("decode batch and sequence length must be positive")
    if sequence_length < _CUDA_DECODE_CONTEXT_LIMITS[0]:
        return 1
    architecture_table = _CUDA_DECODE_TUNING.get(tuple(map(int, capability)))
    if architecture_table is None:
        major, minor = capability
        raise RuntimeError(
            f"no pure-CUDA tuning table is installed for SM{major}{minor}"
        )
    batch_table = architecture_table.get(int(batch))
    if batch_table is None:
        raise ValueError(f"no pure-CUDA tuning is installed for batch {batch}")
    for context_index, context_limit in enumerate(_CUDA_DECODE_CONTEXT_LIMITS):
        if sequence_length <= context_limit:
            return batch_table[context_index]
    raise ValueError(
        f"pure-CUDA decode is tuned through context {_CUDA_DECODE_CONTEXT_LIMITS[-1]}"
    )


@dataclass(frozen=True)
class CompressedVDecodeWorkspace:
    """One flat split-K arena shared by sequential transformer layers."""

    batch: int
    query_heads: int
    max_splits: int
    max_value_dim: int
    workspace_storage: Tensor
    output_storage: Tensor

    @classmethod
    def allocate(
        cls,
        *,
        batch: int,
        query_heads: int,
        max_value_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        max_splits: int = 32,
    ) -> CompressedVDecodeWorkspace:
        rows = int(batch) * int(query_heads)
        if rows <= 0 or max_value_dim <= 0 or max_splits <= 0:
            raise ValueError("compressed-V workspace dimensions must be positive")
        return cls(
            batch=int(batch),
            query_heads=int(query_heads),
            max_splits=int(max_splits),
            max_value_dim=int(max_value_dim),
            workspace_storage=torch.empty(
                rows * int(max_splits) * (int(max_value_dim) + 2),
                dtype=torch.float32,
                device=device,
            ),
            output_storage=torch.empty(
                rows * int(max_value_dim),
                dtype=dtype,
                device=device,
            ),
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.workspace_storage, self.output_storage)
        )

    def views(
        self,
        *,
        batch: int,
        query_heads: int,
        splits: int,
        value_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if int(batch) != self.batch or int(query_heads) != self.query_heads:
            raise ValueError("decode inputs differ from the configured workspace")
        if not 0 < splits <= self.max_splits or not 0 < value_dim <= self.max_value_dim:
            raise ValueError("decode request exceeds the configured workspace")
        if self.output_storage.dtype != dtype or self.output_storage.device != device:
            raise ValueError("decode workspace dtype or device differs from its inputs")
        rows = self.batch * self.query_heads
        workspace_elements = rows * int(splits) * (int(value_dim) + 2)
        output_elements = rows * int(value_dim)
        workspace = self.workspace_storage[:workspace_elements].view(
            rows,
            int(splits),
            int(value_dim) + 2,
        )
        output = self.output_storage[:output_elements].view(
            self.batch,
            self.query_heads,
            1,
            int(value_dim),
        )
        return workspace, output


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return tuple(map(int, torch.cuda.get_device_capability(device_index)))


def compressed_v_decode_attention_cuda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    splits: int = 1,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
    feature_major_output: Tensor | None = None,
) -> Tensor:
    """Run the exact-rank split-K CUDA decode.

    ``workspace`` and the selected output can be allocated once and reused so
    timed serving calls contain only the partial and reduction kernels. Passing
    ``feature_major_output=[query_heads * rank, batch]`` makes attention write
    directly into a packed collective slot.
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
        raise ValueError("compressed-V CUDA decode accepts one query token")
    if qk_dim != 128:
        raise ValueError("compressed-V CUDA requires QK dim 128")
    if value_dim not in (32, 48, 64, 80, 96, 112):
        raise ValueError("compressed-V CUDA requires rank in {32,48,64,80,96,112}")
    if splits not in (1, 2, 4, 8, 16, 32):
        raise ValueError("splits must be in {1,2,4,8,16,32}")
    if sequence_length < splits:
        raise ValueError("each CUDA split must contain at least one KV token")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

    expected_workspace = (batch * query_heads, splits, value_dim + 2)
    if workspace is None:
        workspace = torch.empty(
            expected_workspace,
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        tuple(workspace.shape) != expected_workspace
        or workspace.dtype != torch.float32
        or workspace.device != query.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("invalid CUDA split-K workspace")
    if output is not None and feature_major_output is not None:
        raise ValueError("select either token-major or feature-major CUDA output")
    uses_feature_major_output = feature_major_output is not None
    if uses_feature_major_output:
        output = feature_major_output
        expected_output = (query_heads * value_dim, batch)
    else:
        expected_output = (batch, query_heads, 1, value_dim)
        if output is None:
            output = torch.empty(
                expected_output,
                dtype=query.dtype,
                device=query.device,
            )
    assert output is not None
    if (
        tuple(output.shape) != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        layout = "feature-major" if uses_feature_major_output else "token-major"
        raise ValueError(f"invalid {layout} CUDA split-K output")

    from others.compressed_v_decode_cuda import launch_compressed_v_decode_cuda

    return launch_compressed_v_decode_cuda(
        query,
        key,
        value,
        workspace,
        output,
        scale=selected_scale,
        splits=splits,
        feature_major_output=uses_feature_major_output,
    )


def c1_dense_gqa_v96_decode_attention_cuda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    valid_sequence_length: Tensor,
    *,
    scale: float | None = None,
    splits: int = 32,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Scan dense exact-K/C1-V96 once per physical GQA group.

    Shapes are ``query=[B,4*Hkv,1,128]``, ``key=[B,Hkv,S,128]``, and
    ``value=[B,Hkv,S,96]``. ``valid_sequence_length`` is a device scalar, so a
    fixed-capacity cache can advance without a host synchronization. Four
    Query-head warps share each K/V tile before split-K reduction.
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
    if (
        query_tokens != 1
        or qk_dim != 128
        or value_dim != 96
        or query_heads != 4 * kv_heads
    ):
        raise ValueError("dense GQA V96 CUDA requires QK128, C1-V96, and GQA ratio 4")
    if (
        valid_sequence_length.ndim != 0
        or valid_sequence_length.dtype != torch.int64
        or not valid_sequence_length.is_cuda
        or valid_sequence_length.device != query.device
        or not valid_sequence_length.is_contiguous()
    ):
        raise ValueError(
            "valid sequence length must be a contiguous CUDA int64 scalar "
            "on the QKV device"
        )
    if splits not in (1, 2, 4, 8, 16, 32, 64, 128, 256) or sequence_length < splits:
        raise ValueError("dense GQA V96 splits/cache length are incompatible")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("dense GQA V96 scale must be finite and positive")

    expected_workspace = (batch * query_heads, splits, value_dim + 2)
    if workspace is None:
        workspace = torch.empty(
            expected_workspace,
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        tuple(workspace.shape) != expected_workspace
        or workspace.dtype != torch.float32
        or workspace.device != query.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("invalid dense GQA V96 split-K workspace")
    expected_output = (batch, query_heads, 1, value_dim)
    if output is None:
        output = torch.empty(
            expected_output,
            dtype=query.dtype,
            device=query.device,
        )
    elif (
        tuple(output.shape) != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("invalid dense GQA V96 output")

    from others.compressed_v_decode_cuda import launch_c1_dense_gqa_v96_decode_cuda

    return launch_c1_dense_gqa_v96_decode_cuda(
        query,
        key,
        value,
        valid_sequence_length,
        workspace,
        output,
        scale=selected_scale,
        splits=splits,
    )


def c1_paged_sparse_decode_attention_cuda(
    query: Tensor,
    packed_key_pages: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    scale: float | None = None,
    splits: int = 16,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Attend to packed exact-K pages and matching resident C1-V96 pages.

    ``packed_key_pages=[B,Hkv,P,64,128]`` is the GPU staging layout produced
    by an exact-Key page fetch. ``selected_page_ids=[B,Hkv,P]`` maps each
    staging slot back to its page in resident ``value=[B,Hkv,S,96]``; ``-1``
    marks padded staging slots. Four Query heads share every physical KV page.
    """

    if (
        query.ndim != 4
        or packed_key_pages.ndim != 5
        or value.ndim != 4
        or selected_page_ids.ndim != 3
    ):
        raise ValueError("paged sparse Q/K/V/page IDs have incompatible ranks")
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    key_batch, kv_heads, page_slots, page_size, key_dim = map(
        int, packed_key_pages.shape
    )
    value_batch, value_heads, sequence_length, value_dim = map(int, value.shape)
    if (
        query_tokens != 1
        or qk_dim != 128
        or key_dim != 128
        or page_size != 64
        or value_dim != 96
        or query_heads != 4 * kv_heads
    ):
        raise ValueError(
            "paged sparse CUDA requires QK128, C1-V96, page64, and GQA ratio 4"
        )
    if (
        key_batch != batch
        or value_batch != batch
        or value_heads != kv_heads
        or tuple(selected_page_ids.shape) != (batch, kv_heads, page_slots)
        or page_slots <= 0
        or sequence_length <= 0
    ):
        raise ValueError("paged sparse cache geometry is incompatible")
    tensors = (query, packed_key_pages, value)
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise TypeError("paged sparse Q/K/V dtypes differ")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("paged sparse CUDA supports FP16 and BF16")
    if not all(tensor.is_cuda for tensor in (*tensors, selected_page_ids)):
        raise ValueError("paged sparse attention requires CUDA tensors")
    if any(tensor.device != query.device for tensor in (*tensors, selected_page_ids)):
        raise ValueError("paged sparse tensors must share one CUDA device")
    if selected_page_ids.dtype != torch.int64 or not selected_page_ids.is_contiguous():
        raise TypeError("selected page IDs must be contiguous int64")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("paged sparse feature dimensions must be contiguous")
    if splits not in (1, 2, 4, 8, 16, 32):
        raise ValueError("paged sparse splits must be in {1,2,4,8,16,32}")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("paged sparse attention scale must be finite and positive")

    expected_workspace = (batch * query_heads, splits, value_dim + 2)
    if workspace is None:
        workspace = torch.empty(
            expected_workspace,
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        tuple(workspace.shape) != expected_workspace
        or workspace.dtype != torch.float32
        or workspace.device != query.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("invalid paged sparse split-K workspace")
    expected_output = (batch, query_heads, 1, value_dim)
    if output is None:
        output = torch.empty(
            expected_output,
            dtype=query.dtype,
            device=query.device,
        )
    elif (
        tuple(output.shape) != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("invalid paged sparse output")

    from others.compressed_v_decode_cuda import launch_c1_paged_sparse_decode_cuda

    return launch_c1_paged_sparse_decode_cuda(
        query,
        packed_key_pages,
        value,
        selected_page_ids,
        workspace,
        output,
        scale=selected_scale,
        splits=splits,
    )


def c1_pack_exact_key_pages_cuda(
    exact_key: Tensor,
    selected_page_ids: Tensor,
    *,
    output: Tensor | None = None,
) -> Tensor:
    """Pack resident exact K into ``[B,Hkv,P,64,128]`` staging pages."""

    batch, kv_heads, _, head_dim = map(int, exact_key.shape)
    page_slots = int(selected_page_ids.shape[-1])
    if output is None:
        output = torch.empty(
            batch,
            kv_heads,
            page_slots,
            64,
            head_dim,
            dtype=exact_key.dtype,
            device=exact_key.device,
        )
    from others.compressed_v_decode_cuda import launch_c1_pack_exact_key_pages_cuda

    return launch_c1_pack_exact_key_pages_cuda(
        exact_key,
        selected_page_ids,
        output,
    )


def compressed_v_decode_attention_cuda_tuned(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    workspace: CompressedVDecodeWorkspace,
    scale: float | None = None,
    feature_major_output: Tensor | None = None,
) -> Tensor:
    """Run only CUDA kernels using an offline-tuned split configuration."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("compressed-V attention Q/K/V must be rank-4 tensors")
    batch = int(query.shape[0])
    query_heads = int(query.shape[1])
    sequence_length = int(key.shape[2])
    value_dim = int(value.shape[3])
    device_index = query.device.index
    selected_device = (
        torch.cuda.current_device() if device_index is None else device_index
    )
    splits = select_compressed_v_decode_cuda_splits(
        capability=_device_capability(selected_device),
        batch=batch,
        sequence_length=sequence_length,
    )
    cuda_workspace, token_major_output = workspace.views(
        batch=batch,
        query_heads=query_heads,
        splits=splits,
        value_dim=value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    return compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        scale=scale,
        splits=splits,
        workspace=cuda_workspace,
        output=None if feature_major_output is not None else token_major_output,
        feature_major_output=feature_major_output,
    )


def reference_compressed_v_decode_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materializing correctness reference that supports arbitrary float dtype."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("reference compressed-V decode Q/K/V must be rank-4")
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    if query_tokens != 1:
        raise ValueError("reference compressed-V decode accepts one query token")
    kv_heads = int(key.shape[1])
    if (
        int(key.shape[0]) != batch
        or int(value.shape[0]) != batch
        or int(value.shape[1]) != kv_heads
        or int(key.shape[2]) != int(value.shape[2])
        or int(key.shape[3]) != qk_dim
        or query_heads % kv_heads
    ):
        raise ValueError("reference compressed-V decode shapes are incompatible")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    work_dtype = (
        torch.float64
        if query.dtype == torch.float64
        or key.dtype == torch.float64
        or value.dtype == torch.float64
        else torch.float32
    )
    head_map = torch.arange(query_heads, device=query.device) // (
        query_heads // kv_heads
    )
    expanded_key = key.index_select(1, head_map)
    expanded_value = value.index_select(1, head_map)
    scores = (
        torch.matmul(
            query.to(work_dtype),
            expanded_key.to(work_dtype).transpose(-1, -2),
        )
        * selected_scale
    )
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, expanded_value.to(work_dtype)).to(query.dtype)


def reference_c1_paged_sparse_decode_attention(
    query: Tensor,
    packed_key_pages: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materializing reference for packed exact-K/resident C1-V pages."""

    if (
        query.ndim != 4
        or packed_key_pages.ndim != 5
        or value.ndim != 4
        or selected_page_ids.ndim != 3
    ):
        raise ValueError("paged sparse reference inputs have incompatible ranks")
    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    key_batch, kv_heads, page_slots, page_size, key_dim = map(
        int, packed_key_pages.shape
    )
    if (
        query_tokens != 1
        or key_batch != batch
        or key_dim != head_dim
        or tuple(value.shape[:2]) != (batch, kv_heads)
        or tuple(selected_page_ids.shape) != (batch, kv_heads, page_slots)
        or query_heads % kv_heads
    ):
        raise ValueError("paged sparse reference geometry is incompatible")
    selected_scale = head_dim**-0.5 if scale is None else float(scale)
    sequence_length = int(value.shape[2])
    value_dim = int(value.shape[3])
    heads_per_kv = query_heads // kv_heads
    work_dtype = (
        torch.float64
        if query.dtype == torch.float64
        or packed_key_pages.dtype == torch.float64
        or value.dtype == torch.float64
        else torch.float32
    )
    result = torch.empty(
        batch,
        query_heads,
        1,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    offsets = torch.arange(page_size, device=query.device)
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            page_ids = selected_page_ids[batch_index, kv_head]
            valid_slots = torch.nonzero(page_ids >= 0, as_tuple=False).flatten()
            chosen_pages = page_ids.index_select(0, valid_slots)
            token_ids = chosen_pages[:, None] * page_size + offsets[None, :]
            valid_tokens = token_ids < sequence_length
            flat_token_ids = token_ids[valid_tokens]
            keys = packed_key_pages[batch_index, kv_head].index_select(0, valid_slots)[
                valid_tokens
            ]
            values = value[batch_index, kv_head].index_select(0, flat_token_ids)
            head_start = kv_head * heads_per_kv
            head_stop = head_start + heads_per_kv
            scores = (
                torch.matmul(
                    query[batch_index, head_start:head_stop, 0].to(work_dtype),
                    keys.to(work_dtype).transpose(0, 1),
                )
                * selected_scale
            )
            probabilities = torch.softmax(scores, dim=-1)
            result[batch_index, head_start:head_stop, 0] = torch.matmul(
                probabilities,
                values.to(work_dtype),
            ).to(query.dtype)
    return result


def reference_compressed_v_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materializing causal prefill reference for correctness tests."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("reference compressed-V prefill Q/K/V must be rank-4")
    query_tokens = int(query.shape[2])
    sequence_length = int(key.shape[2])
    if query_tokens <= 1 or query_tokens != sequence_length:
        raise ValueError(
            "reference prefill requires equal Q/K lengths greater than one"
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )


__all__ = ['CompressedVDecodeWorkspace', 'c1_dense_gqa_v96_decode_attention_cuda', 'c1_pack_exact_key_pages_cuda', 'c1_paged_sparse_decode_attention_cuda', 'compressed_v_decode_attention_cuda', 'compressed_v_decode_attention_cuda_tuned', 'reference_c1_paged_sparse_decode_attention', 'reference_compressed_v_decode_attention', 'reference_compressed_v_prefill_attention', 'select_compressed_v_decode_cuda_splits']
