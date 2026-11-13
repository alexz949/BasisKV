"""Paged DiffKV kernels and the single-KV-head prepared-output specialization."""

import torch
from basiskv_kernels.compressed_v_prefill import compressed_v_prefill_attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend, FlashAttentionDiffKVImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash_diffkv
from functools import lru_cache
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend, TritonAttentionDiffKVImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from basiskv_kernels.diffkv_decode import diffkv_decode
from basiskv_kernels.diffkv_prefill import diffkv_prefill

NUM_KV_SPLITS = 8

def _kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, value_head_size):
    assert block_size % 16 == 0, 'the compact-V kernels tile the paged cache in blocks of 16'
    return (num_blocks, block_size, num_kv_heads, head_size + value_head_size)

@register_backend(AttentionBackendEnum.CUSTOM)
class PagedBackend(FlashAttentionDiffKVBackend):
    """Independent key/value widths; cache insertion stays inside forward."""
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        return 'CUSTOM'

    @staticmethod
    def get_impl_cls():
        return PagedImpl

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)

class PagedImpl(FlashAttentionDiffKVImpl):
    """Use the same dynamic-Value Triton kernels around the paged cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (self.attn_type == AttentionType.DECODER and self.kv_cache_dtype in ('auto', 'bfloat16') and self.alibi_slopes is None
                and self.sinks is None and self.sliding_window == (-1, -1) and self.dcp_world_size == 1), (
            'BasisKV compact-V attention supports BF16 decoder attention without ALiBi, sinks, sliding window, or DCP')

    def _run_decode(self, *, layer, query, kv_cache, block_table, seq_lens, output):
        num_sequences = int(query.shape[0])
        key_cache = kv_cache[..., :self.head_size]
        value_cache = kv_cache[..., self.head_size:]
        value_head_size = int(value_cache.shape[-1])
        intermediate = torch.empty((num_sequences, self.num_heads, NUM_KV_SPLITS, value_head_size + 1), dtype=torch.float32, device=query.device)
        lse = torch.empty((num_sequences, self.num_heads), dtype=torch.float32, device=query.device)
        decode_attention_fwd(query, key_cache, value_cache, output, lse, block_table, seq_lens, intermediate, NUM_KV_SPLITS, self.scale, page_size=int(kv_cache.shape[1]), logit_cap=self.logits_soft_cap, k_scale=layer._k_scale, v_scale=layer._v_scale)

    @staticmethod
    def _host_sequence_layout(attn_metadata):
        """Copy dynamic varlen metadata to CPU once for all layers in this step."""
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        num_sequences = int(query_start_loc.shape[0]) - 1
        cache_key = (query_start_loc.data_ptr(), query_start_loc._version, seq_lens.data_ptr(), seq_lens._version, int(attn_metadata.num_actual_tokens), num_sequences)
        cached = getattr(attn_metadata, '_basiskv_host_sequence_layout', None)
        if cached is None or cached[0] != cache_key:
            starts = tuple(map(int, query_start_loc[:num_sequences + 1].cpu().tolist()))
            lengths = tuple(map(int, seq_lens[:num_sequences].cpu().tolist()))
            cached = (cache_key, starts, lengths)
            setattr(attn_metadata, '_basiskv_host_sequence_layout', cached)
        return (cached[1], cached[2])

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None, 'compact-V output quantization is not supported'
        if attn_metadata is None:
            return output.fill_(0)
        triton_reshape_and_cache_flash_diffkv(key, value, kv_cache, attn_metadata.slot_mapping, self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        if attn_metadata.max_query_len == 1 and (not attn_metadata.use_cascade):
            num_tokens = int(attn_metadata.num_actual_tokens)
            query_view = query[:num_tokens]
            output_view = output[:num_tokens]
            self._run_decode(layer=layer, query=query_view, kv_cache=kv_cache, block_table=attn_metadata.block_table[:num_tokens], seq_lens=attn_metadata.seq_lens[:num_tokens], output=output_view)
            return output
        num_tokens = int(attn_metadata.num_actual_tokens)
        assert attn_metadata.causal and not attn_metadata.use_cascade, 'compact-V prefill requires causal, non-cascade attention'
        (starts, sequence_lengths) = self._host_sequence_layout(attn_metadata)
        for (sequence_index, (query_start, query_stop, sequence_length)) in enumerate(zip(starts[:-1], starts[1:], sequence_lengths, strict=True)):
            query_length = query_stop - query_start
            query_slice = query[query_start:query_stop]
            output_slice = output[query_start:query_stop]
            if query_length == 1:
                self._run_decode(layer=layer, query=query_slice, kv_cache=kv_cache, block_table=attn_metadata.block_table[sequence_index:sequence_index + 1], seq_lens=attn_metadata.seq_lens[sequence_index:sequence_index + 1], output=output_slice)
                continue
            assert query_length == sequence_length, 'BasisKV compact-V does not yet support chunked prefill'
            compact_output = compressed_v_prefill_attention(query_slice.transpose(0, 1).unsqueeze(0), key[query_start:query_stop].transpose(0, 1).unsqueeze(0), value[query_start:query_stop].transpose(0, 1).unsqueeze(0), scale=self.scale)
            output_slice.copy_(compact_output.squeeze(0).transpose(0, 1))
        return output

class JointValueDiffKVBackend(TritonAttentionDiffKVBackend):

    @staticmethod
    def get_impl_cls():
        return JointValueDiffKVImpl

class JointValueDiffKVImpl(TritonAttentionDiffKVImpl):

    def __init__(self, *args, **kwargs):
        self.value_rank = int(kwargs.pop('value_rank', 64))
        super().__init__(*args, **kwargs)
        assert (self.head_size == 128 and self.num_heads in (4, 8) and self.num_kv_heads == 1 and 0 < self.value_rank <= 128
                and self.alibi_slopes is None and self.sinks is None and self.sliding_window == (-1, -1)
                and self.logits_soft_cap == 0 and not self.use_alibi_sqrt), (
            'the DiffKV kernels are compiled for one local KV head with 4 or 8 query heads, head_dim 128, a Value rank up to 128, '
            'and plain softmax attention')

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None, 'compact-V output quantization is not supported'
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.max_query_len <= 1:
            n = attn_metadata.num_actual_tokens
            cache = kv_cache.transpose(1, 2)
            diffkv_decode(q=query[:n], k=cache[..., :128], v=cache[..., 128:128 + self.value_rank], out=output[:n], cu_seqlens_q=attn_metadata.query_start_loc, seqused_k=attn_metadata.seq_lens, block_table=attn_metadata.block_table, softmax_scale=self.scale, softmax_segm_output=attn_metadata.softmax_segm_output, softmax_segm_max=attn_metadata.softmax_segm_max, softmax_segm_expsum=attn_metadata.softmax_segm_expsum, split_threshold=attn_metadata.seq_threshold_3D, max_sequence_length=attn_metadata.max_seq_len)
            return output
        assert not attn_metadata.use_cascade, 'compact-V prefill excludes cascade attention'
        n = attn_metadata.num_actual_tokens
        cache = kv_cache.transpose(1, 2)
        diffkv_prefill(q=query[:n], k=cache[..., :128], v=cache[..., 128:128 + self.value_rank], out=output[:n], cu_seqlens_q=attn_metadata.query_start_loc, seqused_k=attn_metadata.seq_lens, block_table=attn_metadata.block_table, softmax_scale=self.scale, max_query_len=attn_metadata.max_query_len, max_sequence_length=attn_metadata.max_seq_len)
        return output

@lru_cache(maxsize=None)
def joint_value_diffkv_backend(value_rank):
    """Return a cached DiffKV backend with V-sized split-KV scratch."""
    selected_rank = int(value_rank)

    class RankedMetadataBuilder(TritonAttentionMetadataBuilder):

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.softmax_segm_output = torch.empty((self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments, next_power_of_2(selected_rank)), dtype=torch.float32, device=device)

    class RankedImpl(JointValueDiffKVImpl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, value_rank=selected_rank, **kwargs)

    class RankedBackend(JointValueDiffKVBackend):
        head_size_v = selected_rank

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
            return _kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, selected_rank)

        @staticmethod
        def get_impl_cls():
            return RankedImpl

        @staticmethod
        def get_builder_cls():
            return RankedMetadataBuilder
    RankedMetadataBuilder.__name__ = f'JointValueDiffKVMetadataBuilderV{selected_rank}'
    RankedMetadataBuilder.__qualname__ = RankedMetadataBuilder.__name__
    RankedBackend.__name__ = f'JointValueDiffKVBackendV{selected_rank}'
    RankedBackend.__qualname__ = RankedBackend.__name__
    return RankedBackend


@lru_cache(maxsize=None)
def paged_backend(value_rank):
    """Cache width comes from the factors, independently of Q/K width."""
    class RankedPagedBackend(PagedBackend):
        head_size_v = value_rank
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
            return _kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, value_rank)
    RankedPagedBackend.__name__ = f'BasisKVPagedV{value_rank}'
    return RankedPagedBackend
