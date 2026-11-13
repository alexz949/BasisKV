"""Config-driven Llama and Qwen3 compact-Value TP adapters for vLLM."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from vllm.distributed import get_pp_group, get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.utils.torch_utils import direct_register_custom_op

from inference.backends import joint_value_diffkv_backend, paged_backend
from core.model_layout import ValueGeometry
from inference.tp import fold_value_weight, head_slices, layer_ranks, load_layer, load_manifest, single_kv_head_layout
from basiskv_kernels.allgather import AllGather

def check_supported(vllm_config):
    """One envelope check for every adapter: BF16 text decoding, no quantization, LoRA, speculation, or PP/DCP/DP."""
    parallel = vllm_config.parallel_config
    assert (get_pp_group().world_size == 1 and parallel.decode_context_parallel_size == 1 and parallel.data_parallel_size == 1
            and vllm_config.quant_config is None and vllm_config.lora_config is None and vllm_config.speculative_config is None
            and vllm_config.model_config.dtype == torch.bfloat16 and vllm_config.cache_config.cache_dtype in ('auto', 'bfloat16')), (
        'BasisKV vLLM adapters support BF16 text decoding without quantization, LoRA, speculative decoding, '
        'pipeline, context, or data parallelism')

def factor_configuration(vllm_config):
    config = vllm_config.model_config.hf_config
    group = get_tp_group()
    check_supported(vllm_config)
    root = Path(config.basiskv_joint_value_factor_dir).expanduser().resolve()
    validation = getattr(config, 'basiskv_joint_value_validation', 'sha256')
    manifest = load_manifest(root)
    geometry = manifest['fit_config']['geometry']
    assert ValueGeometry.from_config(config, range(config.num_hidden_layers)).to_dict() == geometry, (
        f'the Value factors in {root} were fitted for another model geometry')
    head_slices(geometry, group.rank_in_group, group.world_size)
    return root, manifest, validation


class JointValueQKVParallelLinear(QKVParallelLinear):
    def __init__(self, encoder, prefix, hidden_size, num_query_heads, num_kv_heads):
        _, dim, rank = encoder.shape
        super().__init__(hidden_size, dim, num_query_heads, num_kv_heads,
                         bias=False, prefix=prefix, v_head_size=rank)
        self.value_rank = rank
        self.register_buffer('encoder', encoder, persistent=False)
        self.value_loaded = False

    def _load_value(self, param, loaded_weight):
        rows = self.num_kv_heads * self.head_size
        local = loaded_weight.narrow(0, self.tp_rank * rows, rows).to(self.encoder.device)
        compact = fold_value_weight(local, self.encoder)
        offset = (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(0, offset, self.num_kv_heads * self.value_rank).copy_(compact)
        self.value_loaded = True

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ('q', 'k', 'v')
        if loaded_shard_id == 'v':
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ('q', 'k', 'v')
        if loaded_shard_id == 'v':
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)


class JointValueAttention(nn.Module):
    """Compact-Value attention of one TP rank, built from the native Llama or Qwen3 attention.

    Both models keep their own rotary module. Qwen3 normalizes Q and K per head before RoPE; Llama
    has no such norms. The Value encoder is folded into the V projection and the fitted decoder
    replaces the dense output projection, so neither projection may carry a bias.
    """

    def __init__(self, native, encoder, decoder, boundary, config):
        super().__init__()
        assert native.qkv_proj.bias is None and native.o_proj.bias is None, 'the compact adapter folds bias-free V and output projections'
        self.num_heads, self.num_kv_heads = native.num_heads, native.num_kv_heads
        self.head_dim, self.value_rank = encoder.shape[-2:]
        self.boundary = boundary
        name = native.attn.layer_name
        config.compilation_config.static_forward_context.pop(name)
        hf = config.model_config.hf_config
        self.qkv_proj = JointValueQKVParallelLinear(encoder, name[:-5] + '.qkv_proj',
                                          hf.hidden_size, hf.num_attention_heads, hf.num_key_value_heads)
        self.q_norm, self.k_norm = (native.q_norm, native.k_norm) if hf.model_type == 'qwen3' else (nn.Identity(), nn.Identity())
        self.rotary_emb = native.rotary_emb
        self.calls = 0  # Python-level executions: every eager step and each CUDA Graph capture
        backend = joint_value_diffkv_backend(self.value_rank) if boundary is not None else paged_backend(self.value_rank)
        self.attn = Attention(self.num_heads, self.head_dim, native.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name, attn_backend=backend, head_size_v=self.value_rank)
        self.register_buffer('decoder', decoder.flatten(0, 1), persistent=False)
        if boundary is not None:
            object.__setattr__(self.attn, '_basiskv_joint_value_output', boundary)

    def forward(self, positions, hidden_states):
        self.calls += 1
        qkv, _ = self.qkv_proj(hidden_states)
        qsize, ksize = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = qkv.split((qsize, ksize, self.num_kv_heads * self.value_rank), dim=-1)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, qsize)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(-1, ksize)
        q, k = self.rotary_emb(positions, q, k)
        if self.boundary is not None:
            return torch.ops.vllm.basiskv_joint_value_attention_output(q, k, v, self.decoder, self.attn.layer_name)
        # This backend inserts K/V inside forward; the specialized op updates them externally.
        local = self.attn(q, k, v).reshape(-1, self.num_heads * self.value_rank)
        return tensor_model_parallel_all_gather(local, dim=-1) @ self.decoder


class CompactValueModel:
    """Compact-Value construction shared by the Llama and Qwen3 adapters.

    The native vLLM class, next in the MRO, builds the embeddings, norms, MLPs, and LM head and loads
    their weights. This mixin replaces every attention layer with JointValueAttention and keeps the dense
    output projections out of the checkpoint stream; the fitted decoders stand in for them.
    """

    model_type = None

    def __init__(self, *, vllm_config, prefix=''):
        config = vllm_config.model_config.hf_config
        assert config.model_type == self.model_type, f'{type(self).__name__} serves {self.model_type} checkpoints, not {config.model_type}'
        root, manifest, validation = factor_configuration(vllm_config)
        group = get_tp_group()
        geometry = manifest['fit_config']['geometry']
        ranks = layer_ranks(manifest)
        assert set(ranks) == set(range(config.num_hidden_layers)) and all(0 < rank <= geometry['head_dim'] for rank in ranks.values()), (
            f'{root} does not give every layer a Value rank between 1 and {geometry["head_dim"]}')
        prepared = single_kv_head_layout(geometry, group.world_size)
        assert prepared or not vllm_config.scheduler_config.enable_chunked_prefill, (
            'chunked prefill needs the DiffKV kernels, which serve one KV head and 4 or 8 query heads per rank; '
            'disable chunked prefill for this tensor-parallel size')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        device = next(self.parameters()).device
        # One communicator serves every layer; its workspace holds the widest layer and each layer gathers its own width.
        self.joint_value_boundary = PreparedJointValueOutput(group.device_group, device, geometry['num_heads'] // group.world_size * max(ranks.values()),
            vllm_config.scheduler_config.max_num_batched_tokens) if prepared else None
        for index, layer in enumerate(self.model.layers):
            encoder, decoder = load_layer(root, manifest, index, group.rank_in_group, tp_size=group.world_size,
                device=device, dtype=torch.bfloat16, validation=validation)
            layer.self_attn = JointValueAttention(layer.self_attn, encoder, decoder, self.joint_value_boundary, vllm_config)

    def load_weights(self, weights):
        dense_outputs = tuple(f'model.layers.{layer}.self_attn.o_proj.' for layer in range(self.config.num_hidden_layers))
        return super().load_weights((name, tensor) for name, tensor in weights if not name.startswith(dense_outputs))

    def joint_value_statistics(self):
        """Per-layer evidence of the compact path on this rank, read from the installed modules and vLLM's cache tensors."""
        attentions = [layer.self_attn for layer in self.model.layers]
        stats = self.joint_value_boundary.statistics() if self.joint_value_boundary is not None else dict(backend='vllm_allgather', decoder_gemms=1)
        caches = [layer.attn.kv_cache for layer in attentions]
        return stats | dict(model_type=self.model_type, loaded_value_layers=sum(layer.qkv_proj.value_loaded for layer in attentions),
            compact_attention_calls=sum(layer.calls for layer in attentions),
            layer_value_ranks=[layer.value_rank for layer in attentions],
            # Widths and sizes of the cache tensors vLLM allocated, not of the configuration that requested them.
            cache_value_widths=[int(cache.shape[-1]) - layer.head_dim for cache, layer in zip(caches, attentions)],
            kv_cache_shapes=[list(cache.shape) for cache in caches],
            kv_cache_bytes=sum(cache.numel() * cache.element_size() for cache in caches),
            collective_local_widths=[layer.num_heads * layer.value_rank for layer in attentions],
            attention_backends=sorted({layer.attn.attn_backend.get_name() for layer in attentions}),
            attention_impls=sorted({type(layer.attn.impl).__name__ for layer in attentions}),
            decode_output_layout='feature_major_allgather_slot' if self.joint_value_boundary is not None else 'token_major',
            key_head_size=attentions[0].head_dim, local_query_heads=attentions[0].num_heads, local_kv_heads=attentions[0].num_kv_heads)


class BasisKVLlamaForCausalLM(CompactValueModel, LlamaForCausalLM):
    model_type = 'llama'


class BasisKVQwen3ForCausalLM(CompactValueModel, Qwen3ForCausalLM):
    model_type = 'qwen3'


class PreparedJointValueOutput:
    """One communicator shared by sequential attention layers.

    Plans are warmed on the actual capture stream before capture begins.
    The custom operator returns a separate GEMM output tensor, preserving
    its non-aliasing output contract for compilation.
    """

    def __init__(self, group, device, max_local_width: int, max_tokens: int):
        self.communicator = AllGather.from_distributed(group, device=device)
        # The workspace is sized for the widest layer; every plan gathers exactly its layer's width.
        self.communicator.configure_workspace(tokens=max_tokens, max_total_width=max_local_width * self.communicator.world_size, dtype=torch.bfloat16)
        self.prepared = {}
        self.capture_calls = 0
        self.eager_calls = 0
        self.direct_decode_calls = 0

    def _prepared(self, local_width: int, tokens: int, dtype: torch.dtype, device: torch.device):
        stream = torch.cuda.current_stream(device)
        key = (local_width, tokens, stream.cuda_stream)
        capturing = torch.cuda.is_current_stream_capturing()
        prepared = self.prepared.get(key)
        if prepared is None:
            assert not capturing, 'Warm this width and token count on the capture stream first'
            prepared = self.communicator.prepare(local_width, tokens=tokens, dtype=dtype)
            self.prepared[key] = prepared
        if capturing:
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        return prepared

    @staticmethod
    def _gather_decode(prepared, decoder: Tensor) -> Tensor:
        arena = prepared.gather()
        return torch.mm(arena.T, decoder)

    def __call__(self, local: Tensor, decoder: Tensor) -> Tensor:
        prepared = self._prepared(int(local.shape[1]), int(local.shape[0]), local.dtype, local.device)
        prepared.local.copy_(local.T)
        return self._gather_decode(prepared, decoder)

    def attention(self, attn_layer, query: Tensor, key: Tensor, value: Tensor, kv_cache: Tensor, attn_metadata, decoder: Tensor) -> Tensor:
        tokens = int(query.shape[0])
        if attn_metadata is not None and attn_metadata.max_query_len <= 1:
            prepared = self._prepared(attn_layer.impl.num_heads * attn_layer.head_size_v, tokens, query.dtype, query.device)
            local = prepared.local
            output = local.T.view(tokens, attn_layer.impl.num_heads, attn_layer.head_size_v)
            attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output)
            self.direct_decode_calls += 1
            return self._gather_decode(prepared, decoder)
        output = torch.empty((tokens, attn_layer.impl.num_heads, attn_layer.head_size_v), dtype=query.dtype, device=query.device)
        attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output)
        return self(output.flatten(1), decoder)

    def statistics(self):
        return dict(backend='uniform_nccl', decoder_gemms=1, capture_calls=self.capture_calls, eager_calls=self.eager_calls, direct_decode_calls=self.direct_decode_calls, prepared_shapes=sorted({(int(key[0]), int(key[1])) for key in self.prepared}))

def joint_value_attention_output(query: Tensor, key: Tensor, value: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    (attn_metadata, attn_layer, kv_cache, slot_mapping) = get_attention_context(layer_name)
    query = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    key = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)
    if slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(attn_layer, key, value, kv_cache, slot_mapping)
    boundary = attn_layer._basiskv_joint_value_output
    return boundary.attention(attn_layer, query, key, value, kv_cache, attn_metadata, decoder)

def joint_value_attention_output_fake(query: Tensor, key: Tensor, value: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    return query.new_empty((query.shape[0], decoder.shape[1]))
direct_register_custom_op(op_name='basiskv_joint_value_attention_output', op_func=joint_value_attention_output, fake_impl=joint_value_attention_output_fake)



class BasisKVWorkerExtension:
    """Expose small model statistics through vLLM's string RPC interface."""

    def basiskv_cuda_statistics(self) -> dict[str, Any]:
        """Return device-local CUDA statistics for any vLLM model."""
        return {'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()), 'cuda_device': torch.cuda.get_device_name()}

    def basiskv_joint_value_statistics(self) -> dict[str, Any]:
        return self.basiskv_cuda_statistics() | self.get_model().joint_value_statistics()

    def basiskv_profile_start(self) -> None:
        """Profile rank zero only; benchmark timing runs never enable this."""
        if torch.distributed.get_rank() == 0:
            torch.cuda.synchronize()
            self._basiskv_profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            self._basiskv_profiler.start()

    def basiskv_profile_stop(self, output_prefix: str) -> None:
        if torch.distributed.get_rank() == 0:
            import csv
            import gzip
            from pathlib import Path
            import shutil
            import tempfile
            torch.cuda.synchronize()
            profiler = self._basiskv_profiler
            profiler.stop()
            prefix = Path(output_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            kernels = {}
            for event in profiler.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    row = kernels.setdefault(event.name, [0, 0.0])
                    row[0] += 1
                    row[1] += event.device_time_total
            with prefix.with_suffix('.kernels.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['kernel', 'count', 'total_us', 'mean_us'])
                for (name, (count, total)) in sorted(kernels.items(), key=lambda item: -item[1][1]):
                    writer.writerow([name, count, total, total / count])
            with tempfile.TemporaryDirectory(prefix='scripts-profile-') as temporary:
                trace = Path(temporary) / 'trace.json'
                profiler.export_chrome_trace(str(trace))
                with trace.open('rb') as source, gzip.open(prefix.with_suffix('.trace.json.gz'), 'wb') as target:
                    shutil.copyfileobj(source, target)
            del self._basiskv_profiler
