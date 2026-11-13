"""CPU numerics of the compact-Value vLLM adapters against the Hugging Face reference with the same factors.

Uniform banks and a per-layer (`core.allocate`-style) bank are served. Every tensor-parallel rank is a real adapter instance (BasisKVLlamaForCausalLM or
BasisKVQwen3ForCausalLM), built and weight-loaded by vLLM's own code. The ranks run as threads of one
process and exchange tensors through an in-process stand-in for the TP group, so the sharded
embeddings, MLPs, LM head, Q/K/V projections, norms, RoPE, Value folding, and decoders are the
production ones. Two CUDA pieces are replaced for this CPU test: the paged
attention kernels (here a plain softmax attention over a [K | compact V] cache) and the NCCL
AllGather arena (here a concatenation). The models run in FP32 so that the comparison is tight.

Run from the repository root, without a GPU:  python -m tests.test_compact_tp
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import types

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from core.factor_io import assemble, merge, write_layer
from core.model_layout import ValueGeometry
from inference import COMPACT_ARCHITECTURES, register
from inference.model import install_factors
from inference.tp import load_manifest, single_kv_head_layout

LAYERS, HEADS, KV_HEADS, HEAD_DIM, HIDDEN, VOCAB = 4, 32, 8, 128, 512, 128
PREFILL, DECODE = 32, 4
UNIFORM_RANKS = (32, 64, 96, 128)
ADAPTIVE_RANKS = (32, 96, 128, 64)  # one rank per layer: a small one, 96, the dense endpoint, 64


class EmulatedGroup:
    """In-process TP group: each thread is one rank, and collectives are barrier-synchronized exchanges."""

    device_group = None
    is_first_rank = is_last_rank = True

    def __init__(self, world_size):
        self.world_size = world_size
        self.local = threading.local()
        self.barrier = threading.Barrier(world_size)
        self.slots = [None] * world_size

    @property
    def rank_in_group(self):
        return self.local.rank

    def exchange(self, tensor):
        self.slots[self.rank_in_group] = tensor
        self.barrier.wait(timeout=120)
        values = list(self.slots)
        self.barrier.wait(timeout=120)
        return values

    def all_reduce(self, tensor):
        return torch.stack(self.exchange(tensor)).sum(0)

    def all_gather(self, tensor, dim=-1):
        return torch.cat(self.exchange(tensor), dim)

    def gather(self, tensor, dst=0, dim=-1):
        values = self.exchange(tensor)
        return torch.cat(values, dim) if self.rank_in_group == dst else None


class ReferenceAttention(nn.Module):
    """Causal softmax attention of new tokens over everything cached so far, for one sequence."""

    def __init__(self, real):
        super().__init__()
        self.num_heads, self.num_kv_heads = real.num_heads, real.num_kv_heads
        self.head_size, self.head_size_v, self.scale = real.head_size, real.head_size_v, real.impl.scale
        self.keys = self.values = None

    def attend(self, query, key, value):
        self.keys = key if self.keys is None else torch.cat((self.keys, key))
        self.values = value if self.values is None else torch.cat((self.values, value))
        total, new, group = self.keys.shape[0], query.shape[0], self.num_heads // self.num_kv_heads
        keys = self.keys.transpose(0, 1).repeat_interleave(group, 0)
        values = self.values.transpose(0, 1).repeat_interleave(group, 0)
        scores = query.transpose(0, 1) @ keys.mT * self.scale
        future = torch.arange(total)[None, :] > (total - new + torch.arange(new))[:, None]
        return (scores.masked_fill(future, float('-inf')).softmax(-1) @ values).transpose(0, 1)

    def forward(self, query, key, value):
        return self.attend(query.view(-1, self.num_heads, self.head_size), key.view(-1, self.num_kv_heads, self.head_size),
                           value.view(-1, self.num_kv_heads, self.head_size_v)).flatten(1)


class ConcatenatingBoundary:
    """Emulate PreparedJointValueOutput with in-process concatenation and the same call contract."""

    def __init__(self, group, device, max_local_width, max_tokens):
        self.max_local_width = max_local_width
        self.gathered_widths = []

    def attention(self, attn_layer, query, key, value, kv_cache, attn_metadata, decoder):
        from vllm.distributed import get_tp_group
        local = attn_layer.attend(query, key, value).flatten(1)
        assert local.shape[1] <= self.max_local_width
        gathered = get_tp_group().all_gather(local, -1)
        self.gathered_widths.append(gathered.shape[1])
        return gathered @ decoder


def write_model(directory, kind):
    common = dict(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=256, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                  num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, max_position_embeddings=256)
    model = LlamaForCausalLM(LlamaConfig(**common)) if kind == 'llama' else Qwen3ForCausalLM(Qwen3Config(**common))
    with torch.no_grad():
        for parameter in model.parameters():  # norm weights away from one, so that a skipped or misplaced norm shows
            parameter.normal_(0, 0.05) if parameter.ndim > 1 else parameter.uniform_(0.5, 1.5)
    model.save_pretrained(directory)
    return model.config


def write_uniform(directory, config, rank):
    geometry = ValueGeometry.from_config(config, range(LAYERS))
    fit = dict(cache_rank_per_head=rank, geometry=geometry.to_dict(), fit_windows=0, heldout_windows=0, encoder_sweeps=0)
    directory.mkdir()
    for layer in range(LAYERS):
        encoder = (torch.randn(KV_HEADS, HEAD_DIM, rank) / HEAD_DIM ** 0.5).bfloat16()
        decoder = (torch.randn(HEADS, rank, HIDDEN) * 0.05).bfloat16()
        write_layer(directory, layer, encoder, decoder, dict(fit_config=fit, solver=dict(method='random'),
            fit=dict(factor_dtype_relative_mse=0.0), heldout=dict(factor_dtype_relative_mse=0.0)))
    merge(directory, range(LAYERS), geometry)


def write_adaptive(directory, root, kind, ranks):
    """The bank `core.allocate` writes: layer l copied from the uniform bank of rank ranks[l]."""
    assemble(directory, {rank: root / f'{kind}_v{rank}' for rank in set(ranks)}, list(ranks), dict(method='test'))


def planned_cache_bytes(vllm_config, model):
    """Per-layer page bytes and pool bytes per block from vLLM's own cache planner, for the specs the adapter's layers report."""
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
    attentions = [layer.self_attn.attn for layer in model.model.layers]
    specs = {attention.layer_name: attention.get_kv_cache_spec(vllm_config) for attention in attentions}
    layouts = get_supported_kv_cache_layouts([attention.attn_backend for attention in attentions])
    resolve_kv_cache_layout(vllm_config, [[layout.name for layout in layouts]], specs.values())  # as the engine core does
    groups = get_kv_cache_groups(vllm_config, specs)
    config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory=1 << 30)
    pool = {tensor.size // config.num_blocks for tensor in config.kv_cache_tensors}
    return [spec.page_size_bytes for spec in specs.values()], pool, next(iter(specs.values())).block_size, len(groups)


def build_rank(model_dir, factors, architecture, group, references, chunked_prefill):
    from vllm.config import CompilationConfig, DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.model_loader.utils import get_model_architecture

    overrides = dict(architectures=[architecture], basiskv_joint_value_factor_dir=str(factors), basiskv_joint_value_validation='sha256')
    model_config = ModelConfig(model=str(model_dir), dtype='bfloat16', seed=0, hf_overrides=overrides)
    # As in benchmarks.tp, the model runs without torch.compile. vLLM's compiled CUDA operators (RoPE, RMSNorm,
    # SiLU) do not exist on CPU, so its own PyTorch-native implementations of them are selected.
    vllm_config = VllmConfig(model_config=model_config, device_config=DeviceConfig(device='cpu'),
                             compilation_config=CompilationConfig(mode='NONE', custom_ops=['none']))
    vllm_config.scheduler_config.enable_chunked_prefill = chunked_prefill
    model_class, _ = get_model_architecture(model_config)
    with set_current_vllm_config(vllm_config):
        model = model_class(vllm_config=vllm_config).float()
    model.logits_processor.head_dtype = torch.float32  # vLLM's LM head otherwise casts to the configured BF16
    loaded = model.load_weights(load_file(str(model_dir / 'model.safetensors')).items())
    assert {name for name, _ in model.named_parameters()} <= loaded, 'a parameter was left uninitialized'
    ranks = load_manifest(factors)['layer_ranks']
    local_kv = model.model.layers[0].self_attn.num_kv_heads
    # vLLM's cache planner keeps the layers in one group and sizes each layer's tensor by its own rank: no padding to the widest.
    pages, pool, block_size, groups = planned_cache_bytes(vllm_config, model)
    assert pages == [2 * block_size * local_kv * (HEAD_DIM + rank) for rank in ranks], 'a layer cache page is not sized by its own rank'
    assert groups == 1 and pool == {sum(pages)}, 'the cache pool does not hold exactly the sum of the layer pages'
    for layer, rank in zip(model.model.layers, ranks, strict=True):
        attention = layer.self_attn
        real = attention.attn
        assert attention.qkv_proj.value_loaded and attention.value_rank == real.head_size_v == rank
        # Check the compact Value width in the backend cache layout.
        assert real.attn_backend.get_kv_cache_shape(4, 16, local_kv, HEAD_DIM) == (4, 16, local_kv, HEAD_DIM + rank)
        assert attention.qkv_proj.weight.shape[0] == (attention.num_heads + local_kv) * HEAD_DIM + local_kv * rank
        reference = ReferenceAttention(real)
        if attention.boundary is None:
            attention.attn = reference
        else:
            object.__setattr__(reference, '_basiskv_joint_value_output', attention.boundary)
            references[group.rank_in_group, real.layer_name] = reference
    return model


def serve(kind, bank, world_size, root):
    """Teacher-forced logits of the emulated TP deployment and of the HF reference with the same factors."""
    from vllm.distributed import parallel_state
    import vllm.model_executor.layers.attention.attention as attention_layer
    import inference.vllm as adapter
    from inference.backends import paged_backend

    model_dir, factors = root / kind, root / f'{kind}_{bank}'
    ranks = load_manifest(factors)['layer_ranks']
    group = EmulatedGroup(world_size)
    parallel_state._TP = group
    parallel_state._PP = types.SimpleNamespace(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)
    # The native attention layers are replaced by the adapter; they only need some backend class to construct.
    attention_layer.get_attn_backend = lambda head_size, *args, **kwargs: paged_backend(head_size)
    references = {}
    adapter.PreparedJointValueOutput = ConcatenatingBoundary
    adapter.get_attention_context = lambda name: (None, references[group.rank_in_group, name], None, None)
    prepared = single_kv_head_layout(ValueGeometry.from_factors(factors).to_dict(), world_size)
    models = []
    for index in range(world_size):
        group.local.rank = index
        models.append(build_rank(model_dir, factors, COMPACT_ARCHITECTURES[kind], group, references, chunked_prefill=prepared))
    assert all((model.joint_value_boundary is not None) == prepared for model in models)

    tokens = torch.randint(0, VOCAB, (PREFILL + DECODE,), generator=torch.Generator().manual_seed(sum(ranks) + world_size))
    # The DiffKV layout takes the prompt in two chunks, the second attending to the cached first; the paged
    # layout admits it whole. Decode then feeds one token per step.
    half = PREFILL // 2
    steps = ([(0, half), (half, PREFILL)] if prepared else [(0, PREFILL)]) + [(PREFILL + i, PREFILL + i + 1) for i in range(DECODE)]

    def run(index):
        finished = False
        try:
            group.local.rank = index
            hidden_states, logits = [], []
            for start, stop in steps:
                hidden = models[index](tokens[start:stop], torch.arange(start, stop))
                hidden_states.append(hidden)
                logits.append(models[index].compute_logits(hidden))
            finished = True
            return torch.cat(hidden_states), None if logits[0] is None else torch.cat(logits)
        finally:
            if not finished:
                group.barrier.abort()

    with torch.inference_mode(), ThreadPoolExecutor(world_size) as pool:
        outputs = list(pool.map(run, range(world_size)))
    hidden, logits = outputs[0]
    assert all(torch.equal(hidden, other) for other, _ in outputs[1:]), 'ranks disagree on the residual stream'
    for model in models:
        attentions = [layer.self_attn for layer in model.model.layers]
        assert all(attention.calls == len(steps) for attention in attentions)
    local_heads = HEADS // world_size
    for index, model in enumerate(models):
        for layer, rank in zip(model.model.layers, ranks, strict=True):
            cache = references[index, layer.self_attn.attn.layer_name] if prepared else layer.self_attn.attn
            assert cache.values.shape == (PREFILL + DECODE, KV_HEADS // world_size, rank), 'a cached Value does not have its layer rank'
            assert layer.self_attn.decoder.shape == (HEADS * rank, HIDDEN)
        if prepared:  # each layer gathered its own width, in layer order, on every step
            assert model.joint_value_boundary.gathered_widths == [HEADS * rank for rank in ranks] * len(steps), 'a collective did not carry its layer width'
            assert model.joint_value_boundary.max_local_width == local_heads * max(ranks)

    reference = install_factors(AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True).eval(),
                                value_factors=factors)
    with torch.inference_mode():
        expected = reference(input_ids=tokens[None]).logits[0]
    return logits, expected


def refused(build):
    """True when construction stops on one of the adapter's assertions."""
    try:
        build()
    except AssertionError:
        return True
    return False


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    register()
    with tempfile.TemporaryDirectory(prefix='basiskv-tp-') as temporary:
        root = Path(temporary)
        for kind in COMPACT_ARCHITECTURES:
            config = write_model(root / kind, kind)
            for rank in UNIFORM_RANKS:
                write_uniform(root / f'{kind}_v{rank}', config, rank)
            write_adaptive(root / f'{kind}_adaptive', root, kind, ADAPTIVE_RANKS)
        worst = 0.0
        for kind in COMPACT_ARCHITECTURES:
            for bank, world_size in (('v64', 8), ('v96', 8), ('adaptive', 8), ('adaptive', 4), ('adaptive', 2), ('v96', 1), ('adaptive', 1)):
                logits, expected = serve(kind, bank, world_size, root)
                error = float((logits - expected).abs().max() / expected.abs().max())
                worst = max(worst, error)
                print(f'{kind:5s} {bank:8s} TP{world_size}: relative max logit error {error:.2e}, top-1 agreement '
                      f'{int((logits.argmax(-1) == expected.argmax(-1)).sum())}/{len(expected)}', flush=True)
                assert torch.isfinite(logits).all() and error < 1e-4, 'the compact path differs from the reference with the same factors'

        from vllm.distributed import parallel_state
        group = EmulatedGroup(8)
        parallel_state._TP = group
        group.local.rank = 0
        build = lambda model, factors, architecture, chunked=True: build_rank(root / model, root / factors, architecture, group, {}, chunked)
        assert refused(lambda: build('llama', 'llama_v96', COMPACT_ARCHITECTURES['qwen3'])), 'a Llama checkpoint was accepted by the Qwen3 adapter'
        assert refused(lambda: build('qwen3', 'llama_v96', COMPACT_ARCHITECTURES['qwen3'])), 'factors of another model were accepted'
        parallel_state._TP = group = EmulatedGroup(4)
        group.local.rank = 0
        assert refused(lambda: build('llama', 'llama_adaptive', COMPACT_ARCHITECTURES['llama'])), 'chunked prefill was accepted on the paged layout'
    print(f'PASS: worst relative max logit error {worst:.2e}')


if __name__ == '__main__':
    main()
