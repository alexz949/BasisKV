"""Shared factor loading, TP head ownership, and projection folding."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from core.factor_io import DECODER_KEY, ENCODER_KEY, FORMAT, read_layer
from core.router_io import read_json


def load_manifest(root):
    root = Path(root)
    manifest = read_json(root / 'results.json')
    assert manifest['format'] == FORMAT and manifest['status'] == 'complete', f'{root} is not a complete Value-factor bank'
    assert 'cache_rank_per_head' in manifest['fit_config'], 'TP inference uses a uniform-rank bank, not an allocated one'
    return manifest


def head_slices(geometry, tp_rank, tp_size):
    heads, groups = geometry['num_heads'], geometry['num_kv_heads']
    assert heads % tp_size == groups % tp_size == 0, f'{heads} query heads and {groups} KV heads do not shard over TP {tp_size}'
    return (slice(tp_rank * groups // tp_size, (tp_rank + 1) * groups // tp_size),
            slice(tp_rank * heads // tp_size, (tp_rank + 1) * heads // tp_size))


def single_kv_head_layout(geometry, tp_size):
    """Each rank holds one KV head and 4 or 8 query heads: the layout the DiffKV kernels and the prepared AllGather serve."""
    return geometry['num_kv_heads'] == tp_size and geometry['num_heads'] // tp_size in (4, 8)


def load_layer(root, manifest, layer, tp_rank, *, tp_size, device, dtype, validation='sha256'):
    root = Path(root)
    fit = manifest['fit_config']
    geometry = fit['geometry']
    artifact = manifest['artifacts'][str(layer)]
    if validation == 'sha256':
        record = read_layer(root, layer, fit)
        assert record is not None and record['artifact'] == artifact, f'layer {layer} of {root} is missing or corrupt'
    payload = load_file(str(root / artifact['file']))
    encoder, decoder = payload[ENCODER_KEY], payload[DECODER_KEY]
    groups, _ = head_slices(geometry, tp_rank, tp_size)
    # Head-major AllGather order matches the replicated decoder's first axis.
    return (encoder[groups].to(device=device, dtype=dtype).contiguous(),
            decoder.to(device=device, dtype=dtype).contiguous())


def fold_value_weight(dense_value, encoder):
    groups, dim, rank = encoder.shape
    weight = dense_value.float().reshape(groups, dim, -1)
    return (encoder.float().mT @ weight).reshape(groups * rank, -1).to(dense_value.dtype)


def folded_projection_weights(dense_value, encoder, decoder):
    """Put latent coordinates in ordinary dense V/O slots for quality evaluation."""
    groups, dim, rank = encoder.shape
    heads, _, hidden = decoder.shape
    value = torch.zeros_like(dense_value).reshape(groups, dim, hidden)
    value[:, :rank] = fold_value_weight(dense_value, encoder).reshape(groups, rank, hidden)
    output = decoder.new_zeros(heads, dim, hidden, device=dense_value.device, dtype=dense_value.dtype)
    output[:, :rank] = decoder.to(output)
    return value.flatten(0, 1), output.permute(2, 0, 1).reshape(hidden, heads * dim)




class ChunkedTokenwise(torch.nn.Module):
    def __init__(self, inner, chunk_size=1024):
        super().__init__()
        self.inner = inner
        self.chunk_size = chunk_size

    def forward(self, hidden_states, *tokenwise_inputs):
        if hidden_states.shape[1] <= self.chunk_size:
            return self.inner(hidden_states, *tokenwise_inputs)
        output = torch.empty_like(hidden_states)
        for start in range(0, hidden_states.shape[1], self.chunk_size):
            stop = start + self.chunk_size
            output[:, start:stop] = self.inner(hidden_states[:, start:stop],
                *(value[:, start:stop] for value in tokenwise_inputs))
        return output


def install_chunked_prefill_mlps(model):
    for layer in model.model.layers:
        layer.mlp = ChunkedTokenwise(layer.mlp)


def install_chunked_prefill_norms(model):
    for layer in model.model.layers:
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm)
        layer.post_attention_layernorm = ChunkedTokenwise(layer.post_attention_layernorm)
        layer.self_attn.q_norm = ChunkedTokenwise(layer.self_attn.q_norm)
        layer.self_attn.k_norm = ChunkedTokenwise(layer.self_attn.k_norm)
    model.model.norm = ChunkedTokenwise(model.model.norm)


def chunk_tokenwise_prefill(model, tokens):
    """Run the per-token modules (layer norms, MLPs, final norm) `tokens` rows at a time; exact, and bounds the 128K prefill memory."""
    for layer in model.model.layers:
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=tokens)
        layer.post_attention_layernorm = ChunkedTokenwise(layer.post_attention_layernorm, chunk_size=tokens)
        layer.mlp = ChunkedTokenwise(layer.mlp, chunk_size=tokens)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=tokens)
