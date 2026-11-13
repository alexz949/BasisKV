"""Shared compact-Value attention for Llama and Qwen3."""
import importlib
import torch
from torch import nn
from torch.nn import functional as F
from inference.routing import append_sidecar, routed_key_value

PREFILL_CHUNK = 4096  # tokens per Q/K projection, norm, and RoPE step; bounds the 128K prefill temporaries

def compressed_attention(query, key, value, mask, scale):
    rank = value.shape[-1]
    padded = F.pad(value, (0, query.shape[-1] - rank))
    output = F.scaled_dot_product_attention(query, key, padded, attn_mask=mask,
        is_causal=mask is None and query.shape[2] > 1, dropout_p=0.0, scale=scale, enable_gqa=True)
    return output[..., :rank]

class ValueAttention(nn.Module):
    """Fold the Value encoder into Wv and use the fitted joint output decoder."""

    def __init__(self, native, encoder, decoder):
        super().__init__()
        (self.config, self.layer_idx, self.head_dim) = (native.config, native.layer_idx, native.head_dim)
        (self.heads, self.groups) = (self.config.num_attention_heads, self.config.num_key_value_heads)
        (self.rank, self.scaling) = (encoder.shape[-1], native.scaling)
        (self.q_proj, self.k_proj) = (native.q_proj, native.k_proj)
        self.q_norm = native.q_norm if self.config.model_type == 'qwen3' else nn.Identity()
        self.k_norm = native.k_norm if self.config.model_type == 'qwen3' else nn.Identity()
        self.rotary = importlib.import_module(f'transformers.models.{self.config.model_type}.modeling_{self.config.model_type}').apply_rotary_pos_emb
        (device, dtype) = (native.v_proj.weight.device, native.v_proj.weight.dtype)
        hidden = native.v_proj.in_features
        self.v_proj = nn.Linear(hidden, self.groups * self.rank, bias=native.v_proj.bias is not None, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.heads * self.rank, hidden, bias=native.o_proj.bias is not None, device=device, dtype=dtype)
        e = encoder.to(device=device, dtype=torch.float32)
        weight = native.v_proj.weight.float().reshape(self.groups, self.head_dim, hidden)
        self.v_proj.weight.data.copy_((e.mT @ weight).reshape(self.groups * self.rank, hidden))
        self.o_proj.weight.data.copy_(decoder.to(device).permute(2, 0, 1).reshape(hidden, -1))
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_((e.mT @ native.v_proj.bias.float().reshape(self.groups, self.head_dim, 1)).flatten())
        if self.o_proj.bias is not None:
            self.o_proj.bias.data.copy_(native.o_proj.bias)
        self.key_cache_transform = nn.Identity()
        self.router = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        (batch, length, _) = hidden_states.shape
        (cos, sin) = position_embeddings
        q = hidden_states.new_empty(batch, self.heads, length, self.head_dim)
        k = hidden_states.new_empty(batch, self.groups, length, self.head_dim)
        for start in range(0, length, PREFILL_CHUNK):
            stop = start + PREFILL_CHUNK
            rows = hidden_states[:, start:stop]
            chunk_q = self.q_norm(self.q_proj(rows).view(batch, -1, self.heads, self.head_dim)).transpose(1, 2)
            chunk_k = self.k_norm(self.k_proj(rows).view(batch, -1, self.groups, self.head_dim)).transpose(1, 2)
            (q[:, :, start:stop], k[:, :, start:stop]) = self.rotary(chunk_q, chunk_k, cos[:, start:stop], sin[:, start:stop])
        v = self.v_proj(hidden_states).view(batch, length, self.groups, self.rank).transpose(1, 2)
        if self.router is not None:
            sidecar = append_sidecar(past_key_values, self.layer_idx, v, k, position_embeddings, self.router)
        k = self.key_cache_transform(k)
        if past_key_values is not None:
            (k, v) = past_key_values.update(k, v, self.layer_idx)
        if self.router is not None and length == 1:
            (k, v, attention_mask) = routed_key_value(q, k, v, sidecar, self.router, self.page_size, self.routed_tokens,
                                                      self.recent_tokens, self.scaling, self.pinned_pages)
        output = compressed_attention(q, k, v, attention_mask, self.scaling)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return (self.o_proj(output), None)
