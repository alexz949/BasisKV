"""Shared resident and host-Key attention for the decode and memory benchmarks.

CUDA routing uses GQA4 / QK128 / B16R16 with 1–32-token power-of-two pages, no pinned prefix, and
2,048 historical tokens plus 64 recent tokens. Value factors use core.fit_value.
"""

import math
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from inference.model import router_factors
from core.router import align_value_basis
from inference.tp import ChunkedTokenwise, fold_value_weight, head_slices, load_layer, load_manifest
from kernels.allgather import AllGather
from kernels.decode import load_paged_attention, load_router
from kernels.routing_config import PINNED_PAGES, ROUTED_TOKENS, RECENT_TOKENS, SUPPORT

SPLITS = 16


def local_weight(weight):
    return weight.to_local() if hasattr(weight, 'to_local') else weight


class Attention(nn.Module):
    def __init__(self, original, *, batch, capacity, heads, groups, encoder, decoder,
                 factors, router, paged, communicator, shared_rope, host_keys, page_size):
        super().__init__()
        self.heads, self.groups, self.dim = heads, groups, original.head_dim
        self.width = self.dim if encoder is None else encoder.shape[-1]
        self.capacity, self.length = capacity, 0
        self.page_size = page_size
        self.scaling = original.scaling
        self.q_norm = getattr(original, 'q_norm', nn.Identity())
        self.k_norm = getattr(original, 'k_norm', nn.Identity())
        self.router, self.factors, self.paged = router, factors, paged
        self.communicator, self.shared_rope = communicator, shared_rope
        self.profile_events = None
        self.base_in_value = encoder is not None and router is not None
        device, dtype = original.q_proj.weight.device, original.q_proj.weight.dtype
        weights = [local_weight(proj.weight).detach() for proj in (original.q_proj, original.k_proj, original.v_proj)]
        if encoder is not None:
            weights[2] = fold_value_weight(weights[2], encoder)
        self.register_buffer('qkv_weight', torch.cat(weights).contiguous(), persistent=False)
        self.register_buffer('decoder', None if decoder is None else decoder.flatten(0, 1), persistent=False)
        self.o_proj = original.o_proj if decoder is None else None
        shape = (batch, groups, capacity)
        self.value_cache = torch.empty(*shape, self.width, device=device, dtype=dtype)
        self.host_key = router.host_key_buffer(batch, groups, capacity) if host_keys else None
        self.key_pointer = router.device_pointer(self.host_key) if host_keys else 0
        self.key_cache = None if host_keys else torch.empty(*shape, self.dim, device=device, dtype=dtype)
        self.output = torch.empty(batch, heads, 1, self.width, device=device, dtype=dtype)
        if communicator is not None:
            self.decode_plan = communicator.prepare(heads * self.width, tokens=batch, dtype=dtype)
            self.output = self.decode_plan.local.T.view(batch, heads, 1, self.width)
        if router is None:
            self.page_ids = torch.arange(math.ceil(capacity / self.page_size), device=device)[None, None].expand(batch, groups, -1).contiguous()
            self.dense_workspace = torch.empty(batch * heads, SPLITS, self.width + 2, device=device, dtype=torch.float32)
        else:
            self.partial = torch.empty(batch, heads, SPLITS, self.width, device=device, dtype=torch.float32)
            self.lse = torch.empty(batch, heads, SPLITS, device=device, dtype=torch.float32)
            self.base_cache = self.value_cache[..., :16] if self.base_in_value else torch.empty(*shape, 16, device=device, dtype=dtype)
            self.residual_cache = torch.empty_like(self.base_cache)
            self.query_code = torch.empty(batch, groups, 4, 16, device=device, dtype=dtype)
            self.scores = torch.empty(batch, groups, 4, math.ceil(capacity / self.page_size), device=device)
            self.pages = torch.empty(batch, groups, ROUTED_TOKENS // self.page_size, device=device, dtype=torch.long)
            self.selection_scratch = torch.empty(batch, groups, router.selection_workspace_size(), device=device, dtype=torch.float32)
            self.support_ids = torch.empty(batch, groups, SUPPORT, device=device, dtype=torch.long)
            if host_keys:
                self.slot_keys = torch.empty(batch, groups, SUPPORT, self.dim, device=device, dtype=dtype)
                self.resident = torch.full_like(self.support_ids, -1)
                self.lookup = torch.full((batch, groups, capacity), -1, device=device, dtype=torch.int32)
                self.selected_slots = torch.empty_like(self.support_ids)
                self.missing = torch.empty_like(self.support_ids, dtype=torch.int32)
                self.counts = torch.empty(batch, groups, 2, device=device, dtype=torch.int32)

    def reset(self):
        self.length = 0
        if self.host_key is not None:
            self.resident.fill_(-1)
            self.lookup.fill_(-1)

    def project_output(self, attention, decode):
        batch, heads, tokens, width = attention.shape
        if self.decoder is None:
            return self.o_proj(attention.transpose(1, 2).reshape(batch, tokens, heads * width))
        if self.communicator is None:
            return attention.transpose(1, 2).reshape(batch, tokens, -1) @ self.decoder
        plan = self.decode_plan if decode else self.communicator.prepare(heads * width, tokens=batch * tokens, dtype=attention.dtype)
        if not decode:
            plan.local.copy_(attention.permute(1, 3, 0, 2).reshape(heads * width, batch * tokens))
        return (plan.gather().T @ self.decoder).view(batch, tokens, -1)

    def project(self, hidden, which):
        sizes = (self.heads * self.dim, self.groups * self.dim, self.groups * self.width)
        offset = sum(sizes[:which])
        value = F.linear(hidden, self.qkv_weight[offset:offset + sizes[which]])
        heads = self.heads if which == 0 else self.groups
        width = self.width if which == 2 else self.dim
        value = value.view(hidden.shape[0], hidden.shape[1], heads, width)
        if which < 2:
            value = (self.q_norm if which == 0 else self.k_norm)(value)
        return value.transpose(1, 2)

    def append(self, key, value, cos, sin, start, decode):
        stop = start + key.shape[2]
        if decode and self.router is not None:
            f = self.factors
            self.router.append_decode(key=key, value=value,
                base_left=f['base_left_b16'], base_right=f['base_right_b16'], base_bias=f['base_bias_b16'],
                residual_encoder=f['residual_encoder_b16_r16'], rope_cos=cos[0, :, :64].contiguous(),
                rope_sin=sin[0, :, :64].contiguous(), value_cache=self.value_cache,
                base_cache=self.base_cache, residual_cache=self.residual_cache,
                start=start, host_key_pointer=self.key_pointer)
        else:
            self.value_cache[:, :, start:stop].copy_(value)
            if self.router is not None:
                f = self.factors
                base = value[..., :16] if self.base_in_value else (value @ f['base_left_b16']).to(value.dtype)
                predicted = (base @ f['base_right_b16'] + f['base_bias_b16'][None, :, None]).to(value.dtype)
                _, predicted = apply_rotary_pos_emb(predicted, predicted, cos, sin)
                self.base_cache[:, :, start:stop].copy_(base)
                self.residual_cache[:, :, start:stop].copy_((key - predicted) @ f['residual_encoder_b16_r16'])
            if self.host_key is not None:
                self.router.append_host_key(self.host_key, key, start)
        if self.key_cache is not None:
            self.key_cache[:, :, start:stop].copy_(key)

    def decode_attention(self, query, end):
        if self.router is None:
            return self.paged.attention(self.key_cache, query, self.value_cache,
                self.page_ids[:, :, :math.ceil(end / self.page_size)], self.dense_workspace, self.output,
                end, self.scaling, SPLITS)
        from kernels.sparse_attention import sparse_attention

        historical = end - RECENT_TOKENS
        scores = self.scores[..., :math.ceil(historical / self.page_size)]
        f = self.factors
        self.router.page_lse(query, self.base_cache[:, :, :historical], self.residual_cache[:, :, :historical],
            f['base_right_b16'], f['base_bias_b16'], f['residual_query_b16_r16'],
            self.shared_rope['cos'][:historical], self.shared_rope['sin'][:historical],
            self.query_code, scores, self.scaling)
        if self.host_key is None:
            self.router.select_pages(scores, self.pages, self.support_ids, self.selection_scratch, historical, end)
            keys, slots = self.key_cache, self.support_ids
        else:
            self.router.select_pack_refresh(scores, self.pages, self.support_ids, self.key_pointer,
                self.slot_keys, self.resident, self.lookup, self.selected_slots, self.missing, self.counts,
                self.selection_scratch, historical, end)
            keys, slots = self.slot_keys, self.selected_slots
        return sparse_attention(query, keys, self.value_cache, self.support_ids, key_slots=slots,
            workspace=(self.partial, self.lse, self.output), scale=self.scaling)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        batch, tokens, _ = hidden_states.shape
        start, end = self.length, self.length + tokens
        assert end <= self.capacity and (start == 0 or tokens == 1), 'the harness prefills once, then decodes one token at a time within capacity'
        cos, sin = position_embeddings
        if self.write_rope:
            self.shared_rope['cos'][start:end].copy_(cos[0, :, :64])
            self.shared_rope['sin'][start:end].copy_(sin[0, :, :64])
        if start:
            packed = F.linear(hidden_states, self.qkv_weight)
            raw_query, raw_key, value = packed.split((self.heads * self.dim, self.groups * self.dim, self.groups * self.width), dim=-1)
            raw_query = self.q_norm(raw_query.view(batch, 1, self.heads, self.dim)).transpose(1, 2)
            raw_key = self.k_norm(raw_key.view(batch, 1, self.groups, self.dim)).transpose(1, 2)
            query, key = apply_rotary_pos_emb(raw_query, raw_key, cos, sin)
            value = value.view(batch, 1, self.groups, self.width).transpose(1, 2)
            if self.profile_events is not None:
                self.profile_events[0].record()
            self.append(key, value, cos, sin, start, True)
            attention = self.decode_attention(query, end)
            if self.profile_events is not None:
                self.profile_events[1].record()
            hidden_states.copy_(self.project_output(attention, True))
        else:
            # Build all K/V before overwriting input chunks with attention output.
            key = torch.empty(batch, self.groups, tokens, self.dim, device=hidden_states.device, dtype=hidden_states.dtype)
            padded_value = torch.zeros_like(key)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                block = hidden_states[:, left:right]
                raw_key = self.project(block, 1)
                _, post_key = apply_rotary_pos_emb(raw_key, raw_key, cos[:, left:right], sin[:, left:right])
                value = self.project(block, 2)
                key[:, :, left:right].copy_(post_key)
                padded_value[:, :, left:right, :self.width].copy_(value)
                self.append(post_key, value, cos[:, left:right], sin[:, left:right], left, False)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                query = self.project(hidden_states[:, left:right], 0)
                query, _ = apply_rotary_pos_emb(query, query, cos[:, left:right], sin[:, left:right])
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(query, key[:, :, :right], padded_value[:, :, :right],
                        attn_mask=causal_lower_right(right - left, right), scale=self.scaling, enable_gqa=True)[..., :self.width]
                hidden_states[:, left:right].copy_(self.project_output(attention, False))
        self.length = end
        return hidden_states, None


def decoder_layer_forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
    residual = hidden_states
    hidden_states, _ = self.self_attn(self.input_layernorm(hidden_states), position_embeddings=position_embeddings)
    hidden_states.add_(residual)
    for left in range(0, hidden_states.shape[1], 1024):
        block = hidden_states[:, left:left + 1024]
        block.add_(self.mlp(self.post_attention_layernorm(block)))
    return hidden_states


def install(model, args, *, batch, capacity, rank=0, world=1, host_keys=False):
    config = model.config
    assert config.model_type in ('llama', 'qwen3') and config.head_dim == 128, 'the kernels are compiled for Llama/Qwen3 attention with head_dim 128'
    geometry = dict(model_type=config.model_type, num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim, hidden_size=config.hidden_size)
    groups, heads = head_slices(geometry, rank, world)
    local_heads, local_groups = heads.stop - heads.start, groups.stop - groups.start
    assert not host_keys or args.arm == 'basis', 'host-resident Keys belong to the basis arm'
    manifest = load_manifest(args.value_factors) if args.value_factors else None
    assert manifest is None or len(set(manifest['layer_ranks'])) == 1, 'the HF TP runtime serves uniform-rank banks; benchmarks.tp serves allocated ones'
    width = config.head_dim if manifest is None else manifest['layer_ranks'][0]
    assert width in (64, 96, 128), 'the router kernel is compiled for Value widths 64, 96, and 128'
    router = paged = communicator = None
    if args.arm == 'basis':
        assert args.router_dir is not None and local_heads == local_groups * 4, 'basis needs --router-dir and the GQA-4 kernel build'
        router = load_router(width, page_size=args.page_size)
    else:
        paged = load_paged_attention(width, local_heads // local_groups, page_size=args.page_size)
    if manifest is not None and world > 1:
        communicator = AllGather.from_distributed(device=next(model.parameters()).device)
        communicator.configure_workspace(tokens=batch * min(capacity, 2048), max_total_width=geometry['num_heads'] * width, dtype=torch.bfloat16)
    shared_rope = None if router is None else {key: torch.empty(capacity, 64, device=next(model.parameters()).device, dtype=torch.bfloat16) for key in ('cos', 'sin')}
    layers = []
    for index, layer in enumerate(model.model.layers):
        encoder = decoder = factors = None
        if manifest is not None:
            encoder, decoder = load_layer(args.value_factors, manifest, index, rank, tp_size=world,
                device=next(model.parameters()).device, dtype=torch.bfloat16)
        if router is not None:
            payload, protocol = router_factors(args.router_dir, index)
            base, residual = protocol['base_rank'], protocol['residual_rank']
            assert base == residual == 16, 'the register router kernel is compiled for B16R16'
            assert protocol['page_size'] == args.page_size and protocol['excluded_prefix_pages'] == PINNED_PAGES
            assert protocol['excluded_recent_tokens'] == RECENT_TOKENS
            if encoder is not None:
                # Orthogonal Value rotation: c'=cQ, D'=Q^T D; the first 16 coordinates supply Base.
                left = payload['base_left_b16'].double()
                rotation, right = align_value_basis(left, payload['base_right_b16'].double())
                encoder = (encoder.double() @ rotation[groups].to(encoder.device)).to(encoder.dtype)
                decoder = (rotation.mT.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=0).to(decoder.device) @ decoder.double()).to(decoder.dtype)
                payload['base_right_b16'] = right
                selector = torch.zeros_like(left)
                selector[:, :16] = torch.eye(16, dtype=selector.dtype)
                payload['base_left_b16'] = selector
            factors = {name: tensor[heads if name.startswith('residual_query') else groups].to(device=next(model.parameters()).device, dtype=torch.bfloat16).contiguous()
                       for name, tensor in payload.items() if name in ('base_left_b16', 'base_right_b16', 'base_bias_b16', 'residual_encoder_b16_r16', 'residual_query_b16_r16')}
        attention = Attention(layer.self_attn, batch=batch, capacity=capacity, heads=local_heads, groups=local_groups,
            encoder=encoder, decoder=decoder, factors=factors, router=router, paged=paged,
            communicator=communicator, shared_rope=shared_rope, host_keys=host_keys, page_size=args.page_size)
        layer.self_attn = attention
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.forward = MethodType(decoder_layer_forward, layer)
        attention.write_rope = router is not None and not layers
        layers.append(attention)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return layers, communicator
