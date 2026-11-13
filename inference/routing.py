"""Page selection and the FP32 Base/Residual routing cache."""

import math

import torch
from torch.nn import functional as F

from core.router import build_conditional_routing_sidecar


def attention_support(attention_mask, *, batch, query_start, query_stop, query_length, sequence_length, device):
    positions = sequence_length - query_length + torch.arange(query_start, query_stop, device=device)
    valid = (torch.arange(sequence_length, device=device)[None] <= positions[:, None])[None, None]
    valid = valid.expand(batch, 1, -1, -1)
    if attention_mask is None:
        return valid
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.shape[-2] == query_length:
        mask = mask[..., query_start:query_stop, :]
    allowed = mask if mask.dtype == torch.bool else torch.isfinite(mask) & (mask > -1e20)
    return valid & allowed


def select_pages(scores, valid, *, kv_heads, page_size, page_budget, pinned_prefix_pages=0):
    """Normalize page mass per query head, then take the GQA group's maximum."""
    batch, heads, queries, length = scores.shape
    pages = math.ceil(length / page_size)
    padding = pages * page_size - length
    valid = valid.expand_as(scores)
    scores = F.pad(scores.float().masked_fill(~valid, -torch.inf), (0, padding), value=-torch.inf)
    valid = F.pad(valid, (0, padding), value=False).reshape(batch, heads, queries, pages, page_size).any(-1)
    candidates = valid & (torch.arange(pages, device=scores.device) >= pinned_prefix_pages)
    mass = scores.reshape(batch, heads, queries, pages, page_size).logsumexp(-1).masked_fill(~candidates, -torch.inf)
    mass = torch.where(candidates.any(-1, keepdim=True), mass, torch.zeros_like(mass))
    mass = mass.softmax(-1).masked_fill(~candidates, 0)
    shape = (batch, kv_heads, heads // kv_heads, queries, pages)
    group_valid = valid.reshape(shape).any(2)
    group_candidates = candidates.reshape(shape).any(2)
    mass = mass.reshape(shape).amax(2).masked_fill(~group_candidates, -torch.inf)
    pinned = min(pinned_prefix_pages, page_budget, pages)
    count = min(page_budget - pinned, pages - pinned)
    ids = mass.topk(count, dim=-1, sorted=False).indices
    selected_valid = group_candidates.gather(-1, ids)
    if pinned:
        prefix = torch.arange(pinned, device=scores.device).expand(batch, kv_heads, queries, -1)
        ids = torch.cat((prefix, ids), -1)
        selected_valid = torch.cat((group_valid.gather(-1, prefix), selected_valid), -1)
    return ids, selected_valid


def selected_support(scores, page_size, routed_tokens, recent_tokens=64, pinned_pages=0):
    """Historical page budget plus an additional exact recent tail.

    `pinned_pages` leading pages are always selected and count toward the page budget,
    as in the Appendix Table 8 Page32 control; the main protocol pins none.
    """
    batch, groups, heads, length = scores.shape
    if length <= routed_tokens + recent_tokens:
        ids = torch.arange(length, device=scores.device).expand(batch, groups, -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)
    historical = length - recent_tokens
    proxy = scores[..., :historical].reshape(batch, groups * heads, 1, historical)
    pages, page_valid = select_pages(proxy, torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=groups, page_size=page_size, page_budget=routed_tokens // page_size, pinned_prefix_pages=pinned_pages)
    ids = (pages.squeeze(2)[..., None] * page_size + torch.arange(page_size, device=scores.device)).flatten(-2)
    valid = (ids < historical) & page_valid.squeeze(2).repeat_interleave(page_size, -1)
    recent = torch.arange(historical, length, device=scores.device).expand(batch, groups, -1)
    return torch.cat((ids, recent), -1), torch.cat((valid, torch.ones_like(recent, dtype=torch.bool)), -1)


def append_sidecar(cache, layer, value, key, position_embeddings, router):
    """Append this step's [predicted K, residual code] rows, computed and stored in the cache dtype as in the paper."""
    cos, sin = (None, None) if position_embeddings is None else position_embeddings
    code = build_conditional_routing_sidecar(value, key, cos=cos, sin=sin,
        **{name: router[name] for name in ('base_left', 'base_right', 'base_bias', 'residual_encoder')})
    if not hasattr(cache, '_basiskv_sidecars'):
        cache._basiskv_sidecars = {}
    previous = cache._basiskv_sidecars.get(layer)
    cache._basiskv_sidecars[layer] = code if previous is None else torch.cat((previous, code), 2)
    return cache._basiskv_sidecars[layer]


def routed_key_value(query, key, value, sidecar, router, page_size, routed_tokens, recent_tokens, scale, pinned_pages=0):
    batch, heads, _, _ = query.shape
    groups = key.shape[1]
    projected = torch.einsum('bhqd,hdr->bhqr', query.float(), router['projector'])
    projected = projected[:, :, 0].reshape(batch, groups, heads // groups, -1)
    scores = (projected @ sidecar.float().transpose(-1, -2)) * scale  # FP32 scores over the stored sidecar
    ids, valid = selected_support(scores, page_size, routed_tokens, recent_tokens, pinned_pages)
    ids = ids.clamp_max(key.shape[2] - 1)
    key = key.gather(2, ids[..., None].expand(-1, -1, -1, key.shape[-1]))
    value = value.gather(2, ids[..., None].expand(-1, -1, -1, value.shape[-1]))
    mask = valid.repeat_interleave(heads // groups, 1).unsqueeze(2)
    return key, value, mask
