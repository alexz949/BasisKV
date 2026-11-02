"""Historical calibration helpers kept for the archived experiments in others/.

The release stages live in calibration.windows, calibration.value, and calibration.router.
"""
import hashlib
import logging
import json

import torch

from core.page_fisher import FisherStatistics, pack_grams, residual_fisher_gram
from core.router import apply_base, post_rope_rows, restore_base, stack_base, value_codes
from inference.model import install_factors

log = logging.getLogger("basiskv.calibration.legacy")

def candidate_positions(context_length, candidate_stride=64, excluded_query_prefix=32):
    return [p for p in range(candidate_stride-1, context_length, candidate_stride)
            if p >= excluded_query_prefix]


def canonical_hash(manifest):
    content = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def uniform_query_positions(context_length, count=32, terminal_fraction=1.):
    assert 0 < terminal_fraction <= 1 and context_length >= count > 0
    start = int(context_length*(1-terminal_fraction))
    result = [start+(i+1)*(context_length-start)//count-1 for i in range(count)]
    assert len(set(result)) == count
    return result


def select_positions_pivoted_gram(gram, positions, count, tolerance=1e-10):
    gram = (gram.double()+gram.double().T)*.5
    scale = max(float(gram.diag().abs().max()), 1.)
    assert float(torch.linalg.eigvalsh(gram).min()) >= -tolerance*scale
    diagonal = gram.diag().clone()
    diagonal.clamp_min_(0)
    factor = torch.zeros(len(positions), count, dtype=torch.float64, device=gram.device)
    selected = torch.zeros(len(positions), dtype=torch.bool, device=gram.device)
    diagnostics = []
    for step in range(count):
        candidates = diagonal.masked_fill(selected, -torch.inf)
        pivot = int(candidates.argmax())  # positions are sorted: first exact maximum wins.
        energy = float(diagonal[pivot])
        original = float(gram[pivot, pivot])
        diagnostics.append({'absolute_position': positions[pivot], 'pivot_order':step,
                            'residual_energy':energy, 'original_energy':original,
                            'relative_residual_energy':energy/original if original > 0 else 0.,
                            'numerically_zero_pivot':energy <= tolerance*scale})
        selected[pivot] = True
        if energy > tolerance*scale:
            column = gram[:, pivot] - factor[:, :step] @ factor[pivot, :step]
            factor[:, step] = column / energy**.5
            diagonal -= factor[:, step].square()
            assert float(diagonal.min()) >= -tolerance*scale*10
            diagonal.clamp_min_(0)
        diagonal[selected] = 0.
    trace = float(gram.diag().sum())
    return [d['absolute_position'] for d in diagnostics], diagnostics, {
        'residual_trace_fraction': float(diagonal.sum())/trace if trace > 0 else 0.,
        'minimum_residual_diagonal':float(diagonal.min()),
    }


@torch.inference_mode()
def select_stratified_query_positions(queries, positions, *, context_length, num_bins=4, queries_per_bin=8, whitening_eps=1e-06):
    (windows, candidates, heads, dim) = queries.shape
    moment = torch.zeros(heads, dim, dim, dtype=torch.float64, device=queries.device)
    for window in queries:
        q = window.double().permute(1, 0, 2)
        moment += q.transpose(-1, -2) @ q
    moment /= windows * candidates
    (values, vectors) = torch.linalg.eigh((moment + moment.transpose(-1, -2)) * 0.5)
    floor = whitening_eps * values[:, -1:]
    keep = values > floor
    inv = values.clamp_min(torch.finfo(torch.float64).tiny).rsqrt().masked_fill(~keep, 0.0)
    whitening = vectors * inv[:, None] @ vectors.transpose(-1, -2)
    assert torch.isfinite(whitening).all()
    bins = [[i for (i, p) in enumerate(positions) if p * num_bins // context_length == b] for b in range(num_bins)]
    assert all((len(indices) >= queries_per_bin for indices in bins)), 'a query bin has fewer candidate positions than queries per bin; lower --candidate-stride or --queries-per-bin'
    grams = [torch.zeros(len(i), len(i), dtype=torch.float64, device=queries.device) for i in bins]
    for window in queries:
        q = window.double().permute(1, 0, 2) @ whitening
        for (gram, indices) in zip(grams, bins):
            x = q[:, indices]
            gram += (x @ x.transpose(-1, -2)).sum(0) / (windows * heads)
    (records, selected) = ([], [])
    for (b, (gram, indices)) in enumerate(zip(grams, bins)):
        gram.copy_((gram + gram.T) * 0.5)
        local_positions = [positions[i] for i in indices]
        (pivots, diagnostics, residual) = select_positions_pivoted_gram(gram, local_positions, queries_per_bin)
        norm = gram.diag().clamp_min(0).sqrt()
        correlation = gram / (norm[:, None] * norm[None]).clamp_min(torch.finfo(torch.float64).tiny)
        off = ~torch.eye(len(indices), device=gram.device, dtype=torch.bool)
        records.append({'bin': b, 'candidate_count': len(indices), 'pivot_order': pivots, 'selected_positions': sorted(pivots), 'pivots': diagnostics, **residual, 'mean_absolute_offdiagonal_correlation': float(correlation[off].abs().mean()) if off.any() else 0.0, 'minimum_gram_eigenvalue': float(torch.linalg.eigvalsh(gram).min())})
        selected.extend(pivots)
    return ({'selected_positions': sorted(selected), 'bins': records, 'whitening_effective_ranks': keep.sum(-1).cpu().tolist(), 'whitening_eps': whitening_eps}, whitening.float(), grams)


@torch.inference_mode()
def build_multi_query_statistics(queries, rows, *, query_positions, cos, sin, value_encoder, base_maps, page_size, excluded_prefix_pages, device, excluded_recent_tokens=0):
    """Per-query causal Page-Fisher Grams of the base residual, plus base reconstruction diagnostics.

    `queries` is [windows, query positions, heads, dim] and `rows` is
    [windows, tokens, KV groups, 2 dim] holding dense V and post-RoPE K. Output
    order is query-position-major, window-minor. Tokens the deployed selector
    always retains (the pinned prefix pages and the recent window) never compete
    for a routed page, so they are excluded from the causal prefix.
    """
    positions = torch.as_tensor(query_positions, dtype=torch.long).tolist()
    (windows, query_count, heads, dim) = map(int, queries.shape)
    groups = int(rows.shape[2])
    heads_per_group = heads // groups
    stop = max(positions) + 1
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = stack_base(base_maps, device)
    grams = torch.empty(heads, query_count * windows, dim, dim, dtype=torch.float32)
    energy = [0.0] * query_count
    errors = [0.0] * query_count
    key_energy = [0.0] * query_count
    scaling = dim ** (-0.5)
    for window in range(windows):
        log.info('residual features and Fisher Grams: window %d/%d, %d queries', window + 1, windows, query_count)
        current = rows[window, :stop].to(device=device, dtype=torch.float32)
        current_queries = queries[window].to(device=device, dtype=torch.float32)
        exact_key = current[..., dim:]
        base_post = post_rope_rows(apply_base(value_codes(current[..., :dim], encoder), factors), cos[:, :stop], sin[:, :stop])
        residual = exact_key - base_post
        for (index, position) in enumerate(positions):
            prefix = position + 1 - excluded_recent_tokens
            key_energy[index] += float(exact_key[:position + 1].square().sum())
            errors[index] += float(residual[:prefix].square().sum())
            slot = index * windows + window
            for group in range(groups):
                (first, last) = (group * heads_per_group, (group + 1) * heads_per_group)
                (gram, teacher_energy) = residual_fisher_gram(current_queries[index, first:last], exact_key[:prefix, group], residual[:prefix, group], scaling=scaling, page_size=page_size, excluded_prefix_pages=excluded_prefix_pages)
                grams[first:last, slot].copy_(gram.float().cpu())
                energy[index] += teacher_energy
    statistics = FisherStatistics(queries=queries.permute(2, 1, 0, 3).reshape(heads, query_count * windows, dim).contiguous().float(), grams=grams, head_to_kv_group=torch.arange(heads) // heads_per_group, scaling=scaling, teacher_energy=sum(energy))
    reconstruction = {str(position): {'post_rope_relative_mse': errors[index] / key_energy[index], 'residual_squared_error': errors[index], 'exact_key_squared_energy': key_energy[index]} for (index, position) in enumerate(positions)}
    return (statistics, reconstruction)


def score_mse_statistics(q, v, k_post, positions, encoder, base_payload, cos, sin, *, excluded_prefix_tokens, excluded_recent_tokens, heads, groups, dim):
    """Unweighted causal residual-score statistics in the Fisher payload layout."""
    factors = stack_base(restore_base(base_payload), v.device)
    codes = value_codes(v[0].float(), encoder.to(device=v.device, dtype=torch.float32))
    predicted = post_rope_rows(apply_base(codes, factors), cos, sin).permute(1, 0, 2)
    residual = k_post[0].float() - predicted
    running = residual.new_zeros(groups, dim, dim)
    cursor, by_position = excluded_prefix_tokens, {}
    for position in sorted(map(int, positions)):
        stop = position + 1 - excluded_recent_tokens
        rows = residual[:, cursor:stop]
        running.add_(torch.einsum("gtd,gte->gde", rows, rows))
        cursor, by_position[position] = stop, running.clone()
    mapping = torch.arange(heads, device=v.device) // (heads // groups)
    grams = torch.stack([by_position[int(p)] for p in positions], dim=1)[mapping]
    queries = q[0, :, positions].float()
    energy = 0.5 / dim * float(torch.einsum("hqd,hqde,hqe->", queries, grams, queries))
    return dict(queries=queries.cpu(), packed=pack_grams(grams).cpu(), teacher_energy=torch.tensor(energy, dtype=torch.float64)), dict(objective="score_mse", score_energy=energy)


def install_deployed_teacher(model, args, identity):
    """Install the saved teacher before replay; preserve its cache-free prefill path."""
    kind = getattr(model.config, 'text_config', model.config).model_type
    if kind.startswith('qwen3_5'):
        install_factors(model, gated_value_bank=identity['v_bank'], gdn_bank=identity['gdn_bank'])
    else:
        install_factors(model, value_factors=None if args.dense_v else identity['checkpoint'])
    model.eval()
