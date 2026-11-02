"""Capture Base moments and Page-Fisher statistics for core.fit_router.

    python -m calibration.router moments --model MODEL --value-factors FACTORS --windows WINDOWS --output CAPTURE
    python -m core.fit_router base --statistics-dir CAPTURE --output-dir CAPTURE/base --base-rank 16
    python -m calibration.router fisher --model MODEL --value-factors FACTORS --windows WINDOWS --output CAPTURE
    python -m core.fit_router fit --statistics-dir CAPTURE --output-dir ROUTER --residual-rank 16

The teacher is the deployed model, with the Value factors installed (Qwen3.5: its gated-V
and GDN banks). `moments` accumulates the dense-V / pre-RoPE-K second moments that
`fit_router base` solves and keeps the queries at the candidate grid, from which the
representative query positions are chosen (Appendix B.3: whitened queries, diagonal-pivoted
Cholesky within four positional strata, 16 fit and 8 diagnostic positions each). `fisher`
replays the windows with the fitted Base and writes one record per window holding the
Page-Fisher Grams at those positions. --dense-v is the dense-Value control of Table 11: a
dense teacher with identity Value codes. The protocol (page size, pinned prefix, ranks,
query grid, window ids, geometry) is stored with every record, and all records of one
capture directory must share it. Shard the moments replay with --window-shard-index /
--window-shard-count and rerun with --merge; Fisher records need no merge.
"""

import argparse
import gc
import importlib
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calibration.common import add_model_arguments, add_window_arguments, load_teacher, read_windows, setup
from core.page_fisher import FisherStatistics, residual_fisher_gram
from core.router import apply_base, post_rope_rows, restore_base, stack_base, value_codes
from core.router_io import encoder_for, layer_file, load_record, pack_statistics, save_record
from inference.model import position_embeddings

log = logging.getLogger("basiskv.router")
FORMAT = "basiskv.calibration.router.v1"
MOMENTS_FORMAT = "basiskv.router_moments.v1"
RECENT_TOKENS = 64  # the deployed selector always keeps this tail, so it never competes for a routed page
CANDIDATE_STRIDE = 16
QUERY_BINS, FIT_QUERIES_PER_BIN, DIAGNOSTIC_QUERIES_PER_BIN = 4, 16, 8
CHUNK_ROWS = 2048


def candidate_positions(context_length, excluded_prefix):
    """Every CANDIDATE_STRIDE-th position whose causal prefix reaches past the pinned pages and the recent tail."""
    return [p for p in range(CANDIDATE_STRIDE - 1, context_length, CANDIDATE_STRIDE) if p >= excluded_prefix]


def pivoted_cholesky_positions(gram, positions, count):
    """Diagonal-pivoted Cholesky on the whitened-query Gram: `count` pivot positions, greedy by residual energy.
    Equivalent to column-pivoted QR on the whitened feature matrix."""
    gram = (gram + gram.T) * 0.5
    scale = max(float(gram.diag().abs().max()), 1.0)
    assert float(torch.linalg.eigvalsh(gram).min()) >= -1e-10 * scale, "the query Gram is not positive semidefinite"
    diagonal = gram.diag().clamp_min(0)
    factor = torch.zeros(len(positions), count, dtype=gram.dtype, device=gram.device)
    selected = torch.zeros(len(positions), dtype=torch.bool, device=gram.device)
    pivots = []
    for step in range(count):
        pivot = int(diagonal.masked_fill(selected, -torch.inf).argmax())  # positions are sorted: the first exact maximum wins
        energy = float(diagonal[pivot])
        pivots.append(positions[pivot])
        selected[pivot] = True
        if energy > 1e-10 * scale:
            column = gram[:, pivot] - factor[:, :step] @ factor[pivot, :step]
            factor[:, step] = column / energy ** 0.5
            diagonal = diagonal - factor[:, step].square()
            assert float(diagonal.min()) >= -1e-9 * scale, "the pivoted Cholesky lost positive semidefiniteness"
            diagonal = diagonal.clamp_min(0)
        diagonal[selected] = 0.0
    return pivots


@torch.inference_mode()
def select_query_positions(queries, positions, *, context_length, per_bin):
    """Appendix B.3: whiten the queries per head, then pick `per_bin` pivot positions in each of QUERY_BINS strata.

    `queries` is [windows, candidates, heads, dim] at the candidate `positions`; returns the sorted positions.
    """
    windows, candidates, heads, dim = queries.shape
    moment = torch.zeros(heads, dim, dim, dtype=torch.float64, device=queries.device)
    for window in queries:
        q = window.double().permute(1, 0, 2)
        moment += q.transpose(-1, -2) @ q
    moment /= windows * candidates
    values, vectors = torch.linalg.eigh((moment + moment.transpose(-1, -2)) * 0.5)
    keep = values > 1e-6 * values[:, -1:]
    inverse_root = values.clamp_min(torch.finfo(torch.float64).tiny).rsqrt().masked_fill(~keep, 0.0)
    whitening = vectors * inverse_root[:, None] @ vectors.transpose(-1, -2)
    assert torch.isfinite(whitening).all(), "query whitening is not finite"
    bins = [[i for i, p in enumerate(positions) if p * QUERY_BINS // context_length == b] for b in range(QUERY_BINS)]
    assert all(len(indices) >= per_bin for indices in bins), "a positional stratum has fewer candidate positions than queries to select; use longer windows"
    grams = [torch.zeros(len(indices), len(indices), dtype=torch.float64, device=queries.device) for indices in bins]
    for window in queries:
        q = window.double().permute(1, 0, 2) @ whitening
        for gram, indices in zip(grams, bins):
            x = q[:, indices]
            gram += (x @ x.transpose(-1, -2)).sum(0) / (windows * heads)
    selected = []
    for gram, indices in zip(grams, bins):
        selected.extend(pivoted_cholesky_positions(gram, [positions[i] for i in indices], per_bin))
    return sorted(selected)


class BaseMoments:
    """FP64 sums and second moments of dense Values and pre-RoPE Keys per KV group."""

    def __init__(self, groups, dim):
        self.count = 0
        self.sum_v = torch.zeros(groups, dim, dtype=torch.float64)
        self.sum_k = torch.zeros_like(self.sum_v)
        self.vv = torch.zeros(groups, dim, dim, dtype=torch.float64)
        self.vk = torch.zeros_like(self.vv)
        self.kk = torch.zeros_like(self.vv)

    def update(self, value, pre_key):
        for start in range(0, len(value), CHUNK_ROWS):
            v = value[start : start + CHUNK_ROWS].double().transpose(0, 1)
            k = pre_key[start : start + CHUNK_ROWS].double().transpose(0, 1)
            self.sum_v += v.sum(1).cpu()
            self.sum_k += k.sum(1).cpu()
            self.vv += (v.mT @ v).cpu()
            self.vk += (v.mT @ k).cpu()
            self.kk += (k.mT @ k).cpu()
            self.count += v.shape[1]

    def tensors(self):
        return dict(count=torch.tensor(self.count), sum_v=self.sum_v, sum_k=self.sum_k, vv=self.vv, vk=self.vk, kk=self.kk)


@torch.inference_mode()
def fisher_statistics(queries, rows, *, positions, cos, sin, encoder, base, page_size, pinned_pages, device):
    """Page-Fisher Grams of the Base residual for every (query position, window), query-position-major.

    `queries` is [windows, positions, heads, dim]; `rows` is [windows, tokens, KV groups, 2 dim] holding
    dense V and post-RoPE K. The pinned prefix pages and the recent tail are excluded from the causal prefix.
    """
    windows, count, heads, dim = map(int, queries.shape)
    groups = int(rows.shape[2])
    per_group = heads // groups
    stop = max(positions) + 1
    encoder = encoder.to(device=device, dtype=torch.float32)
    factors = stack_base(base, device)
    grams = torch.empty(heads, count * windows, dim, dim, dtype=torch.float32)
    energy, scaling = 0.0, dim ** -0.5
    for window in range(windows):
        current = rows[window, :stop].to(device=device, dtype=torch.float32)
        current_queries = queries[window].to(device=device, dtype=torch.float32)
        exact_key = current[..., dim:]
        predicted = post_rope_rows(apply_base(value_codes(current[..., :dim], encoder), factors), cos[:, :stop], sin[:, :stop])
        residual = exact_key - predicted
        for index, position in enumerate(positions):
            prefix = position + 1 - RECENT_TOKENS
            slot = index * windows + window
            for group in range(groups):
                first, last = group * per_group, (group + 1) * per_group
                gram, teacher_energy = residual_fisher_gram(
                    current_queries[index, first:last], exact_key[:prefix, group], residual[:prefix, group],
                    scaling=scaling, page_size=page_size, excluded_prefix_pages=pinned_pages)
                grams[first:last, slot].copy_(gram.float().cpu())
                energy += teacher_energy
    return FisherStatistics(
        queries=queries.permute(2, 1, 0, 3).reshape(heads, count * windows, dim).contiguous().float(), grams=grams,
        head_to_kv_group=torch.arange(heads) // per_group, scaling=scaling, teacher_energy=energy)


def routing_rows(attention, hidden, kwargs, dense, geometry):
    """Native Q/K normalization and rotation; dense V from the saved projection for the Base moments."""
    native = getattr(attention, "native", attention)
    batch, tokens, _ = hidden.shape
    heads, groups, dim = geometry.num_heads, geometry.num_kv_heads, geometry.head_dim
    q = native.q_proj(hidden)
    q = q.view(batch, tokens, heads, 2 * dim).chunk(2, dim=-1)[0] if geometry.model_type == "qwen3_5" else q.view(batch, tokens, heads, dim)
    k = native.k_proj(hidden).view(batch, tokens, groups, dim)
    v = F.linear(hidden, *dense).view(batch, tokens, groups, dim)
    if geometry.model_type in ("qwen3", "qwen3_5"):
        q, k = native.q_norm(q), native.k_norm(k)
    name = geometry.model_type
    rotary = importlib.import_module(f"transformers.models.{name}.modeling_{name}").apply_rotary_pos_emb
    q, post = rotary(q.transpose(1, 2), k.transpose(1, 2), *kwargs["position_embeddings"])
    return q, k, post, v


def protocol(args, geometry, split_ids):
    settings = dict(
        format=FORMAT, base_rank=args.base_rank, residual_rank=args.residual_rank, page_size=args.page_size,
        excluded_prefix_pages=args.pinned_pages, excluded_recent_tokens=RECENT_TOKENS,
        query_grid_prefix=args.page_size * args.pinned_pages + RECENT_TOKENS, candidate_stride=CANDIDATE_STRIDE,
        query_bins=QUERY_BINS, fit_queries=QUERY_BINS * FIT_QUERIES_PER_BIN,
        diagnostic_queries=QUERY_BINS * DIAGNOSTIC_QUERIES_PER_BIN if split_ids["heldout"] else 0,
        sequence_length=args.sequence_length, fit_ids=split_ids["fit"], diagnostic_ids=split_ids["heldout"],
        dense_v=args.dense_v, geometry=geometry.to_dict())
    return json.loads(json.dumps(settings))  # the persisted form, so it compares equal to loaded records


def write_moments(args, layer, tensors, candidates, grid, settings):
    selections = {"fit": select_query_positions(candidates, grid, context_length=args.sequence_length, per_bin=FIT_QUERIES_PER_BIN)}
    if settings["diagnostic_ids"]:
        selections["heldout"] = select_query_positions(candidates, grid, context_length=args.sequence_length, per_bin=DIAGNOSTIC_QUERIES_PER_BIN)
    save_record(layer_file(args.output, "moments", layer), tensors,
                dict(protocol=settings, layer=layer, selections=selections, statistics_format=MOMENTS_FORMAT))


@torch.inference_mode()
def capture(args, windows, split_ids):
    if args.dense_v:
        banks = {}
    elif args.gated_value_bank is not None:
        banks = dict(gated_value_bank=args.gated_value_bank, gdn_bank=args.gdn_bank)
    else:
        banks = dict(value_factors=args.value_factors)
    model, config, geometry, attention, dense = load_teacher(args, **banks)
    assert geometry.model_type in ("llama", "qwen3", "qwen3_5"), f"unsupported model: {geometry.model_type}"
    settings = protocol(args, geometry, split_ids)
    ids = {split: values[args.window_shard_index :: args.window_shard_count] for split, values in split_ids.items()}
    grid = candidate_positions(args.sequence_length, settings["query_grid_prefix"])
    encoders = {}
    if args.gated_value_bank is not None and not args.dense_v:
        from inference.qwen35 import load_bank
        encoders = {int(index): factors["E_V"] for index, factors in load_bank(args.gated_value_bank)["layers"].items()}
    moments, candidates, bases, selections, phases, handles, active = {}, {}, {}, {}, {}, [], {}
    for layer in attention:
        if args.stage == "fisher":
            bases[layer], metadata = load_record(layer_file(args.output, "base", layer))
            _, moment_metadata = load_record(layer_file(args.output, "moments", layer))
            assert metadata["protocol"] == moment_metadata["protocol"] == settings, "the base or moments records come from another capture"
            selections[layer] = moment_metadata["selections"]
        else:
            moments[layer] = {split: BaseMoments(geometry.num_kv_heads, geometry.head_dim) for split in ids}
            candidates[layer] = torch.empty(len(ids["fit"]), len(grid), geometry.num_heads, geometry.head_dim, dtype=torch.bfloat16)
            if layer not in encoders:
                encoders[layer] = encoder_for(None if args.dense_v else args.value_factors, layer, geometry)

    def observe(layer):
        def hook(module, positional, kwargs):
            split, index, slot = active["split"], active["index"], active["slot"]
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else positional[0]
            q, k, post, v = routing_rows(module, hidden, kwargs, dense[layer], geometry)
            if args.stage == "moments":
                moments[layer][split].update(v[0], k[0])
                if split == "fit":
                    candidates[layer][slot].copy_(q[0, :, grid].transpose(0, 1).cpu())
                return
            if geometry.model_type == "qwen3_5":
                cos, sin = (value.float() for value in kwargs["position_embeddings"])
            else:
                if hidden.device not in phases:
                    phases[hidden.device] = position_embeddings(config, args.sequence_length, hidden.device)
                cos, sin = phases[hidden.device]
            positions = selections[layer][split]
            statistics = fisher_statistics(
                q[:, :, positions].transpose(1, 2), torch.cat((v, post.transpose(1, 2)), -1), positions=positions,
                cos=cos, sin=sin, encoder=bases[layer]["encoder"], base=restore_base(bases[layer]),
                page_size=args.page_size, pinned_pages=args.pinned_pages, device=hidden.device)
            save_record(args.output / "fisher" / f"l{layer:03d}" / f"w{index:03d}.safetensors", pack_statistics(statistics),
                        dict(protocol=settings, layer=layer, window_id=index, split=split))
        return hook

    handles = [module.register_forward_pre_hook(observe(layer), with_kwargs=True) for layer, module in attention.items()]
    device = model.get_input_embeddings().weight.device
    for split, indices in ids.items():
        for slot, index in enumerate(indices):
            active.update(split=split, index=index, slot=slot)
            started = time.monotonic()
            output = model.model(input_ids=windows[index : index + 1].to(device, torch.long), use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all(), "non-finite teacher output"
            del output
            log.info("%s %s window %d: %.1fs", args.stage, split, index, time.monotonic() - started)
    for handle in handles:
        handle.remove()
    if args.stage == "moments":
        for layer in attention:
            tensors = {f"{split}_{name}": value for split, moment in moments[layer].items() for name, value in moment.tensors().items()}
            tensors["encoder"] = encoders[layer]
            if args.window_shard_count > 1:
                tensors["fit_candidates"] = candidates[layer]
                save_record(args.output / "parts" / f"layer_{layer:03d}_s{args.window_shard_index:03d}.safetensors", tensors,
                            dict(protocol=settings, layer=layer, ids=ids, window_shard_count=args.window_shard_count))
            else:
                write_moments(args, layer, tensors, candidates[layer], grid, settings)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merge(args, split_ids):
    for first in sorted((args.output / "parts").glob("layer_*_s000.safetensors")):
        layer = int(first.stem.split("_")[1])
        combined, slots, seen, settings = {}, {}, {split: [] for split in split_ids}, None
        for shard in range(args.window_shard_count):
            tensors, metadata = load_record(first.with_name(f"layer_{layer:03d}_s{shard:03d}.safetensors"))
            settings = settings or metadata["protocol"]
            assert metadata["protocol"] == settings and metadata["window_shard_count"] == args.window_shard_count, (
                f"layer {layer} shard {shard} comes from another capture")
            for split, indices in metadata["ids"].items():
                seen[split].extend(indices)
            for key, value in tensors.items():
                if key == "encoder":
                    combined[key] = value
                elif key == "fit_candidates":
                    slots.update(zip(metadata["ids"]["fit"], value))
                else:
                    combined[key] = combined.get(key, 0) + value
        assert all(sorted(seen[split]) == split_ids[split] for split in split_ids), f"layer {layer}: missing or repeated windows"
        assert settings["sequence_length"] == args.sequence_length and settings["base_rank"] == args.base_rank, "merge options differ from the capture"
        grid = candidate_positions(args.sequence_length, settings["query_grid_prefix"])
        write_moments(args, layer, combined, torch.stack([slots[index] for index in split_ids["fit"]]), grid, settings)
        log.info("merged layer %d from %d shards", layer, args.window_shard_count)


def main(argv=None):
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("moments", "fisher"))
    add_model_arguments(parser)
    add_window_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="capture directory shared by moments, base, and fisher")
    parser.add_argument("--value-factors", type=Path, help="merged core.fit_value bank installed as the teacher (Llama, Qwen3)")
    parser.add_argument("--gated-value-bank", type=Path, help="Qwen3.5 gated-V bank")
    parser.add_argument("--gdn-bank", type=Path, help="Qwen3.5 GDN output-projection bank")
    parser.add_argument("--dense-v", action="store_true", help="dense-Value control: dense teacher and identity Value codes")
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--pinned-pages", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    args.model, args.output = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    assert args.dense_v or args.value_factors is not None or args.gated_value_bank is not None, "pass --value-factors (Qwen3.5: --gated-value-bank and --gdn-bank) or --dense-v"
    assert not args.merge or args.stage == "moments", "Fisher records are written per window and need no merge"
    windows, split_ids = read_windows(args)
    if args.merge:
        merge(args, split_ids)
    else:
        capture(args, windows, split_ids)


if __name__ == "__main__":
    main()
