"""Fresh request state, untimed prefill, and steady decode timing shared by HF runners."""

import time

import torch

from benchmarks.common import distribution


def choose(logits, vocab_size):
    # Reduce per-rank maxima instead of gathering the full vocabulary.
    if hasattr(logits, 'to_local'):
        logits = logits.to_local()
    values, ids = logits.max(dim=-1)
    if logits.shape[-1] != vocab_size:
        dist = torch.distributed
        assert dist.is_initialized() and logits.shape[-1] * dist.get_world_size() == vocab_size, 'logits are neither full nor evenly vocab-sharded'
        ids = ids + dist.get_rank() * logits.shape[-1]
        all_values = [torch.empty_like(values) for _ in range(dist.get_world_size())]
        all_ids = [torch.empty_like(ids) for _ in all_values]
        dist.all_gather(all_values, values)
        dist.all_gather(all_ids, ids)
        winner = torch.stack(all_values).argmax(dim=0, keepdim=True)
        ids = torch.stack(all_ids).gather(0, winner).squeeze(0)
    return ids



@torch.inference_mode()
def steady(model, layers, prompt, warmup, steps, *, profile=False, synchronize_steps=False):
    for layer in layers:
        layer.reset()
        layer.profile_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if profile else None
    prompt = prompt.to(device="cuda")
    positions = [torch.full((prompt.shape[0], 1), prompt.shape[1] + step, device=prompt.device, dtype=torch.long)
                 for step in range(warmup + steps)]
    torch.cuda.synchronize()
    begin = time.perf_counter()
    output = model(input_ids=prompt, use_cache=False, logits_to_keep=1)
    token = choose(output.logits, model.config.vocab_size)
    finite = [torch.isfinite(output.logits).all()]
    del output, prompt
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - begin
    for step in range(warmup):
        output = model(input_ids=token, position_ids=positions[step], use_cache=False, logits_to_keep=1)
        token = choose(output.logits, model.config.vocab_size)
        finite.append(torch.isfinite(output.logits).all())
        del output
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.cuda.reset_peak_memory_stats()
    gpu, wall, attention = [], [], []
    start = time.perf_counter()
    for step, (begin, end) in enumerate(events):
        step_start = time.perf_counter()
        begin.record()
        output = model(input_ids=token, position_ids=positions[warmup + step], use_cache=False, logits_to_keep=1)
        token = choose(output.logits, model.config.vocab_size)
        finite.append(torch.isfinite(output.logits).all())
        end.record()
        del output
        if profile:
            # Separate attribution run; synchronizing here is outside primary timing.
            end.synchronize()
            attention.append(sum(a.elapsed_time(b) for a, b in (layer.profile_events for layer in layers)))
        else:
            # Keep the per-step wall metric from the TP1 protocol, including synchronization.
            if synchronize_steps:
                end.synchronize()
            wall.append((time.perf_counter() - step_start) * 1000)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gpu = [a.elapsed_time(b) for a, b in events]
    assert bool(torch.stack(finite).all()), 'non-finite model logits'
    for layer in layers:
        layer.profile_events = None
    return dict(prefill_seconds=prefill_seconds, decode_seconds=seconds,
        decode_step_ms=gpu, decode_ms=distribution(gpu), step_wall_ms=distribution(wall) if synchronize_steps and wall else None,
        attention_block_ms=distribution(attention) if attention else None,
        decode_peak_allocated_bytes=peak_allocated, decode_peak_reserved_bytes=peak_reserved)
