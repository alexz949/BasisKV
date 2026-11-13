"""Figure 4: TP steady decode throughput versus peak per-rank decode allocation."""

import argparse
import os
from datetime import timedelta

from benchmarks.common import add_model_arguments, arguments, distribution, load_model, metadata, prompts, save
from basiskv_kernels.routing_config import PINNED_PAGES, ROUTED_TOKENS, RECENT_TOKENS, SUPPORT


def summarize(ranks, batch, steps):
    times = [max(row['decode_step_ms'][step] for row in ranks) for step in range(steps)]
    seconds = max(row['decode_seconds'] for row in ranks)
    return dict(decode_ms=distribution(times), decode_seconds=seconds,
        decode_tokens_per_second=batch * steps / seconds,
        peak_per_rank_decode_allocated_bytes=max(row['decode_peak_allocated_bytes'] for row in ranks),
        peak_per_rank_decode_reserved_bytes=max(row['decode_peak_reserved_bytes'] for row in ranks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--numa-node', type=int, help='Optional explicit host allocation NUMA binding; no assumed machine topology')
    args = parser.parse_args()
    assert args.context > SUPPORT and not args.output.exists(), 'the context must exceed the support budget, and the output must be new'
    assert (args.router_dir is not None) == (args.arm == 'basis') and (args.arm != 'dense' or args.value_factors is None), 'basis takes --router-dir; dense takes neither factors nor router'
    import torch
    import torch.distributed as dist
    from benchmarks.attention import install
    from benchmarks.timing import steady

    rank, world, local_rank = (int(os.environ.get(name, default)) for name, default in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    torch.cuda.set_device(local_rank)
    if args.numa_node is not None:
        from benchmarks.numa_memory import bind_host_allocations
        bind_host_allocations(args.numa_node)
    torch.set_num_threads(2)
    torch.manual_seed(20260924)
    torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    model = load_model(args.model, world, args.config)
    capacity = args.context + args.warmup_steps + args.decode_steps
    layers, communicator = install(model, args, batch=args.batch_size, capacity=capacity,
        rank=rank, world=world, host_keys=args.arm == 'basis')
    prompt = prompts(args.tokens, args.batch_size, args.context)
    result = dict(schema='basiskv.benchmark.memory.v1', status='running', arguments=arguments(args), environment=metadata(), trials=[],
        protocol=dict(tp=world, page_size=args.page_size, pinned_pages=PINNED_PAGES,
            routed_tokens=ROUTED_TOKENS, recent_tokens=RECENT_TOKENS, support_tokens=SUPPORT,
            key_placement='pinned_host_with_gpu_slots' if args.arm == 'basis' else 'gpu',
            memory='Peak PyTorch allocated bytes during steady decode, maximum across ranks; excludes prefill peaks. Compact Value factors store Base inside V; residual codes and selected-Key slots are included.',
            throughput='batch * measured steps / maximum rank wall time; prefill and conditioning excluded.'))
    print(f'rank={rank} phase=warmup', flush=True)
    steady(model, layers, prompt, args.warmup_steps, args.decode_steps)
    for repeat in range(args.repeats):
        row = steady(model, layers, prompt, args.warmup_steps, args.decode_steps)
        row.update(rank=rank, step_count=args.decode_steps)
        rows = [None] * world
        if world > 1:
            dist.all_gather_object(rows, row)
        else:
            rows[0] = row
        trial = dict(repeat=repeat, ranks=rows, **summarize(rows, args.batch_size, args.decode_steps))
        result['trials'].append(trial)
        print(f'rank={rank} repeat={repeat} tokens_per_second={trial["decode_tokens_per_second"]:.3f} peak_gib={trial["peak_per_rank_decode_allocated_bytes"] / 2**30:.3f}', flush=True)
    if communicator is not None:
        communicator.close()
    if rank == 0:
        result['summary'] = dict(decode_tokens_per_second=distribution(trial['decode_tokens_per_second'] for trial in result['trials']),
            peak_per_rank_decode_allocated_bytes=max(trial['peak_per_rank_decode_allocated_bytes'] for trial in result['trials']))
        result['status'] = 'complete'
        save(args.output, result)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
