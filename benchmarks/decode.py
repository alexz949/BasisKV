"""Table 3: GPU-resident TP1/B1 steady attention-block and full-model decode latency."""

import argparse
import statistics

from benchmarks.common import add_model_arguments, arguments, load_model, metadata, prompts, save
from kernels.routing_config import PAGE_SIZE, PINNED_PAGES, ROUTED_TOKENS, RECENT_TOKENS, SUPPORT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--contexts', type=int, nargs='+', default=[24576, 32768, 49152, 65536, 98304, 131072])
    parser.add_argument('--sample-row', type=int, default=0)
    parser.add_argument('--profile-steps', type=int, default=32)
    args = parser.parse_args()
    assert min(args.contexts) > SUPPORT and not args.output.exists(), 'contexts must exceed the support budget, and the output must be new'
    assert (args.router_dir is not None) == (args.arm == 'basis') and (args.arm != 'dense' or args.value_factors is None), 'basis takes --router-dir; dense takes neither factors nor router'
    import torch
    from benchmarks.attention import install
    from benchmarks.timing import steady

    torch.set_num_threads(2)
    torch.manual_seed(20260924)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.model, config_path=args.config)
    capacity = max(args.contexts) + args.warmup_steps + max(args.decode_steps, args.profile_steps)
    layers, _ = install(model, args, batch=1, capacity=capacity)
    result = dict(schema='basiskv.benchmark.decode.v1', status='running', arguments=arguments(args), environment=metadata(), contexts=[],
        protocol=dict(tp=1, batch=1, key_placement='gpu', value_placement='gpu', page_size=PAGE_SIZE,
            pinned_pages=PINNED_PAGES, routed_tokens=ROUTED_TOKENS, recent_tokens=RECENT_TOKENS,
            support_tokens=SUPPORT, timing='Prefill excluded; CUDA-event full-model decode. Attention includes append, routing, and attention, excludes QKV/O projection. Separate attribution run.'))
    for length in args.contexts:
        prompt = prompts(args.tokens, 1, length, args.sample_row)
        print(f'context={length} phase=warmup', flush=True)
        steady(model, layers, prompt, args.warmup_steps, args.decode_steps, synchronize_steps=True)
        trials = []
        for repeat in range(args.repeats):
            measured = steady(model, layers, prompt, args.warmup_steps, args.decode_steps, synchronize_steps=True)
            profile = steady(model, layers, prompt, args.warmup_steps, args.profile_steps, profile=True)
            measured['attention_block_ms'] = profile['attention_block_ms']
            trials.append(measured)
            print(f'context={length} repeat={repeat} model_ms={measured["decode_ms"]["median"]:.4f} attention_ms={profile["attention_block_ms"]["median"]:.4f}', flush=True)
        result['contexts'].append(dict(context_tokens=length, trials=trials,
            model_median_ms=statistics.median(row['decode_ms']['median'] for row in trials),
            attention_median_ms=statistics.median(row['attention_block_ms']['median'] for row in trials)))
    result['status'] = 'complete'
    save(args.output, result)


if __name__ == '__main__':
    main()
