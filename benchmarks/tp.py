#!/usr/bin/env python3
"""Figure 3: vLLM cohort throughput, TTFT, and TPOT for Llama and Qwen3 (dense, or compact Value with uniform or per-layer ranks)."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

from core.factor_io import sha256, write_json
from core.router_io import read_json
from inference.tp import load_manifest, single_kv_head_layout
from inference import COMPACT_ARCHITECTURES, register
from benchmarks.common import distribution, prompts, arguments, command


def run_batch(llm, prompts, sampling, decode_tokens):
    preemptions_before = sum(metric.value for metric in llm.get_metrics()
                             if metric.name == "vllm:num_preemptions")
    # Admission is outside timing. All requests are queued before scheduling.
    llm.sleep(level=0, mode="keep")
    llm.enqueue(prompts, sampling, use_tqdm=False)
    start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    seconds = time.perf_counter() - start
    preemptions_after = sum(metric.value for metric in llm.get_metrics()
                            if metric.name == "vllm:num_preemptions")
    assert all(len(output.outputs[0].token_ids) == decode_tokens for output in outputs), 'a request stopped early; timings would not cover the full decode'
    requests = []
    for output in outputs:
        metric = output.metrics
        assert metric is not None and not metric.is_corrupted, 'vLLM returned no usable request timings'
        requests.append(dict(
            request_id=output.request_id,
            queued_ts=metric.queued_ts, scheduled_ts=metric.scheduled_ts,
            first_token_ts=metric.first_token_ts, last_token_ts=metric.last_token_ts,
            ttft_ms=1000*(metric.first_token_ts-metric.queued_ts),
            tpot_ms=1000*(metric.last_token_ts-metric.first_token_ts)/(decode_tokens-1),
            output_token_ids=list(output.outputs[0].token_ids),
        ))
    return dict(wall_seconds=seconds, preemptions=preemptions_after-preemptions_before,
                output_tokens_per_second=len(outputs)*decode_tokens/seconds,
                ttft_ms=distribution([row["ttft_ms"] for row in requests]),
                tpot_ms=distribution([row["tpot_ms"] for row in requests]), requests=requests)


def factor_record(root, manifest):
    """What the compact arm serves: the rank schedule, its retention, and fingerprints of the factor files."""
    geometry, ranks = manifest["fit_config"]["geometry"], list(manifest["layer_ranks"])
    head_dim, kv_heads, mean = geometry["head_dim"], geometry["num_kv_heads"], sum(ranks) / len(ranks)
    return dict(
        directory=str(root), schedule="uniform" if len(set(ranks)) == 1 else "adaptive", layer_ranks=ranks, mean_rank=mean,
        value_retention=mean / head_dim, kv_retention=(head_dim + mean) / (2 * head_dim),
        # BF16 bytes one token occupies across all layers and KV heads: every layer keeps a 128-wide Key and its own Value rank.
        logical_cache_bytes_per_token=sum(2 * kv_heads * (head_dim + rank) for rank in ranks),
        dense_cache_bytes_per_token=2 * kv_heads * 2 * head_dim * len(ranks),
        results_sha256=sha256(root / "results.json"),
        layer_sha256={layer: artifact["sha256"] for layer, artifact in manifest["artifacts"].items()},
        allocation={key: manifest["allocation"][key] for key in ("method", "target_average_rank", "candidate_ranks")
                    if key in manifest.get("allocation", {})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "basis"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--value-factors", type=Path)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--factor-validation", choices=("sha256", "structure"), default="sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8,16,32,64,128,256])
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--profile-batches", type=int, nargs="*", default=[])
    args = parser.parse_args()
    assert set(args.profile_batches) <= set(args.batch_sizes), '--profile-batches must be among --batch-sizes'
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.arm}.json"
    assert not result_path.exists(), f"Keep existing results: {result_path}"
    assert (args.value_factors is not None) == (args.arm == "basis"), "basis takes --value-factors; dense does not"
    source_prompts = prompts(args.tokens, max(args.batch_sizes), args.prefill_tokens).tolist()
    # The adapter follows the checkpoint's own configuration; dense keeps the model's native vLLM architecture.
    model_path = args.model.resolve()
    model_type = read_json(model_path / "config.json")["model_type"]
    model = dict(path=str(model_path), model_type=model_type, config_sha256=sha256(model_path / "config.json"),
                 revision=model_path.name if model_path.parent.name == "snapshots" else None)
    ranks, factors, overrides = None, None, {}
    if args.arm == "basis":
        root = args.value_factors.resolve()
        manifest = load_manifest(root)
        fit, ranks, factors = manifest["fit_config"], list(manifest["layer_ranks"]), factor_record(root, manifest)
        local_heads = fit["geometry"]["num_heads"] // args.tp_size
        assert model_type in COMPACT_ARCHITECTURES and fit["geometry"]["model_type"] == model_type, (
            f"compact Value serving covers {sorted(COMPACT_ARCHITECTURES)} checkpoints with factors fitted for the same model; "
            f"got a {model_type} model and {fit['geometry']['model_type']} factors")
        assert single_kv_head_layout(fit["geometry"], args.tp_size), (
            "this benchmark admits prompts in chunks, which the compact path serves with one KV head and 4 or 8 query heads "
            "per rank (TP 8 for Llama-3.1-8B and Qwen3-8B)")
        overrides = {"architectures": [COMPACT_ARCHITECTURES[model_type]], "basiskv_c1_factor_dir": str(root),
                     "basiskv_c1_validation": args.factor_validation}
    import torch
    import transformers
    import triton
    import vllm
    from vllm import LLM, SamplingParams

    register()
    configuration = dict(
        model=str(args.model.resolve()), tensor_parallel_size=args.tp_size, dtype="bfloat16",
        max_model_len=args.prefill_tokens+args.decode_tokens,
        max_num_seqs=max(args.batch_sizes), max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False, disable_log_stats=False, seed=0,
        compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": sorted(set(args.batch_sizes)),
                            "cudagraph_num_of_warmups": 1},
        hf_overrides=overrides,
        worker_extension_cls="inference.vllm.BasisKVWorkerExtension",
    )
    payload = dict(
        schema="basiskv.benchmark.tp.v1", status="initializing", arm=args.arm, created_at=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]), conda_environment=os.environ.get("CONDA_DEFAULT_ENV", sys.prefix),
        torch=torch.__version__, torch_cuda=torch.version.cuda, vllm=vllm.__version__,
        transformers=transformers.__version__, triton=triton.__version__, model_type=model_type,
        model=model, value_factors=factors,
        git_head=command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        git_status=command(["git", "status", "--short"])["stdout"],
        factor_validation=args.factor_validation, configuration=configuration,
        arguments=arguments(args),
        metric_notes={"throughput": "All output tokens divided by cohort wall time, includes prefill.",
                      "ttft": "Per request: first_token_ts minus queued_ts; includes paused admission time.",
                      "tpot": "Per request: (last_token_ts-first_token_ts)/(decode_tokens-1); includes scheduling interleaving.",
                      "profiles": "Separate unmeasured full-cohort runs, rank zero only; summed kernel durations are not wall time."},
        batches=[], profiles=[],
    )

    def save():
        write_json(result_path, payload)

    save()
    llm = LLM(**configuration)
    assert any(metric.name == "vllm:num_preemptions" for metric in llm.get_metrics()), "vLLM exposes no preemption counter; enable metrics"
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True, detokenize=False)
    payload["status"] = "running"
    save()
    compact_calls = [0] * args.tp_size
    for batch in args.batch_sizes:
        cohort = [{"prompt_token_ids": row} for row in source_prompts[:batch]]
        print(f"BENCH arm={args.arm} batch={batch} phase=warmup", flush=True)
        for _ in range(args.warmups):
            run_batch(llm, cohort, sampling, args.decode_tokens)
        row = dict(batch=batch, runs=[])
        payload["batches"].append(row)
        for repeat in range(args.repeats):
            measured = run_batch(llm, cohort, sampling, args.decode_tokens)
            row["runs"].append(measured)
            save()
            print(json.dumps(dict(event="BENCH", arm=args.arm, batch=batch, repeat=repeat,
                                  wall_seconds=measured["wall_seconds"],
                                  preemptions=measured["preemptions"],
                                  output_tokens_per_second=measured["output_tokens_per_second"],
                                  ttft_ms=measured["ttft_ms"], tpot_ms=measured["tpot_ms"])), flush=True)
        row["workers"] = llm.collective_rpc("basiskv_c1_statistics" if args.arm == "basis" else "basiskv_cuda_statistics")
        if args.arm == "basis":
            # Layer by layer, every rank loaded the scheduled factors, holds a cache tensor whose Value width is that
            # layer's rank, and gathered exactly that layer's width; it also executed the compact attention during
            # this batch size (prefill always runs it eagerly).
            assert all(worker["loaded_value_layers"] == len(ranks) and worker["layer_value_ranks"] == ranks
                       and worker["cache_value_widths"] == ranks
                       and {width for width, _ in worker["prepared_shapes"]} == {local_heads * rank for rank in ranks}
                       and worker["compact_attention_calls"] > before
                       for worker, before in zip(row["workers"], compact_calls, strict=True)), "a worker did not run the compact-Value path"
            compact_calls = [worker["compact_attention_calls"] for worker in row["workers"]]
        save()
    # Keep CUPTI and trace export entirely after the timed sweep.
    for batch in args.profile_batches:
        cohort = [{"prompt_token_ids": row} for row in source_prompts[:batch]]
        prefix = output_dir / "profiles" / f"{args.arm}_b{batch}"
        print(f"PROFILE arm={args.arm} batch={batch} prefix={prefix}", flush=True)
        llm.collective_rpc("basiskv_profile_start")
        run_batch(llm, cohort, sampling, args.decode_tokens)
        llm.collective_rpc("basiskv_profile_stop", args=(str(prefix),))
        payload["profiles"].append(str(prefix))
        save()
    payload["status"] = "complete"
    save()
    print(f"COMPLETE {result_path}", flush=True)


if __name__ == "__main__":
    main()
