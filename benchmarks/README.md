# Performance benchmarks

Three public entrypoints share the current Value/router loaders and timing helpers.
Run from the `BasisKV/` root. Model weights, factors, token inputs, and results are
external inputs/outputs; no old `results/` source snapshot is required.

| Entrypoint | Paper measurement | Scope |
| --- | --- | --- |
| `tp.py` | Figure 3: cohort throughput, TTFT, TPOT | vLLM; compact Qwen3 V64/V96, configurable TP |
| `decode.py` | Table 3: steady attention-block and model decode latency | HF Llama/Qwen3, TP1/B1, GPU-resident K/V |
| `memory.py` | Figure 4: decode throughput versus peak per-rank allocation | HF Llama/Qwen3, torchrun TP, host exact K with GPU selected-Key slots for BasisKV |

The compact vLLM adapter currently supports Qwen3. This entrypoint does not
provide the Llama compact-vLLM arm of Figure 3. Dense vLLM uses the model's
native architecture. The vLLM path requires the paper runtime's DiffKV,
admission barrier, and timestamp interfaces; an arbitrary vLLM installation
is not sufficient. The HF TP loader also requires a Torch build compatible with
Transformers 5.17's distributed API; Torch 2.6 lacks its `src_data_rank`
argument. A TP1 smoke check does not validate that loader.

## Inputs

- `--model`: local Hugging Face model directory.
- `--tokens`: safetensors containing integer `input_ids` with shape
  `[samples, tokens]`. Rows are distinct requests; no hidden repetition or
  synthetic text generation. Supply at least the maximum requested batch size.
- `--value-factors`: optional current `core.fit_value` output (`results.json`
  plus layer files). Required for compact vLLM. HF BasisKV can omit it for the
  dense-V routing control used in Table 3.
- `--router-dir`: required for HF BasisKV. Current layer safetensors and JSON
  records must follow the **Page32/B16R16** system protocol; only B16R16 is checked.
  No pinned prefix; exact recent-64 tail. An incompatible Page4 quality bank
  must be fitted separately for Page32 before using this CUDA benchmark.

The existing system kernel selects **62 × 32 = 1,984 routed tokens + 64 recent
= 2,048 total support**. Main quality evaluation instead uses Page4 and its
own routed-token budget. These protocols are recorded separately.

HF attention derives layer/head counts from the model configuration and supports
QK128 and V64/V96/V128. Routed execution requires four query heads per KV head;
TP evenly partitions KV groups. Qwen3 Q/K normalization is retained. Compact
routed V is changed into Base-aligned coordinates during loading, so the first
16 Value coordinates also hold Base. The decoder absorbs the inverse change.
No separate checkpoint preparation or Base cache is needed for compact V.

## Commands

Use `lowrank` or `basis` with compatible Transformers/Torch/CUDA and, for
`tp.py`, the required vLLM build. Run full benchmarks through Slurm with the
requested GPUs; do not run them on the login node. The TP8 paper measurements
use eight L40S GPUs. Match hardware, input cohorts, and factors when comparing
with the published numbers. These commands specify measurement protocols,
not a guarantee of identical numbers after code consolidation.

```bash
# Figure 3: one model/rank and all requested batch sizes.
python -m benchmarks.tp --arm basis --model MODEL \
  --value-factors FACTORS --tokens TOKENS.safetensors --tp-size 8 \
  --batch-sizes 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 \
  --output-dir OUTPUT

# Table 3: all K/V resident, dense V, three trials per context.
python -m benchmarks.decode --arm basis --model MODEL \
  --router-dir PAGE32_V128_ROUTER --tokens TOKENS.safetensors \
  --contexts 24576 32768 49152 65536 98304 131072 --output OUTPUT.json

# Figure 4: one point of the context/batch sweep per invocation.
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m benchmarks.memory --arm basis --model MODEL \
  --value-factors FACTORS --router-dir PAGE32_ROUTER \
  --tokens TOKENS.safetensors --context 65536 --batch-size 8 --output OUTPUT.json
```

For matched dense controls, use `--arm dense` and omit factor/router arguments.
Change `--context` and `--batch-size` for memory-curve points; the runner does
not assume model names, machine paths, NUMA topology, or a batch grid. Optional
`--numa-node` binds future host allocations explicitly on the local process.

## Timing and memory

`tp.py` queues a complete cohort while scheduling is paused, then times wakeup
through completion. Throughput includes prefill. TTFT is first-token timestamp
minus queued timestamp, including paused admission; TPOT is last minus first
divided by generated tokens minus one. It records preemptions and per-request
metrics. Optional `--profile-batches` runs attribution after the timed sweep.

HF runners reset cache lengths and persistent-Key mappings for each request.
Prefill and conditioning are excluded from steady timing. `decode.py` performs
one same-shape warmup request, then reports the median of three trial medians.
Attention-block attribution runs separately and includes cache append, routing,
and attention; QKV and output projection are excluded. Full-model timing also
includes greedy token selection, with no early EOS termination.

`memory.py` reports batch × measured decode steps divided by the slowest rank's
wall time. Per-step CUDA times use the maximum across ranks. Peak decode
memory is the maximum per-rank PyTorch allocation after prefill and conditioning;
reserved memory is reported separately. Host K bytes are not GPU allocation.
No profiler runs during its primary timing.

`attention.py` is the shared HF runtime; `timing.py`, `common.py`, and
`numa_memory.py` provide the remaining helpers. Its CUDA extensions come from
`kernels.decode`.
Historical per-model drivers, microbenchmarks, two-stage routing experiments,
Ampere preparation, and frozen-source baseline launchers live in `others/`.
