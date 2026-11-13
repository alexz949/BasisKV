# Performance benchmarks

Three public entrypoints share the current Value/router loaders and timing helpers.
Run from the `BasisKV/` root. Model weights, factors, token inputs, and results are
external inputs/outputs.

| Entrypoint | Paper measurement | Scope |
| --- | --- | --- |
| `tp.py` | Figure 3: cohort throughput, TTFT, TPOT | vLLM; Llama-3.1-8B-Instruct and Qwen3, dense or compact Value with uniform or per-layer ranks, TP 8 |
| `decode.py` | Table 3: steady attention-block and model decode latency | HF Llama/Qwen3, TP1/B1, GPU-resident K/V |
| `memory.py` | Figure 4: decode throughput versus peak per-rank allocation | HF Llama/Qwen3, torchrun TP, host exact K with GPU selected-Key slots for BasisKV |

`tp.py` serves Llama and Qwen3 through one entrypoint. It reads `model_type`
from the checkpoint's `config.json` and selects the matching compact adapter
(`inference.vllm.BasisKVLlamaForCausalLM` or `BasisKVQwen3ForCausalLM`); the
dense arm uses the model's native vLLM architecture. The two adapters share the
factor loading, the Value encoder folded into the V projection, the per-rank
compact cache, and the gathered decoder GEMM. They differ where the models do:
Qwen3 normalizes Q and K per head before RoPE, Llama does not, and each keeps
its native rotary module and weight loading.

Scope of the compact arm:

- Models: Llama-3.1-8B-Instruct and Qwen3 with `head_dim` 128, BF16, and
  factors fitted for the same model geometry. Attention projections must be
  bias-free.
- Ranks: a uniform bank from `core.fit_value` (one rank for every layer) or a
  per-layer bank from `core.allocate` (one rank per layer, shared by that
  layer's KV heads). Any rank from 1 to 128 is served, including the dense
  endpoint 128; the schedule is read from the factor directory. Layer `l`
  runs at its own rank `r_l` throughout: the V projection writes `r_l`
  coordinates per KV head, the cache stores them, attention aggregates them,
  each rank gathers `local query heads x r_l` coordinates, and the decoder
  GEMM takes that width. Nothing is padded to the largest rank in the cache
  or on the wire. Inside the Triton kernels a Value tile is 32, 64, or 128
  wide and the unused columns are masked; that padding exists only during the
  computation. Kernel tile shapes were tuned at ranks 64 and 96 and are reused
  for other ranks by tile width.
- Tensor parallelism: layouts with one KV head and 4 or 8 query heads per rank,
  that is TP 8 for Llama-3.1-8B and Qwen3-8B. These use the DiffKV kernels in
  `kernels/`, which read the paged cache and therefore serve the chunked
  prefill this benchmark enables. The entrypoint refuses other TP sizes for the
  compact arm: their paged backend admits only whole prompts.
- The cache is compact layer by layer: layer `l`'s cache entry is `128 + r_l`
  wide on every rank. vLLM keeps the layers in one cache group and sizes each
  layer's tensor by its own page, and the benchmark checks the widths of the
  tensors it actually allocated.

The vLLM path needs the environment of `requirements-tp.txt` (upstream vLLM
0.30.0+cu129, unmodified); see the [installation](../README.md#installation).
The HF TP loader of `memory.py` requires the same Torch 2.13.0 build
(Transformers 5.17's distributed API; Torch 2.6 lacks its `src_data_rank`
argument).

## CUDA setup

Two things compile on first use on a GPU node: the CUDA extensions in
`kernels/` (NCCL AllGather, paged attention, router), and, when vLLM starts,
the sampling kernels of its FlashInfer dependency. Both need `nvcc` and the
CUDA headers (including `curand.h`) of the same CUDA release as the Torch
wheel. The CUDA runtime and NCCL (headers and `libnccl.so.2`, 2.29.7) come from
the wheels Torch already installed. Without a system CUDA 12.9 toolkit,
install the compiler and headers into a directory of your choice:

```bash
bash kernels/install_cuda_toolkit.sh /path/to/cuda-12.9   # nvcc 12.9.86, cudart, cccl, curand; checksums verified
export CUDA_HOME=/path/to/cuda-12.9
```

A host C++17 compiler is required, and `nvcc` uses the one named by `CC`
(default: `gcc` on the `PATH`). On machines whose default `gcc` cannot compile
C++, for example when `gcc` and `g++` are different versions, vLLM stops at
startup with `gcc: fatal error: cannot execute 'cc1plus'`; name a working C++
compiler explicitly:

```bash
export CC=$(command -v g++) CXX=$(command -v g++)
```

Set `BASISSERVE_EXT_BUILD_DIR` to keep the built extensions between runs.

## Verification

Run from the repository root after installing `requirements-tp.txt`:

```bash
python -m tests.test_compact_tp
```

Verification status of the Llama/Qwen3 `tp.py` path in this repository state:

- Verified on CPU (`python -m tests.test_compact_tp`, clean environment): both
  adapters with uniform V64 and V96 and with a per-layer bank of ranks
  `[32, 96, 128, 64]` assembled by `core.factor_io.assemble`, against the HF
  reference with the same factors, teacher-forced over one prompt and four
  decode steps (relative logit error below 2e-6 in FP32, identical top-1 at
  every position). The per-layer bank runs at TP 8, 4, 2, and 1; every TP
  rank is built. TP8 uses a two-chunk prefill, while TP1/2/4 use a
  whole-prompt prefill. This covers factor loading and TP head ownership, the
  folded V projection, Q/K norms, RoPE, weight loading, each layer's
  collective width, and the decoder. The test also asks vLLM's cache planner
  for the layout: every layer's page is `128 + r_l` wide and the pool per
  block is exactly their sum.
- Verified on GPUs (2 x RTX A5000, TP 2, the environment above installed from
  scratch): `python -m benchmarks.tp` end to end for synthetic Llama and Qwen3
  checkpoints with the served layout (`head_dim` 128, 8 query heads, 2 KV
  heads, so one KV head and 4 query heads per rank) and four banks each:
  uniform V64, uniform V96, and per-layer `[32, 96, 128, 64]` and
  `[48, 80, 112, 96]`. The prompts were longer than the per-step token budget,
  so every request went through chunked prefill; decode ran under CUDA Graph
  capture. Each rank reported the DiffKV implementation, the NCCL AllGather
  output, cache tensors `128 + r_l` wide for every layer, and AllGather plans
  of exactly `4 x r_l` local coordinates per layer; the allocated cache bytes
  per token equaled the logical figure (3,328 and 3,392 against 4,096 dense
  for the two per-layer banks). Fed back to the HF reference with the same
  factors, the served token was the reference top-1 at 372 of 384 decode
  steps; the other 12 were its second or third choice, within 0.06 logit
  standard deviations, the level the dense arm shows on the same check. The
  dense arm also completed for the real Llama-3.1-8B-Instruct and Qwen3-8B at
  TP 2, matching the HF reference at 16 of 16 decode steps each.

## Inputs

- `--model`: local Hugging Face model directory (Llama-3.1-8B-Instruct or
  Qwen3). Its `config.json` selects the adapter.
- `--tokens`: safetensors containing integer `input_ids` with shape
  `[samples, tokens]`. Each row supplies one request. Supply at least the maximum
  requested batch size.
  `calibration.windows` writes this layout (`windows.safetensors`) from C4 with
  the model's tokenizer; pass `--sequence-length` of at least `--prefill-tokens`.
- `--value-factors`: a Value-factor directory (`results.json` plus layer
  files) fitted for the served model, prepared as in the root README's
  calibration and fitting steps. Required for compact vLLM, which accepts
  both kinds and recognizes them from `results.json`: a uniform bank written
  by `core.fit_value ... --rank R` followed by `core.fit_value merge`, or the
  directory `core.allocate --banks ... --target-average-rank R --output DIR`
  assembles from several uniform banks. No rank argument is passed to the
  benchmark. The HF runners (`decode.py`, `memory.py`) take uniform banks;
  HF BasisKV can omit the factors for the dense-V routing control used in
  Table 3.
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
The loader applies this coordinate change to the fitted Value factors.

## Commands

Use the environment from the root README; `tp.py` needs its vLLM layer and
`CUDA_HOME`. Run full benchmarks through Slurm with the
requested GPUs; do not run them on the login node. The TP8 paper measurements
use eight L40S GPUs. Match hardware, input cohorts, and factors when comparing
with the published numbers.

```bash
# Figure 3: one model and all requested batch sizes. MODEL is a Llama-3.1-8B-Instruct or Qwen3
# directory; the same command serves both. UNIFORM_BANK is a merged core.fit_value output.
python -m benchmarks.tp --arm basis --model MODEL \
  --value-factors UNIFORM_BANK --tokens TOKENS.safetensors --tp-size 8 \
  --batch-sizes 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 \
  --output-dir OUTPUT_UNIFORM
# Per-layer ranks: the same command with the directory core.allocate wrote.
python -m benchmarks.tp --arm basis --model MODEL \
  --value-factors ALLOCATED_BANK --tokens TOKENS.safetensors --tp-size 8 \
  --batch-sizes 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 \
  --output-dir OUTPUT_ALLOCATED
# Matched dense control: the model's native vLLM architecture, same prompts and decode length.
python -m benchmarks.tp --arm dense --model MODEL \
  --tokens TOKENS.safetensors --tp-size 8 \
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
Set `--context` and `--batch-size` for each memory-curve point. Optional
`--numa-node` binds future host allocations explicitly on the local process.

## Timing and memory

`tp.py` queues a complete cohort while scheduling is paused, then times wakeup
through completion. Throughput includes prefill. TTFT is first-token timestamp
minus queued timestamp, including paused admission; TPOT is last minus first
divided by generated tokens minus one. Decoding is greedy with a fixed output
length; warmup cohorts precede the timed repeats. It records preemptions,
per-request metrics, the command, and the Torch, vLLM, Transformers, and Triton
versions. Optional `--profile-batches` runs attribution after the timed sweep.

For the compact arm, `basis.json` records what was served under
`value_factors`: `schedule` (`uniform` or `adaptive`), the full `layer_ranks`,
`mean_rank`, `value_retention` (mean rank / 128), `kv_retention`,
`logical_cache_bytes_per_token` (BF16 bytes per token summed over layers and
KV heads at each layer's own rank) next to `dense_cache_bytes_per_token`, the
SHA-256 of `results.json` and of every layer file, and the allocation method.
`model` records the model path, its `config.json` SHA-256, and the snapshot
revision when the path is a Hugging Face cache snapshot.

After each batch size every rank reports, per layer, what it runs
(`batches[i].workers[rank]`): `layer_value_ranks`, `cache_value_widths` and
`kv_cache_shapes` read from the cache tensors vLLM allocated,
`kv_cache_bytes` (the allocator's total for that rank),
`collective_local_widths`, and `prepared_shapes`, the `(local width, tokens)`
pairs the AllGather plans were built for. To confirm a run, compare
`cache_value_widths` with `layer_ranks` and the widths in `prepared_shapes`
with `local query heads x r_l`; the benchmark itself stops if, on any rank,
either differs from the schedule, a layer's factors are missing, or the
compact attention did not execute during that batch size. AllGather plans
are prepared once per width and token count before CUDA Graph capture and
reused afterwards.

A run with a per-layer bank measures that bank only. Throughput measured
with a uniform bank does not carry over to a different schedule; pair any
quality number with a throughput number taken from the same factor directory.

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
