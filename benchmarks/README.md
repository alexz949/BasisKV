# Performance benchmarks

Three public entrypoints share the current Value/router loaders and timing helpers.
Run from the `BasisKV/` root. Model weights, factors, token inputs, and results are
external inputs/outputs; no old `results/` source snapshot is required.

| Entrypoint | Paper measurement | Scope |
| --- | --- | --- |
| `tp.py` | Figure 3: cohort throughput, TTFT, TPOT | vLLM; Llama-3.1-8B-Instruct and Qwen3, dense or compact V64/V96, TP 8 |
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

- Models: Llama-3.1-8B-Instruct and Qwen3 with `head_dim` 128, BF16, uniform
  V64 or V96 factors from `core.fit_value` fitted for the same model geometry.
  Attention projections must be bias-free.
- Tensor parallelism: layouts with one KV head and 4 or 8 query heads per rank,
  that is TP 8 for Llama-3.1-8B and Qwen3-8B. These use the DiffKV kernels in
  `kernels/`, which read the paged cache and therefore serve the chunked
  prefill this benchmark enables. The entrypoint refuses other TP sizes for the
  compact arm: their paged backend admits only whole prompts.
- The cache is compact: each rank's KV cache entry is `128 + rank` wide, and the
  benchmark checks the width of the cache tensors vLLM actually allocated.

The vLLM path needs the environment of `requirements-tp.txt` (upstream vLLM
0.30.0+cu129, unmodified); see the [installation](../README.md#installation).
The HF TP loader of `memory.py` requires the same Torch 2.13.0 build
(Transformers 5.17's distributed API; Torch 2.6 lacks its `src_data_rank`
argument). A TP1 smoke check does not validate that loader.

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
  adapters with V64 at TP8 and V96 at TP1/2/4/8, against the HF reference with
  the same factors (relative logit error below 2e-6 in FP32). Each case covers
  every TP rank and decode steps; TP8 uses a two-chunk prefill, while TP1/2/4
  use a whole-prompt prefill. This covers factor loading and TP head ownership,
  the folded V projection, Q/K norms, RoPE, weight loading, and the decoder.
- Verified on GPUs (2 x RTX A5000, TP 2, the environment above installed from
  scratch, a repository copy without `others/`): `python -m benchmarks.tp`
  end to end for synthetic Llama and Qwen3 checkpoints with the served layout
  (`head_dim` 128, 8 query heads, 2 KV heads, so one KV head and 4 query heads
  per rank), dense, V64, and V96. The prompts were longer than the per-step
  token budget, so every request went through chunked prefill; decode ran
  under CUDA Graph capture. Each rank reported the DiffKV implementation, the
  NCCL AllGather output, all layers' factors, and allocated cache entries 192
  and 224 wide (128 + rank). Fed back to the HF reference with the same
  factors, the served token was the reference argmax at 188 of 192 compact
  decode steps; the other 4 were its second choice, within 0.03 logit standard
  deviations (dense: 93 of 96 on the same check). The dense arm also completed
  for the real Llama-3.1-8B-Instruct and Qwen3-8B at TP 2, matching the HF
  reference at 16 of 16 decode steps each.
- Not yet run: the compact arm with real 8B weights. It needs TP 8 (eight GPUs
  on one node); only its layout, kernels, and transport were exercised at TP 2.
  These checks used tiny cohorts and establish correctness, not performance:
  no throughput number here is validated, and the entrypoint does not by itself
  reproduce the paper's speedups.

## Inputs

- `--model`: local Hugging Face model directory (Llama-3.1-8B-Instruct or
  Qwen3). Its `config.json` decides the adapter; the path name is not inspected.
- `--tokens`: safetensors containing integer `input_ids` with shape
  `[samples, tokens]`. Rows are distinct requests; no hidden repetition or
  synthetic text generation. Supply at least the maximum requested batch size.
  `calibration.windows` writes this layout (`windows.safetensors`) from C4 with
  the model's tokenizer; pass `--sequence-length` of at least `--prefill-tokens`.
- `--value-factors`: optional current `core.fit_value` output (`results.json`
  plus layer files), prepared as in the root README's calibration and fitting
  steps (`calibration.windows`, `calibration.value`, `core.fit_value` with
  `--rank 64` or `--rank 96`). Required for compact vLLM, where it must be a
  uniform-rank bank fitted for the served model. HF BasisKV can omit it for
  the dense-V routing control used in Table 3.
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

Use the environment from the root README; `tp.py` needs its vLLM layer and
`CUDA_HOME`. Run full benchmarks through Slurm with the
requested GPUs; do not run them on the login node. The TP8 paper measurements
use eight L40S GPUs. Match hardware, input cohorts, and factors when comparing
with the published numbers. These commands specify measurement protocols,
not a guarantee of identical numbers after code consolidation.

```bash
# Figure 3: one model/rank and all requested batch sizes. MODEL is a Llama-3.1-8B-Instruct
# or Qwen3 directory; the same command serves both, and FACTORS must match MODEL.
python -m benchmarks.tp --arm basis --model MODEL \
  --value-factors FACTORS --tokens TOKENS.safetensors --tp-size 8 \
  --batch-sizes 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 \
  --output-dir OUTPUT
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
Change `--context` and `--batch-size` for memory-curve points; the runner does
not assume model names, machine paths, NUMA topology, or a batch grid. Optional
`--numa-node` binds future host allocations explicitly on the local process.

## Timing and memory

`tp.py` queues a complete cohort while scheduling is paused, then times wakeup
through completion. Throughput includes prefill. TTFT is first-token timestamp
minus queued timestamp, including paused admission; TPOT is last minus first
divided by generated tokens minus one. Decoding is greedy with a fixed output
length; warmup cohorts precede the timed repeats. It records preemptions,
per-request metrics, the command, and the Torch, vLLM, Transformers, and Triton
versions. After each batch size every rank reports its implementation; the
compact arm requires that each rank loaded all layers' factors, holds cache
tensors whose Value width equals the fitted rank, and executed the compact
attention during that batch size. Optional `--profile-batches` runs attribution
after the timed sweep.

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
