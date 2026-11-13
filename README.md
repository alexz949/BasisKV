# BasisKV

BasisKV combines low-rank Value compression with Page-Fisher Key routing for
efficient LLM inference. This repository provides calibration, joint V/O fitting,
routing, inference kernels, and quality and performance evaluation code.

## Installation

One environment is supported: Python 3.11, PyTorch 2.13.0 built for CUDA 12.9,
Transformers 5.17.0, Triton 3.7.1. Calibration, fitting, Hugging Face inference,
and evaluation need only the base layer:

Download and extract the source ZIP from the anonymous repository. Run the
following commands in the extracted directory containing `pyproject.toml`:

```bash
conda create -y -n basiskv -c conda-forge python=3.11
conda activate basiskv
pip install uv
uv pip install torch==2.13.0 --torch-backend=cu129   # the CUDA 12.9 wheel; installed first so nothing replaces it
uv pip install -r requirements.txt
pip install -e . --no-deps
```

The tensor-parallel serving benchmark (`benchmarks.tp`) adds vLLM. It runs on the
unmodified upstream vLLM 0.30.0 release wheel for CUDA 12.9, which provides the
independent Key/Value head widths (DiffKV), the paused-admission scheduling
calls, and the request timestamps the benchmark uses:

```bash
uv pip install -r requirements-tp.txt --torch-backend=cu129
```

`requirements-tp.txt` keeps the Torch build of the base layer (vLLM pins
`torch==2.13.0`). Run the commands inside the activated environment.

CUDA kernels require a CUDA 12.9 toolkit and a host C++17 compiler. See
[CUDA setup and troubleshooting](benchmarks/README.md#cuda-setup) for the toolkit
installer, and [CPU verification](benchmarks/README.md#verification) for the
compact-Value adapter test (requires the vLLM dependencies above).

## Quick Start

Run commands from the repository root. Replace `MODEL` with a local Hugging Face
Llama or Qwen3 model directory. Model weights and datasets are supplied separately.

### 1. Calibrate and fit Value factors

Prepare C4 calibration windows, collect activation statistics, and fit joint V/O
factors. This example retains 96 Value dimensions per KV head.

```bash
python -m calibration.windows \
  --model MODEL --output outputs/windows --sequence-length 2048 \
  --fit-windows 256 --heldout-windows 64
python -m calibration.value \
  --model MODEL --windows outputs/windows/windows.safetensors \
  --output outputs/statistics
python -m core.fit_value fit-shard \
  --snapshot-dir outputs/statistics --output-dir outputs/value --rank 96
python -m core.fit_value merge --output-dir outputs/value
```

#### Adaptive Value allocation: 50% and 75% retention

Adaptive allocation assigns a rank to each layer under a global budget.
For Llama and Qwen3 models with 128-dimensional KV heads, average ranks 64 and
96 retain 50% and 75% of the Value dimensions, respectively. Each layer gets
one rank shared by its physical KV heads; Keys remain dense.

The following Bash commands run the allocation workflow at both budgets. Use
a separate output directory for each model. The calibration uses 256 C4 fit
windows and 64 held-out windows, each 2,048 tokens long. Candidate factors
share these statistics and use six ALS sweeps, 16 CG iterations, FP32 fitting,
and BF16 storage. Rank 128 is the exact dense endpoint.

```bash
python -m calibration.windows \
  --model MODEL --output outputs/adaptive/windows --sequence-length 2048 \
  --fit-windows 256 --heldout-windows 64 --seed 20260821
python -m calibration.value \
  --model MODEL --windows outputs/adaptive/windows/windows.safetensors \
  --output outputs/adaptive/statistics

banks=()
for rank in 32 48 64 80 96 112 128; do
  python -m core.fit_value fit-shard \
    --snapshot-dir outputs/adaptive/statistics \
    --output-dir "outputs/adaptive/banks/R${rank}" --rank "$rank" \
    --encoder-sweeps 6 --cg-iterations 16 --covariance-damping 1e-7 \
    --work-dtype float32 --factor-dtype bfloat16
  python -m core.fit_value merge --output-dir "outputs/adaptive/banks/R${rank}"
  banks+=("${rank}=outputs/adaptive/banks/R${rank}")
done
```

Next, prepare a separate C4 profiling bank and measure output KL against the
dense model. Both budgets use the same uniform rank-64 anchor: each probe
changes just one layer to rank 32 or 96 while keeping the other layers at 64.

```bash
python -m calibration.windows \
  --model MODEL --output outputs/adaptive/profile --sequence-length 2048 \
  --fit-windows 128 --heldout-windows 0 --seed 20260828
python -m calibration.probe_kl \
  --model MODEL --windows outputs/adaptive/profile/windows.safetensors \
  --banks 32=outputs/adaptive/banks/R32 64=outputs/adaptive/banks/R64 \
          96=outputs/adaptive/banks/R96 \
  --anchor-rank 64 --low-rank 32 --high-rank 96 \
  --window-count 128 --sequence-length 2048 --batch-size 1 \
  --output outputs/adaptive/probes

# Run in the same Bash session as the bank-fitting loop above.
for target in 64 96; do
  python -m core.allocate --banks "${banks[@]}" \
    --probes outputs/adaptive/probes/probes.json \
    --target-average-rank "$target" --output "outputs/adaptive/V${target}"
done
```

`core.allocate` selects ranks by exact-budget dynamic programming and assembles
the chosen layer factors into `outputs/adaptive/V64` and `outputs/adaptive/V96`.
Their `results.json` files record the layer ranks and allocation settings.
Evaluate these assembled directories for the adaptive quality results:

```bash
python -m calibration.windows \
  --model MODEL --output outputs/adaptive/c4_validation --split validation \
  --sequence-length 2048 --fit-windows 128 --heldout-windows 0 --seed 20260828
for target in 64 96; do
  python -m eval.ppl --model MODEL --value-factors "outputs/adaptive/V${target}" \
    --dataset both --c4-windows outputs/adaptive/c4_validation/windows.safetensors \
    --sequence-length 2048 --batch-size 2 \
    --output "outputs/adaptive/eval/V${target}/ppl"
  python -m eval.mcq --model MODEL --value-factors "outputs/adaptive/V${target}" \
    --output "outputs/adaptive/eval/V${target}/mcq"
done
```

The same assembled directories can be passed to `benchmarks.tp --value-factors`
to measure each adaptive configuration; see the
[TP benchmark commands](benchmarks/README.md#commands). The router example
below continues to use the uniform V96 factors from the initial Quick Start.

### 2. Fit the Page-Fisher router

Prepare a routing calibration bank with **16 C4 + 16 synthetic fit windows
and 16 held-out C4 windows**, each 131,072 tokens long. The commands below
generate the C4 and synthetic data, then assemble the mixed bank:

```bash
python -m calibration.windows \
  --model MODEL --output outputs/routing_c4 --sequence-length 131072 \
  --fit-windows 32 --heldout-windows 16 --seed 20260921
python -m calibration.retrieval \
  --model MODEL --exclude-manifest outputs/routing_c4/manifest.json \
  --seed 20260922 --haystack-seed 20260923 --output outputs/routing_retrieval
python -m calibration.mix_windows \
  --c4 outputs/routing_c4 --synthetic outputs/routing_retrieval \
  --output outputs/routing_windows
```

The synthetic data cover multikey retrieval, multivalue retrieval, variable
tracking, and aggregation. The assembled bank contains `windows.safetensors`
and a `manifest.json` identifying the fit and held-out windows.

Use the merged Value factors and `outputs/routing_windows/windows.safetensors`
for both capture stages below. For Qwen3-8B, create the [static YaRN configuration](eval/README.md#prepare-long-context-inputs).
The model must support 131,072 tokens; pass
`--config` to both capture commands when using an extended-context configuration.

```bash
python -m calibration.router moments \
  --model MODEL --value-factors outputs/value \
  --windows outputs/routing_windows/windows.safetensors --output outputs/routing \
  --page-size 4 --pinned-pages 0
python -m core.fit_router base \
  --statistics-dir outputs/routing --output-dir outputs/routing/base --base-rank 16
python -m calibration.router fisher \
  --model MODEL --value-factors outputs/value \
  --windows outputs/routing_windows/windows.safetensors --output outputs/routing \
  --page-size 4 --pinned-pages 0
python -m core.fit_router fit \
  --statistics-dir outputs/routing --output-dir outputs/router --residual-rank 16
```

This example fits B16R16 with Page4 and no pinned prefix pages. Use the same
capture settings and Value factors for both stages. B16R16 specifies the Base
and residual dimensions; page size and token budget are configured separately.

### 3. Evaluate

Evaluate Value compression on WikiText-2:

```bash
python -m eval.ppl --model MODEL --value-factors outputs/value \
  --dataset wikitext2 --output outputs/ppl
```

Prepare RULER and LongBench inputs using the [data commands](eval/README.md#prepare-long-context-inputs),
then evaluate with the fitted router:

```bash
python -m eval.ruler --model MODEL --value-factors outputs/value \
  --method basiskv --router outputs/router --data RULER_DATA \
  --page-size 4 --pinned-pages 0 --routed-tokens 2048 --output outputs/ruler
```

| Evaluation | Entrypoint | Instructions |
| --- | --- | --- |
| Perplexity, MCQ, RULER, LongBench | `eval.ppl`, `eval.mcq`, `eval.ruler`, `eval.longbench` | [Quality evaluation](eval/README.md) |
| Throughput, latency, memory | `benchmarks.tp`, `benchmarks.decode`, `benchmarks.memory` | [Performance benchmarks](benchmarks/README.md) |
| Appendix experiments | `appendix/` | [Appendix](appendix/README.md) |

Main quality evaluation uses Page4 without a pinned prefix: RULER selects
2,048 historical tokens plus 64 recent tokens (2,112 total), and LongBench
selects 256 plus 64 (320 total). CUDA routing benchmarks default to the same
Page4, no-sink configuration as RULER and accept `--page-size {1,2,4,8,16,32}`,
independently of Value rank. Both use 2,048 selected historical tokens and 64
recent tokens. See the benchmark guide for system commands and the appendix
guide for table-specific evaluation commands.

## Repository Structure

```text
core/          Joint V/O ALS, rank allocation, and Page-Fisher fitting
calibration/   C4 and synthetic routing windows, activation statistics, and KL probes
inference/     Hugging Face and tensor-parallel vLLM adapters
basiskv_kernels/       CUDA and Triton attention, routing, and communication
eval/          Quality evaluation
appendix/      Appendix experiments
benchmarks/    Throughput, latency, and memory measurements
tests/         Compact tensor-parallel adapter verification
```

Model weights, datasets, fitted factors, and experiment outputs are supplied
separately. The commands above generate calibration windows and fitted factors.

For Qwen3.5 gated Value and Gated-DeltaNet fitting, see [calibration](calibration/README.md).
