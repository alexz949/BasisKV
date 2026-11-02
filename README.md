# BasisKV

BasisKV combines low-rank Value compression with Page-Fisher Key routing for
efficient LLM inference. This repository provides calibration, joint V/O fitting,
routing, inference kernels, and quality and performance evaluation code.

## Installation

Use Python 3.10+ with a CUDA-enabled PyTorch installation.

```bash
git clone https://github.com/alexz949/BasisKV.git
cd BasisKV
pip install -r requirements.txt
pip install -e . --no-deps
```

CUDA extensions compile on first use and require a compatible CUDA toolkit;
communication kernels also require NCCL development files. vLLM benchmarks
require the paper runtime's DiffKV interfaces; see [benchmark setup](benchmarks/README.md).

## Quick Start

Run commands from the repository root. Replace `MODEL` with a local Hugging Face
Llama or Qwen3 model directory. Model weights and datasets are supplied separately.

### 1. Calibrate and fit Value factors

Prepare C4 calibration windows, collect activation statistics, and fit joint V/O
factors. This example retains 96 Value dimensions per KV head.

```bash
python -m calibration.windows \
  --model MODEL --output outputs/windows --sequence-length 2048 \
  --fit-windows 32 --heldout-windows 16
python -m calibration.value \
  --model MODEL --windows outputs/windows/windows.safetensors \
  --output outputs/statistics
python -m core.fit_value fit-shard \
  --snapshot-dir outputs/statistics --output-dir outputs/value --rank 96
python -m core.fit_value merge --output-dir outputs/value
```

For layerwise rank allocation, fit uniform banks at candidate ranks, collect
KL probes with `calibration.probe_kl`, then select ranks with `core.allocate`.
Use `python -m <module> --help` for each entrypoint's options.

### 2. Fit the Page-Fisher router

Use the merged Value factors and calibration windows at your target context
length (`ROUTING_WINDOWS/windows.safetensors`, prepared with `calibration.windows`).
The model must support that length; pass `--config` to both capture commands
when using an extended-context configuration.

```bash
python -m calibration.router moments \
  --model MODEL --value-factors outputs/value \
  --windows ROUTING_WINDOWS/windows.safetensors --output outputs/routing
python -m core.fit_router base \
  --statistics-dir outputs/routing --output-dir outputs/routing/base --base-rank 16
python -m calibration.router fisher \
  --model MODEL --value-factors outputs/value \
  --windows ROUTING_WINDOWS/windows.safetensors --output outputs/routing
python -m core.fit_router fit \
  --statistics-dir outputs/routing --output-dir outputs/router --residual-rank 16
```

Defaults are base rank 16, residual rank 16, page size 4, and no pinned prefix
pages. Use the same capture settings and Value factors for both stages.

### 3. Evaluate

Evaluate Value compression on WikiText-2:

```bash
python -m eval.ppl --model MODEL --value-factors outputs/value \
  --dataset wikitext2 --output outputs/ppl
```

For routed long-context evaluation, supply the router and prepared benchmark data:

```bash
python -m eval.ruler --model MODEL --value-factors outputs/value \
  --method basiskv --router outputs/router --data RULER_DATA --output outputs/ruler
```

| Evaluation | Entrypoint | Instructions |
| --- | --- | --- |
| Perplexity, MCQ, RULER, LongBench | `eval.ppl`, `eval.mcq`, `eval.ruler`, `eval.longbench` | [Quality evaluation](eval/README.md) |
| Throughput, latency, memory | `benchmarks.tp`, `benchmarks.decode`, `benchmarks.memory` | [Performance benchmarks](benchmarks/README.md) |
| Appendix experiments | `appendix/` | [Appendix](appendix/README.md) |

Quality evaluation uses Page4 routers; CUDA routing benchmarks use Page32 and
need separately fitted routers. See the linked instructions for data preparation,
context configuration, supported runtimes, and paper protocols.

## Repository Structure

```text
core/          Joint V/O ALS, rank allocation, and Page-Fisher fitting
calibration/   Calibration windows, activation statistics, and KL probes
inference/     Hugging Face and tensor-parallel vLLM adapters
kernels/       CUDA and Triton attention, routing, and communication
eval/          Quality evaluation
appendix/      Appendix experiments
benchmarks/    Throughput, latency, and memory measurements
others/        Historical experiments and baselines
```

Historical scripts in `others/` may require external baseline code and local
model/data paths. Experiment outputs and checkpoints are not bundled.
