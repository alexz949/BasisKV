# BasisKV

Code staging directory for the BasisKV paper release. It contains the selected
compression, routing, inference, evaluation, benchmark, and baseline code plus
their local source dependencies. PaLU implementations and comparison arms are excluded. Experimental outputs, model weights, datasets,
the paper PDF, and the original repository's test suite are not included.

The original source tree is preserved. Continue release cleanup in this directory.
Implementation and entrypoint code lives directly in the eight functional
directories below. The installable distribution is named `basiskv`; Python
imports use the functional package names, such as `core` and `inference`.

## Layout

```text
BasisKV/
  core/          # V/O joint ALS and Page-Fisher router fitting
  calibration/   # Calibration data, activation capture, and statistics
  kernels/       # Attention, routing, and communication used by the paper runtimes
  inference/     # Shared attention, routing, and TP/vLLM adapters
  eval/          # PPL, MCQ, RULER, LongBench, and shared quality tools
  appendix/      # One entrypoint per appendix table or figure
  benchmarks/    # Serving and kernel performance code
  others/        # Historical experiments, baselines, and checkpoint preparation
  pyproject.toml
  requirements.txt
```

These are the eight functional directories. Main CUDA sources remain in
`kernels/csrc/`; archived CUDA experiments live in `others/csrc/`.
Upstream LongBench scoring resources remain under `eval/`.
The MBPP+ task definitions used by the appendix live in `appendix/tasks/`.
The Value fitting entrypoint is
`core/fit_value.py`. Its implementation is organized into five files:

```text
core/
  fit_value.py      # CLI, fit settings, per-layer fitting, and sharding
  als.py            # Weighted initialization, full-layer ALS, Cholesky, and CG
  snapshots.py      # Snapshot manifests, layer payloads, and covariance blocks
  factor_io.py      # Factor records: write, verify, resume, merge, and assemble
  model_layout.py   # Model dimensions and query/KV head grouping
  allocate.py       # Layerwise rank allocation from uniform banks and KL probes
```

Standard MHA/GQA models share the same fitting entrypoint and obtain their
dimensions from calibration metadata. The old `value.py`, `geometry.py`, and
`gqa_routed_ov_joint.py` modules have been removed; use the modules above.
Page-Fisher routing uses four more files:

```text
core/
  fit_router.py    # CLI: `base` fits the affine Value-to-Key base, `fit` the residual router
  page_fisher.py   # Fisher statistics, teacher Grams, loss, and the alternating CG solves
  router.py        # Affine base fitting from moments and the routing feature transforms
  router_io.py     # Statistics and factor records on disk; shared JSON/tensor helpers
```

These ten implementation files and `__init__.py` are the entire `core/` package.
The softmax-Fisher objectives of the historical S80 experiments (joint [V, K]
rows, explicit target maps, active row subsets, adapter systems) live in
`others/softmax_fisher_routing.py`; the release router is their special case
with residual-key features only. Historical joint-payload experiments,
fixed-model/rank launchers, token routing, KL rank search, output-projection
experiments, and Qwen3.5 gated-V fitting are preserved in `others/`; gated-V
retains its distinct fitting objective.
Calibration uses `calibration/calibrate.py` for windows and statistics and
`calibration/probe_kl.py` for the single-layer KL probes of rank allocation. Historical model-specific
data preparation, raw activation capture, S80, MLP, quantization, and gated-V/GDN
collectors live in `others/`. Tensor parallel runtime helpers live in `inference/`.

Ordinary Hugging Face inference uses four files:

```text
inference/
  model.py        # Shared model loading and current Value/router factor installation
  attention.py    # Compact V attention for Llama and Qwen3
  routing.py      # Page mass, GQA aggregation, recent support, and routing sidecars
  qwen35.py       # Gated V attention and GDN output projection
```

Calibration and evaluation use the same loader and factor installation functions.
The main path reads `core.fit_value` output directly; checkpoint repackaging is
unnecessary. It shares one page selector and uses Page4, no pinned sink, and an
exact recent-64 tail by default. RoPE transforms live in `core/router.py`, so
fitting does not import inference code.

Three shared modules implement tensor-parallel vLLM inference:

- `tp.py`: current factor loading, head partitioning, and projection folding.
- `backends.py`: paged DiffKV attention and the single-KV-head specialization.
- `vllm.py`: Qwen3 compact/folded/dense adapters, prepared communication, worker
  inspection, and the Qwen3.5 quality adapter.

Qwen3 adapters read model geometry and rank from the config and factors. The
same adapter serves 8B/32B and supported TP partitions; no model-size-specific
classes are registered. Compact inference requires QK128 and V64/V96, BF16,
no quantization, and one pipeline stage. The paged path supports complete
prefill and single-token decode. The prepared single-KV path retains its
four/eight local query-head kernel specialization. Folded inference uses native
dense cache slots and is for quality evaluation, not compact-cache timing.

The vLLM plugin lazily registers four model classes. `inference/` contains eight
Python files including `__init__.py`. The HF performance runtime is shared by the decode and memory entrypoints in
`benchmarks/attention.py`; fixed-shape historical drivers live in `others/`.
Archived uniform/KL generation schedules are loaded only by
`appendix/folded_vllm.py`; the main adapters read `core.fit_value` output.
Checkpoint builders, identity preparation, historical rank schedules, sparse
baselines, and experimental runtime variants now live in `others/`. Their
presence does not make their old factor formats part of the main loader.

## CUDA and Triton kernels

`kernels/` holds only the kernels the paper runtimes call:

| File | Used by | What it does |
| --- | --- | --- |
| `sparse_attention.py` | `benchmarks/attention.py` | Triton split-K decode over selected tokens; `selected_ids` indexes V, optional `key_slots` indexes a Key slot cache |
| `compressed_v_prefill.py` | `inference/backends.py` | Triton causal prefill with dense Keys and compact Values |
| `diffkv_prefill.py`, `diffkv_decode.py` | `inference/backends.py` | vLLM paged prefill and decode for QK128 GQA with 64- or 96-wide Values (adapted from vLLM) |
| `decode.py` + `csrc/paged_attention.cu` | `benchmarks/attention.py`, dense arm | CUDA decode attention over selected pages of a GPU-resident K128 cache |
| `decode.py` + `csrc/router.cu` | `benchmarks/attention.py`, BasisKV arm | Page32 B16R16 GQA4 router: decode-time append, page log-sum-exp scan, group-max page selection with support packing, host-Key slot refresh |
| `allgather.py` + `csrc/allgather.cpp` | `inference/vllm.py`, `benchmarks/attention.py` | Equal-width in-place NCCL AllGather whose feature-major arena feeds one decoder GEMM |
| `build.py` | every extension | One lazy CUDA/NCCL extension builder (`CUDA_HOME`, optional `BASISSERVE_EXT_BUILD_DIR`) |

The kernels implement the paper's system protocol and check it with asserts
rather than generalizing beyond it:

| Kernel | head_dim | Query heads per KV head | Value width | Page | Base/residual rank |
| --- | --- | --- | --- | --- | --- |
| paged attention | 128 | any | compile-time; 64, 96, 128 used | 32 | none |
| router | 128 | 4 | compile-time | 32 | 16 (8 also compiles) |
| DiffKV (vLLM) | 128 | 4 or 8 | 64 or 96 | any | none |
| sparse attention, prefill | up to 256 | any | up to 256 | none | none |

Extensions compile per Value width on first use through `kernels.decode.load_router`
and `kernels.decode.load_paged_attention`; the router needs sm_80 or newer. The
router reproduces the BF16 rounding boundaries of `inference/routing.py`, so the
GPU selects the same pages as the PyTorch reference. Full-warp reductions live in
`csrc/warp.cuh`.

Historical variants live in `others/` with their CUDA sources in `others/csrc/`:
mapped-host paged attention, candidate-page routing, the WMMA router fallback
and warp/unroll sweeps, the separate slot-planning and fused postprocess
extensions, the Triton dense-Key decode, R32/Page64 routing, resident-CUDA
decode, SageAttention adapters, packed NUQ4/A8 serving, and the Ampere TP8 runner.
They are imported and compiled only by their archived callers. The main
`core/`, `calibration/`, `inference/`, `eval/`, and `appendix/` packages do not
import these experiments.

## Installation

From this directory, in the existing `lowrank` or `basis` environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m core.fit_value --help
```

Run commands from the `BasisKV/` root using `python -m <package>.<module>`.
Imports, subprocess source paths, CUDA source locations, and the vLLM plugin
registration follow this layout. There is no enclosing `scripts/` package.
`requirements.txt` preserves the research dependency list; it is not an
environment lockfile. Different model and inference paths require compatible
Transformers/vLLM versions. The adapters use the DiffKV interfaces in the paper's
vLLM 0.18 build. CUDA kernels additionally require a compatible CUDA
toolchain and, for communication kernels, NCCL development files.

## Calibration and Value fitting

From the `BasisKV/` root, the four calibration stages share one entrypoint:

```bash
python -m calibration.windows \
  --model MODEL --output WINDOWS --sequence-length 2048 \
  --fit-windows 32 --heldout-windows 16
python -m calibration.value \
  --model MODEL --windows WINDOWS/windows.safetensors --output SNAPSHOTS
```

`windows` samples one excerpt per distinct document and packs excerpts into
fixed-length windows from C4 (`allenai/c4` en, train split, pinned revision);
`--seed` selects the stream (default 20260821, the paper's calibration bank). Fit and
held-out windows use disjoint documents. Under datasets 4.4.2 the builder regenerates the
paper's Qwen3-8B-Base windows byte for byte; the seeds of each bank are listed in
`calibration/windows.py`.
`value` hooks the attention output projection and accumulates full-token
covariances in FP64, in row chunks. It writes normalized fit/held-out matrices
and the dense output weight in the snapshot format consumed by the fitter.

```bash
python -m core.fit_value fit-shard \
  --snapshot-dir SNAPSHOTS --output-dir FACTORS --rank 96
python -m core.fit_value merge --output-dir FACTORS
```

Model geometry and the fit/held-out window partition come from the snapshot's
`manifest.json`; the fitter accepts only the covariance snapshots that
`calibration.value` writes. Held-out error is reported per layer and
read by rank allocation, never fitted.
The default is six ALS sweeps (`--encoder-sweeps`), sixteen fixed CG iterations
per encoder update (`--cg-iterations`), FP32 work, and BF16 factors. The final
decoder is always refitted.
The solver keeps one FP64 objective (`als.Quadratic`: covariance, cross term,
constant) and casts it to the work dtype for the solves. It alternates a
closed-form output-decoder solve with one CG update per KV group on the shared
Value encoders, followed by QR normalization. The objective is full-layer with
cross-head covariance and one shared rank across KV groups. Initialization is
`initialize_weighted_svd` on the damped fit covariance. `fit_joint` returns the
final encoders, decoders, and a solver record with losses and one entry per CG
update; held-out evaluation runs after fitting. The rank equal to `head_dim`
writes the exact dense endpoint without fitting.
Use `--layer-shard-index` and `--layer-shard-count` for separate workers.
Substantial fitting and evaluation should be launched through the scheduler.

The output format is `basiskv.value_factors.v1`. Geometry is recorded in the
factors, so merge does not require the original snapshot directory. The old
model-specific fitter entrypoints and factor-format tags are not supported.

### Rank allocation

The 75% and 50% Value budgets of the paper allocate one rank per layer
(Section 3.3, Appendix A.2). Fit uniform banks at every candidate rank first;
the paper uses 32, 48, 64, 80, 96, 112, and 128, where 128 is the exact dense
endpoint. Then measure the two single-layer probes per layer against the dense
teacher and allocate:

```bash
python -m calibration.windows \
  --model MODEL --output PROFILE --sequence-length 2048 --fit-windows 128 --heldout-windows 0 --seed 20260828
python -m calibration.probe_kl \
  --model MODEL --windows PROFILE/windows.safetensors \
  --banks 32=B32 64=B64 96=B96 --anchor-rank 64 --low-rank 32 --high-rank 96 --output PROBES
python -m core.allocate \
  --banks 32=B32 48=B48 64=B64 80=B80 96=B96 112=B112 128=B128 \
  --probes PROBES/probes.json --target-average-rank 96 --output FACTORS_V96
```

`probe_kl` installs the anchor bank in every layer, moves one layer at a time to
the low and the high probe rank, and records the change of the mean
full-vocabulary KL(dense teacher || model) over all prediction positions of the
profile windows. The profile windows must be disjoint from the fitting windows, so
they use another `--seed`. The paper's profile windows are rows 320 to 447, cut to
2,048 tokens, of the bank built with `--sequence-length 4096 --fit-windows 512
--heldout-windows 128 --seed 20260828`; the paper also estimated each KL on 1,024
sampled positions per window, where `probe_kl` uses all of them. `--layer-shard-index` and
`--layer-shard-count` split the layers across workers; rerun with `--merge`
afterwards. `allocate` turns the probes and the held-out reconstruction errors of
the banks into the one-sided sensitivities of equation 3.3, predicts the KL
change of every candidate rank, solves the exact-budget dynamic program, and
copies the chosen layers into one factor directory. Its `results.json` has the
same layout as a merged uniform bank plus `layer_ranks` and an `allocation`
record; `fit_config` then carries no `cache_rank_per_head`. HF inference and
the quality evaluators accept allocated and uniform banks alike. The TP/vLLM
adapters and the system benchmarks read uniform banks only.

## Page-Fisher routing

Capture moments, fit the Value-to-Key base, capture Fisher statistics using that
base, then fit the residual router. For Llama, Qwen3, and Qwen3.5, from the
`BasisKV/` root:

```bash
python -m calibration.router moments \
  --model MODEL --value-factors FACTORS --windows WINDOWS.safetensors --output CAPTURE \
  --base-rank 16 --residual-rank 16
python -m core.fit_router base \
  --statistics-dir CAPTURE --output-dir CAPTURE/base --base-rank 16
python -m calibration.router fisher \
  --model MODEL --value-factors FACTORS --windows WINDOWS.safetensors --output CAPTURE \
  --base-rank 16 --residual-rank 16
python -m core.fit_router fit \
  --statistics-dir CAPTURE --output-dir ROUTER_BANK --residual-rank 16 \
  --device cuda:0 --sweeps 40 --pcg-iterations 100
```

Use the same settings for both replays: every record carries the capture
protocol (ranks, page size, pinned prefix, query grid, window ids, geometry),
and `fisher` and `core.fit_router` refuse records whose protocol differs. The
teacher is the deployed model, so `--value-factors` is the merged
`core.fit_value` bank; Qwen3.5 passes `--gated-value-bank` and `--gdn-bank`
instead (its gated-V and GDN collectors remain historical objectives in
`others/`). `--dense-v` is the dense-Value control of Table 11: a dense teacher
with identity Value codes. Page size defaults to 4 and `--pinned-pages` to 0.
Sequence length and the fit/held-out partition default to the window manifest.
`--config` supplies an HF config for an extended context, as in the evaluators;
Qwen3-8B uses static YaRN with factor 4 for 128K routing.

`--layers` selects attention layers. `--window-shard-index` and
`--window-shard-count` divide windows for Value or moments capture; after all
shards finish, rerun the same command with `--merge`. Merge reads only the
stored statistics and does not load the teacher. Fisher shards write separate
window records and need no merge. Query selection uses only the joined
fit-window queries, so the shard count does not change the selection.

Query positions follow Appendix B.3: queries are whitened per head, and a
diagonal-pivoted Cholesky within four positional strata selects 16 fit and
eight diagnostic positions per stratum from a 16-token candidate grid. These
are constants in `calibration/router.py`. `--device-map` spreads a large
teacher over several GPUs.

Moment records (`basiskv.router_moments.v1`) include the Value encoder, so
base and residual fitting require only saved statistics; they do not load model
weights, Transformers, tokenizers, or evaluation scripts. The base rank must
match the capture protocol; the residual rank can be changed without another
teacher replay. Head counts and dimensions come from the tensors, not
model-name constants. Fisher windows are checked against the base record's
protocol before fitting.

The residual solver (`page_fisher.fit_router`) starts each group's encoder from
the top eigenvectors of its summed Fisher Gram and alternates Jacobi-preconditioned
CG solves of the per-head query maps and the per-group encoders; the fitted
objective is `page_fisher.fisher_loss`, normalized by the teacher energy in the
record. `FisherStatistics` holds the per-query routing queries, Page-Fisher
Grams, head grouping, score scale, and teacher energy of one layer.
Both fitting stages support `--layers`, `--layer-shard-index`,
`--layer-shard-count`, and verified `--resume`. The residual fit also exposes
`--pcg-tolerance`, `--relative-damping`, and optional `--allow-tf32`.
If no held-out windows were captured, the record labels its held-out NMSE as
in-sample. The Score-MSE ablation uses the same solver with its objective
recorded in the capture protocol. Fitted tensors retain the base/residual names
used by the inference adapters; pass `ROUTER_BANK` to the evaluation/inference loader.
Run substantial capture and fitting jobs through the scheduler.

## Quality evaluation

`eval/` has four main entrypoints: `ppl.py`, `mcq.py`, `ruler.py`, and
`longbench.py`, with shared loading/scoring tools. Commands and data formats
are documented in [`eval/README.md`](eval/README.md).
`benchmarks/` has three performance entrypoints: `tp.py` for vLLM throughput
and TTFT, `decode.py` for GPU-resident TP1 attention/model latency, and
`memory.py` for TP decode throughput versus memory. They share one HF attention
runtime and load current factors without archived `results/` dependencies.
Commands, protocol differences, and supported model/runtime combinations are
documented in [`benchmarks/README.md`](benchmarks/README.md). The appendix
tables and figures each have one entrypoint or one documented pipeline setting
in [`appendix/`](appendix/README.md).

The main paper's routing protocol is Page4/B16R16: RULER-128K uses ten tasks
with 100 prompts per task and 2,048 routed tokens plus 64 recent tokens;
LongBench uses nine tasks with official prompts longer than 4,096 tokens under
the evaluated tokenizer, and 256 routed tokens plus 64 recent tokens. Older
Page32 controls, eleven-task RULER summaries, and small pilots have different
protocols. Historical model-specific runners now live in `others/`; their
defaults remain unchanged. The main runners load current Value/router factors
and implement Full-K and BasisKV quality evaluation. No saved experiment
results are bundled with this release directory.

## Remaining release cleanup

This is a complete working area for the selected release scope, not a claim that
every historical entrypoint is independently runnable. Value fitting has a
shared entrypoint, as does Page-Fisher fitting. Historical launchers in `others/` retain machine-specific defaults.

- `others/prepare_qwen35_mixed_reasoning_calibration.py` imports
  `calibration.prepare_qwen3_32b_reasoning_calibration`, which is already missing
  from the source repository. The Mixed and Reasoning calibration rows of Table 6
  therefore have no release command.
- Historical TP1 baseline launchers in `others/`, including
  `run_tp1_paper_scaling.py` and `audit_tp1_request.py`, require unbundled frozen
  source and external baselines. They are not dependencies of the three public
  benchmark entrypoints. The compact vLLM adapter currently supports Qwen3;
  the Llama Figure 3 compact-vLLM arm is not provided by this adapter.
- Other historical launchers retain local model/data paths and external baseline
  checkout requirements. Their presence does not imply validation in a clean
  environment. Model weights, calibration inputs, and evaluation datasets must
  be supplied separately.

Upstream source attribution is retained where present, including
`eval/longbench_official/README.md`.
