# Appendix experiments

One entrypoint per appendix table or figure. Everything else runs through the
main pipeline (`calibration.windows`, `calibration.value`, `calibration.router`, `core.fit_value`, `core.allocate`,
`core.fit_router`, `eval.*`) with the settings listed below. Run from the
`BasisKV/` root. The entries below cover the Dense and BasisKV rows of each table.

| Result | Code |
| --- | --- |
| Table 5, WikiText-2 calibration | `wikitext_windows.py`, then the main pipeline |
| Table 6, generation quality and calibration domain | `generation.py` |
| Table 7, Group-SVD comparison | `group_svd.py`, then `eval.ppl` and `eval.mcq` |
| Table 8, C4-only Page32 router | Page32, one pinned page, `eval.ruler --routed-tokens 1984`, `routing_tables.py` |
| Table 9, Base and residual components | `core.fit_router` at the four rank pairs, `routing_tables.py` |
| Table 10, page granularity | main pipeline at `--page-size 1/4/8/32`, `routing_tables.py` |
| Table 11, dense-Value routing | `--dense-v` calibration, Page4, no pinned page, `eval.ruler --routed-tokens 2048`, `routing_tables.py` |
| Table 12, Value compression with KVQuant | `kv_quantization.py ppl` |
| Table 13, routing with quantized caches | `kv_quantization.py longbench` |
| Figure 6, Key predictability from Value | `key_predictability.py` |

## Figure 5: ALS initialization and convergence

Fit each rank and initialization at the plotted sweep budgets, then evaluate
WikiText-2. Every fit ends with a joint decoder refit.

```bash
for rank in 64 96; do
  for init in activation-aware random-orthogonal; do
    for sweeps in 0 1 2 4 6 8 12 16; do
      bank="outputs/convergence/${init}_r${rank}_s${sweeps}"
      python -m core.fit_value fit-shard --snapshot-dir SNAPSHOTS \
        --output-dir "$bank" --rank "$rank" --initialization "$init" \
        --seed 42 --encoder-sweeps "$sweeps"
      python -m core.fit_value merge --output-dir "$bank"
      python -m eval.ppl --model MODEL --value-factors "$bank" \
        --dataset wikitext2 --output "$bank/ppl"
    done
  done
done
```

The reconstruction metric is recorded in each bank's `results.json`.

## Table 5: WikiText-2 calibration

```bash
python -m appendix.wikitext_windows --model MODEL --output WT2_WINDOWS --fit-windows 256 --heldout-windows 64
python -m calibration.value --model MODEL --windows WT2_WINDOWS/windows.safetensors --output WT2_SNAPSHOTS
python -m core.fit_value fit-shard --snapshot-dir WT2_SNAPSHOTS --output-dir WT2_R96 --rank 96
python -m core.fit_value merge --output-dir WT2_R96
python -m eval.ppl --model MODEL --value-factors WT2_R96 --dataset wikitext2 --output PPL
python -m eval.mcq --model MODEL --value-factors WT2_R96 --output MCQ
```

The 75% and 50% budgets of the table use rank allocation on the WikiText-2
banks exactly as the main pipeline does on C4 (`core.allocate`).

## Table 6: generation quality and calibration domain

```bash
python -m appendix.generation --model MODEL --value-factors FACTORS \
  --tasks gsm8k,minerva_math500,mbpp_plus_full,ifeval --confirm-run-unsafe-code --output OUT
python -m appendix.generation --model QWEN35 --gated-value-bank V_BANK --gdn-bank GDN_BANK \
  --chat-template auto --thinking off --tasks gsm8k,minerva_math500,mbpp_plus_full,ifeval \
  --confirm-run-unsafe-code --output OUT
```

Few-shot counts and generation caps follow the paper: GSM8K 5-shot/512,
MATH500 4-shot/4096, MBPP+ 0-shot/2048, IFEval 0-shot/1280; decoding is
greedy. The MBPP+ task definition is in `tasks/`. Qwen3 and Llama run as base
completions; Qwen3.5 uses its chat template with thinking disabled. The paper's
numbers were produced with vLLM through the same lm-eval tasks; this entrypoint
runs the Hugging Face model and is slower on 32B models.

The C4 rows use the root calibration workflow. Generate the target model's
own responses to mathematics and code prompts, then form the two Table 6
calibration recipes:

```bash
python -m calibration.reasoning_windows --model MODEL \
  --c4-windows C4_WINDOWS/windows.safetensors --output outputs/domain_windows
```

`outputs/domain_windows/mixed/windows.safetensors` contains 128 C4 + 64 math +
64 code fit windows; `reasoning/windows.safetensors` contains 128 math + 128
code fit windows. Both retain 64 separate C4 diagnostic windows. Every window
has 2,048 tokens. Refit with these windows using `calibration.value` (Qwen3)
or `calibration.qwen35 capture` (Qwen3.5), keeping the chosen layer ranks fixed
across calibration domains.

## Table 7: Group-SVD

```bash
python -m appendix.group_svd --snapshot-dir SNAPSHOTS --output-dir GSVD_R96 --rank 96
python -m eval.ppl --model MODEL --value-factors GSVD_R96 --dataset wikitext2 --output PPL
python -m eval.mcq --model MODEL --value-factors GSVD_R96 --output MCQ
```

`group_svd.py` writes the `basiskv.value_factors.v1` layout; the held-out
relative MSE column of the table is `aggregate.mean_heldout_factor_dtype_relative_mse`
in its `results.json`, as for the BasisKV bank.

## Tables 8 to 11: routing ablations

These tables use `eval.ruler` or `eval.longbench` with the Value configuration,
router, page size, support budget, and task subset specified below, followed
by `routing_tables.py` for per-task scores and paired bootstrap intervals:

```bash
python -m appendix.routing_tables --arms full=RUN_FULL b16r16=RUN_B16R16 r16=RUN_R16 r32=RUN_R32 b16=RUN_B16 \
  --reference b16r16 --output TABLE
```

- Table 8: calibrate and fit the router on C4-only windows with
  uniform V96 and `--page-size 32 --pinned-pages 1` in both capture stages.
  Evaluate with `--page-size 32 --pinned-pages 1 --routed-tokens 1984`
  on the eleven tasks listed in the command below.
- Table 9: run the moments/base/Fisher sequence in separate directories for
  `--base-rank 16` and `--base-rank 0`, passing the same base rank to both
  capture stages and the base fitter. Fit residual ranks 16 and 0 from the
  base-16 statistics, and ranks 16 and 32 from the base-0 statistics.
  B16 is base rank 16 with residual rank 0; R16 and R32 are base rank 0. The
  table uses 30 prompts per task (`eval.ruler --samples-per-task 30`).
- Table 10: repeat the router calibration, fit, and evaluation at
  `--page-size 1`, `4`, `8`, and `32`; LongBench and the five common RULER tasks.
- Table 11: omit `--value-factors` everywhere and calibrate with
  `--dense-v --page-size 4 --pinned-pages 0` in both capture stages.
  Evaluate with `--page-size 4 --pinned-pages 0 --routed-tokens 2048`
  on the same eleven tasks.

Table 8 uses 32 pinned + 1,952 score-selected + 64 recent tokens (2,048 total).
Table 11 uses 2,048 score-selected + 64 recent tokens (2,112 total), with no sink.
Use the fitting workflow in the [benchmark guide](../benchmarks/README.md#commands),
setting `--page-size 32 --pinned-pages 1` and C4-only windows for Table 8;
use the Page4 dense-Value variant for Table 11. Then evaluate:

```bash
# Table 8: uniform V96; use C4-only windows for the router calibration.
python -m eval.ruler --model MODEL --value-factors V96_FACTORS \
  --data RULER_DATA --method basiskv --router PAGE32_V96_ROUTER \
  --page-size 32 --pinned-pages 1 --routed-tokens 1984 \
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2 \
  --samples-per-task 100 --output TABLE8_B16R16

# Table 11: original dense Values and output projection.
python -m eval.ruler --model MODEL --data RULER_DATA \
  --method basiskv --router PAGE4_V128_ROUTER \
  --page-size 4 --pinned-pages 0 --routed-tokens 2048 \
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2 \
  --samples-per-task 100 --output TABLE11_B16R16
```

`--tasks` takes the complete list, not an additional task. `--tasks all` selects
thirteen tasks and is not the eleven-task protocol above. For each Full-K
control, retain the same Value configuration and task list, use `--method full`,
and omit `--router`.

## Tables 12 and 13: quantized KV caches

```bash
git clone https://github.com/SqueezeAILab/KVQuant external/KVQuant
python -m appendix.kv_quantization calibrate --model MODEL --value-factors FACTORS --output NUQ4
python -m appendix.kv_quantization ppl --format nuq4 --quantizers NUQ4 -- \
  --model MODEL --value-factors FACTORS --c4-windows C4_VALIDATION/windows.safetensors --output PPL_NUQ4
python -m appendix.kv_quantization longbench --format fp8 -- \
  --model MODEL --value-factors FACTORS --method basiskv --router ROUTER --data LONGBENCH --output LB_FP8
```

Both formats are quantize-dequantize simulations applied to the module outputs
that feed the cache; the BF16 rows are the plain `eval.ppl` and `eval.longbench`
runs. NUQ4 follows KVQuant: pre-RoPE Keys with static per-channel codebooks and
calibrated outlier thresholds, Value latents per token with dynamic 1% outliers,
Fisher-weighted codebooks from 16 WikiText-2 training segments. FP8 uses one
E4M3 scale per token and KV head for Keys and one per routing page and KV head
for Value latents. FP8 Keys are quantized after RoPE, before cache insertion;
NUQ4 Keys are quantized before RoPE.

## Figure 6: Key predictability from Value

```bash
python -m appendix.key_predictability --model MODEL --value-factors FACTORS \
  --windows WINDOWS/windows.safetensors --output OUT
```

Fit and held-out windows come from the window manifest. The output has the
held-out centered explained Key energy per layer and KV group for local and
all-group, compressed and raw Value predictors; `key_predictability.pdf` is the
figure when matplotlib is installed.
