# Appendix experiments

One entrypoint per appendix table or figure. Everything else runs through the
main pipeline (`calibration.windows`, `calibration.value`, `calibration.router`, `core.fit_value`, `core.allocate`,
`core.fit_router`, `eval.*`) with the settings listed below. Run from the
`BasisKV/` root. Baseline methods (PaLU, Loki, ShadowKV, LRQK) are not part of
this release; the rows below are the Dense and BasisKV rows of each table.

| Result | Code |
| --- | --- |
| Table 5, WikiText-2 calibration | `wikitext_windows.py`, then the main pipeline |
| Table 6, generation quality and calibration domain | `generation.py` |
| Table 7, Group-SVD comparison | `group_svd.py`, then `eval.ppl` and `eval.mcq` |
| Table 8, C4-only Page32 router | main pipeline with `--page-size 32 --pinned-pages 1`, `routing_tables.py` |
| Table 9, Base and residual components | `core.fit_router` at the four rank pairs, `routing_tables.py` |
| Table 10, page granularity | main pipeline at `--page-size 1/4/8/32`, `routing_tables.py` |
| Table 11, dense-Value routing | router calibrated with `--dense-v`, `routing_tables.py` |
| Table 12, Value compression with KVQuant | `kv_quantization.py ppl` |
| Table 13, routing with quantized caches | `kv_quantization.py longbench` |
| Figure 6, Key predictability from Value | `key_predictability.py` |

Figure 5 (ALS initialization and convergence) is not provided: the production
fitter uses the activation-aware initialization only. The sweep axis can be
reproduced by running `core.fit_value` with `--encoder-sweeps 0,1,2,...` and
`eval.ppl` on each bank. Table 4 is analytical.

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

The C4 calibration rows use `calibration.windows` defaults. The
Mixed and Reasoning rows need calibration windows built from the target model's
own responses to OpenThoughts3 prompts; the prompt-selection code of that
pipeline is missing from the source repository, so those two rows have no
release command.

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

All four tables are `eval.ruler` or `eval.longbench` runs that differ only in
how the router was calibrated and fitted, followed by `routing_tables.py` for
the per-task scores and paired bootstrap intervals:

```bash
python -m appendix.routing_tables --arms full=RUN_FULL b16r16=RUN_B16R16 r16=RUN_R16 r32=RUN_R32 b16=RUN_B16 \
  --reference b16r16 --output TABLE
```

- Table 8: calibrate and fit the router on C4-only windows with
  `--page-size 32 --pinned-pages 1`, then `eval.ruler --page-size 32 --pinned-pages 1`
  with all eleven tasks (`--tasks` adds `niah_single_3`).
- Table 9: four routers from one capture. `core.fit_router base --base-rank 16`
  and `--base-rank 0`; `core.fit_router fit --residual-rank 16`, `32`, and `0`.
  B16 is base rank 16 with residual rank 0; R16 and R32 are base rank 0. The
  table uses 30 prompts per task (`eval.ruler --samples-per-task 30`).
- Table 10: repeat the router calibration, fit, and evaluation at
  `--page-size 1`, `4`, `8`, and `32`; LongBench and the five common RULER tasks.
- Table 11: the dense-Value control omits `--value-factors` everywhere and
  calibrates the router with `--dense-v`.

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
for Value latents. The paper's FP8 variant quantized Keys after RoPE; here both
formats quantize the Key before rotation, at the same point as KVQuant.

## Figure 6: Key predictability from Value

```bash
python -m appendix.key_predictability --model MODEL --value-factors FACTORS \
  --windows WINDOWS/windows.safetensors --output OUT
```

Fit and held-out windows come from the window manifest. The output has the
held-out centered explained Key energy per layer and KV group for local and
all-group, compressed and raw Value predictors; `key_predictability.pdf` is the
figure when matplotlib is installed.
