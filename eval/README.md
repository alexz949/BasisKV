# Quality evaluation

Run these four entrypoints from the `BasisKV/` root. Install the release
requirements first; the inference adapter targets Transformers 5.17.0.

| Entrypoint | Default protocol |
| --- | --- |
| `ppl.py` | WikiText-2 test and 128 fixed C4 validation document windows, 2,048 tokens per window, batch size 2; FP32 cross entropy |
| `mcq.py` | ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, OpenBookQA; zero-shot, batch size 8; `acc_norm` where available, otherwise `acc` |
| `ruler.py` | Official base-completion prompts, seed-42 frozen data, 128K, ten tasks excluding `niah_single_3`, 100 prompts per task |
| `longbench.py` | Nine official LongBench-v1 tasks, raw rendered prompt length strictly greater than 4,096 tokens under the evaluated tokenizer, no truncation |

`common.py` shares provenance, greedy generation, per-example records, resume,
and task-balanced summaries. It delegates model loading and factor installation
to `inference/model.py`, as does calibration. `inference/attention.py` implements
the shared Llama/Qwen3 attention adapter. Its persistent Value cache is compact;
it temporarily pads the Value input for PyTorch SDPA. These runners report
quality. Measure speed and memory with `benchmarks/`.

## PPL and MCQ

```bash
python -m eval.ppl --model MODEL --value-factors FACTORS \
  --c4-windows C4_VALIDATION/windows.safetensors --output PPL_OUTPUT
python -m eval.mcq --model MODEL --value-factors FACTORS --output MCQ_OUTPUT
```

`--value-factors` reads the merged `basiskv.value_factors.v1` output of
`core.fit_value`. Omit it for a dense model or a separately converted HF
baseline checkpoint. Model dimensions and retained ranks come from the model
and factor tensors.

WikiText-2 is loaded through `datasets`; `--wikitext-tokens` can instead provide
a frozen safetensors `input_ids` stream. Complete non-overlapping chunks are
used, with no prediction across chunk boundaries. C4 uses the 128 fixed
validation windows built by `calibration.windows --split validation
--fit-windows 128 --heldout-windows 0` with the evaluated tokenizer; the adjacent
`manifest.json` records the split. Documents remain separate.
Use `--dataset wikitext2` or `--dataset c4` to run one dataset. `--limit` records
a partial evaluation.

## Prepare long-context inputs

Prepare the official RULER-v1 generator and its source data:

```bash
git clone --branch main https://github.com/NVIDIA/RULER external/RULER
pip install nltk wonderwords tenacity html2text beautifulsoup4 pyyaml
(
  cd external/RULER/scripts/data/synthetic/json
  python download_paulgraham_essay.py
  bash download_qa_dataset.sh
)
python -m eval.prepare_data ruler --ruler-root external/RULER --model MODEL \
  --sequence-length 131072 --samples 100 --seed 42 --output outputs/ruler_data
python -m eval.prepare_data longbench --output outputs/longbench_data
```

Use `outputs/ruler_data` for `RULER_DATA` and `outputs/longbench_data` for
`LONGBENCH_JSONL`. RULER prompts must be generated with the evaluated model's
tokenizer. The preparation command writes the manifest read by `eval.ruler`.
The main evaluator chooses the ten paper tasks; the generated data also include
the tasks used by the appendix.

For Qwen3-8B at 128K, prepare the paper's static YaRN configuration:

```bash
python -m calibration.context_config --model MODEL --output outputs/qwen3_128k
```

Pass `--config outputs/qwen3_128k` to both router capture stages and to the
long-context evaluation commands. The model weights and tokenizer still come
from `--model MODEL`.

## RULER and LongBench

```bash
python -m eval.ruler --model MODEL --value-factors FACTORS \
  --data RULER_DATA --method basiskv --router ROUTER \
  --page-size 4 --pinned-pages 0 --routed-tokens 2048 --output RULER_OUTPUT
python -m eval.longbench --model MODEL --value-factors FACTORS \
  --data LONGBENCH_JSONL --method basiskv --router ROUTER \
  --page-size 4 --pinned-pages 0 --routed-tokens 256 --output LONGBENCH_OUTPUT
```

`ROUTER` is the directory written by `core.fit_router fit`. The default page
size is 4. RULER selects 2,048 historical tokens per KV group and LongBench
selects 256; both retain 64 additional recent tokens, with no pinned sink page.
The total support budgets are therefore 2,112 and 320 tokens, respectively.
`--page-size` and `--pinned-pages` should match the router's calibration, which
uses `calibration.router ... --page-size 4 --pinned-pages 0` (both are
defaults). `--routed-tokens` counts all selected historical tokens, including
any pinned prefix; the 64 recent tokens are additional. For Page32 with one
pinned page and a total budget of 2,048, pass
`--page-size 32 --pinned-pages 1 --routed-tokens 1984`: 32 pinned tokens plus
1,952 score-selected tokens plus 64 recent tokens. See the appendix guide for
the Table 8 control and the Page4 Table 11 commands.
Base/residual ranks are read from the router record; use the paper's B16R16
factors for its main comparison. Use the same Value factors that calibrated the
router. Record fingerprints are retained in the output.

Use `--method full` and omit `--router` for the Full-K control on the same Value
checkpoint. Omitting Value factors as well evaluates the original dense model.
The generic main runners implement Full-K and BasisKV.

RULER takes the prepared seed-42 data in `<task>/validation.jsonl`.
Each record contains `input`, `outputs` (a list of reference answers), and an
optional `answer_prefix`. The data directory must also contain `manifest.json`
with `status: "complete"` and `protocol.sequence_length` matching the evaluation
length; retain the source hashes and tokenizer identity in that manifest.
LongBench reads the original `<task>.jsonl`
files and uses the vendored official prompts, generation caps, and metrics.
The nine tasks are `narrativeqa`, `multifieldqa_en`, `hotpotqa`, `musique`,
`dureader`, `gov_report`, `samsum`, `passage_retrieval_en`, and `lcc`.
Chat templates are applied when available, except for the official no-chat
tasks; `--chat-template none` makes that choice explicit. Thinking is disabled.

Models must support the requested context length. For an extended-context
configuration, provide `--config CONFIG.json`. `--device-map auto` can
distribute a model across GPUs.
Qwen3.5 uses `inference/qwen35.py` and separate
`--gated-value-bank` / `--gdn-bank` inputs instead of `--value-factors`.

For independent workers, use the same arguments and output directory with
`--num-shards N --shard-index I`. Each worker writes its assigned examples.
After all workers finish, repeat the command with `--num-shards N --stage
summarize`; summarization does not load the model. Completed examples resume
only when their protocol fingerprint matches. Missing examples prevent a
complete summary. Both task scores and the unweighted task mean are saved in
`result.json`.

## Appendix

Appendix-specific drivers and task definitions live in
[`../appendix/`](../appendix/README.md).

