"""Model loading, provenance, and greedy generation shared by quality runners."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys

import torch

from inference.tp import chunk_tokenwise_prefill
from core.router_io import read_json, sha256, write_json

NO_CHAT_TASKS = ('trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p')


def model_arguments(parser):
    parser.add_argument('--model', type=Path, required=True, help='local Hugging Face model directory')
    parser.add_argument('--value-factors', type=Path, help='merged core.fit_value output directory')
    parser.add_argument('--config', type=Path, help='HF config for an explicitly extended context')
    parser.add_argument('--gated-value-bank', type=Path, help='Qwen3.5 gated-V bank (its separate objective)')
    parser.add_argument('--gdn-bank', type=Path, help='Qwen3.5 GDN output-projection factors')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device-map', choices=('none', 'auto', 'balanced'), default='none')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True, help='result directory')


def generation_arguments(parser, *, routed_tokens):
    model_arguments(parser)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--method', choices=('full', 'basiskv'), default='full')
    parser.add_argument('--router', type=Path, help='core.fit_router fit output directory')
    parser.add_argument('--page-size', type=int, default=4, help='must match the router calibration')
    parser.add_argument('--pinned-pages', type=int, choices=(0, 1), default=0,
                        help='always-selected leading pages, counted in the page budget; must match the router calibration')
    parser.add_argument('--routed-tokens', type=int, default=routed_tokens, help='historical tokens; recent 64 are additional')
    parser.add_argument('--sequence-length', type=int, default=131072)
    parser.add_argument('--prefill-chunk-tokens', type=int, default=1024,
                        help='run the layer norms and MLPs this many tokens at a time during prefill (exact; bounds the 128K activation memory)')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--stage', choices=('evaluate', 'summarize'), default='evaluate')


def protocol(args, benchmark):
    settings = {key: value for key, value in vars(args).items()
                if key not in ('output', 'device', 'device_map', 'cpu_threads', 'shard_index', 'stage')}
    settings = json.loads(json.dumps(settings, default=str))
    fingerprints = {'model/config.json': sha256(args.model / 'config.json')}
    for name in ('config', 'gated_value_bank', 'gdn_bank'):
        value = getattr(args, name, None)
        if value is not None:
            path = value / 'config.json' if value.is_dir() else value
            fingerprints[name] = sha256(path)
    if args.value_factors is not None:
        fingerprints['value_factors/results.json'] = sha256(args.value_factors / 'results.json')
    if getattr(args, 'router', None) is not None:
        for path in sorted(args.router.glob('layer_*.*')):
            if path.suffix in ('.json', '.safetensors'):
                fingerprints[f'router/{path.name}'] = sha256(path)
    root = Path(__file__).resolve().parents[1]
    for name in ('eval/common.py', f'eval/{benchmark}.py', 'inference/model.py', 'inference/attention.py',
                 'inference/routing.py', 'inference/qwen35.py', 'core/router.py'):
        fingerprints[name] = sha256(root / name)
    return dict(benchmark=benchmark, quality_only=True, settings=settings, sha256=fingerprints,
                environment=dict(torch=torch.__version__, transformers=importlib.metadata.version('transformers')))


def load_model(args):
    from transformers import AutoConfig, AutoTokenizer
    from inference.model import install_factors, install_router, load_model as load_native_model

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.config or args.model, local_files_only=True)
    model = load_native_model(args.model, config=config, dtype=args.dtype, device=args.device, device_map=args.device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    route = getattr(args, 'method', 'full') == 'basiskv'
    assert route == (getattr(args, 'router', None) is not None), '--method basiskv requires --router; full does not use a router'
    install_factors(model, value_factors=args.value_factors,
        gated_value_bank=args.gated_value_bank, gdn_bank=args.gdn_bank)
    if route:
        install_router(model, args.router, page_size=args.page_size, routed_tokens=args.routed_tokens, pinned_pages=args.pinned_pages)
    if getattr(args, 'prefill_chunk_tokens', None):
        chunk_tokenwise_prefill(model, args.prefill_chunk_tokens)
    return model.eval().requires_grad_(False), tokenizer


def save_result(args, payload):
    value = dict(status='complete', **payload, command=shlex.join(sys.argv), python=sys.executable)
    path = args.output / 'result.json'
    if path.exists():
        saved = read_json(path)
        assert {key: item for key, item in saved.items() if key not in ('command', 'python')} == payload | {'status': 'complete'}, f'{path} holds a different result; use another --output'
    else:
        write_json(path, value)
    print(json.dumps(payload['metrics'], indent=2), flush=True)


@torch.inference_mode()
def greedy_generate(model, tokenizer, ids, maximum):
    input_ids = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    generated, cache = [], None
    for _ in range(maximum):
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        assert torch.isfinite(output.logits).all(), 'non-finite generation logits'
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        cache = output.past_key_values
        if token in eos:
            break
        input_ids = input_ids.new_tensor([[token]])
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated


def evaluate_generation(args, benchmark, samples, scorer, details):
    stamp = protocol(args, benchmark)
    stamp.update(details, routing_dtype='sidecar in the cache dtype, float32 scores')
    stamp['inputs_sha256'] = hashlib.sha256(json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if args.stage == 'evaluate':
        model, tokenizer = load_model(args)
        config = getattr(model.config, 'text_config', model.config)
        assert args.sequence_length <= config.max_position_embeddings, 'supply --config for an extended context'
        for row in samples[args.shard_index::args.num_shards]:
            path = args.output / 'samples' / f"{row['index']:06d}.json"
            if path.exists():
                assert read_json(path)['protocol'] == stamp, f'{path} was produced under another protocol'
                continue
            raw_ids = tokenizer(row['prompt'], add_special_tokens=benchmark == 'ruler')['input_ids']
            excluded = benchmark == 'longbench' and len(raw_ids) <= args.min_input_tokens
            record = dict(status='complete', protocol=stamp, index=row['index'], task=row['task'],
                          source_index=row['source_index'], raw_input_tokens=len(raw_ids), excluded=excluded)
            if not excluded:
                chat = args.chat_template == 'auto' and tokenizer.chat_template and row['task'] not in NO_CHAT_TASKS
                if chat:  # user turn, generation prompt with thinking off, then the task's answer prefix (RULER) as assistant text
                    text = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False) + row.get('answer_prefix', '')
                    ids = tokenizer(text, add_special_tokens=False)['input_ids']
                else:
                    ids = raw_ids if benchmark == 'ruler' else tokenizer(row['prompt'], add_special_tokens=True)['input_ids']
                assert len(ids) + row['max_new_tokens'] <= args.sequence_length, f"prompt {row['index']} exceeds the context"
                prediction, generated = greedy_generate(model, tokenizer, ids, row['max_new_tokens'])
                record.update(prediction=prediction, generated_token_ids=generated, input_tokens=len(ids),
                              score=scorer(row, prediction), prompt_format='chat' if chat else 'raw')
            write_json(path, record)
            print(f"[{benchmark}] sample={row['index']} task={row['task']} excluded={excluded} score={record.get('score')}", flush=True)
        del model
    if args.num_shards > 1 and args.stage != 'summarize':
        return
    records = [read_json(args.output / 'samples' / f"{row['index']:06d}.json") for row in samples]
    assert all(row['status'] == 'complete' and row['protocol'] == stamp for row in records), 'sample records are incomplete or from another protocol'
    included = [row for row in records if not row['excluded']]
    tasks = {}
    for name in dict.fromkeys(row['task'] for row in samples):
        scores = [row['score'] for row in included if row['task'] == name]
        tasks[name] = dict(samples=len(scores), score=100 * sum(scores) / len(scores))
    metrics = dict(tasks=tasks, average_score=sum(row['score'] for row in tasks.values()) / len(tasks),
                   samples=len(included), excluded=len(records) - len(included))
    save_result(args, dict(protocol=stamp, metrics=metrics))
