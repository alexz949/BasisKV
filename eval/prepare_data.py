"""Prepare official RULER-v1 and LongBench-v1 inputs for the evaluation entrypoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from core.router_io import sha256, write_json


def ruler(args):
    from transformers import AutoTokenizer
    from eval.ruler import RULER_TASKS
    source = args.ruler_root.resolve()
    model, output = args.model.resolve(), args.output.resolve()
    script = source / 'scripts/data/prepare.py'
    assert script.is_file(), str(script)
    tasks = list(RULER_TASKS) if args.tasks == 'all' else args.tasks.split(',')
    assert tasks and set(tasks) <= RULER_TASKS.keys()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    artifacts = {}
    for task in tasks:
        command = [sys.executable, str(script), '--save_dir', str(output), '--benchmark', 'synthetic',
                   '--task', task, '--subset', 'validation', '--tokenizer_path', str(model),
                   '--tokenizer_type', 'hf', '--max_seq_length', str(args.sequence_length),
                   '--num_samples', str(args.samples), '--random_seed', str(args.seed), '--model_template_type', 'base']
        subprocess.run(command, cwd=script.parent, env=environment, check=True)
        path = output / task / 'validation.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == args.samples, f'{task}: incomplete generation'
        lengths = [len(tokenizer(row['input'] + row.get('answer_prefix', ''), add_special_tokens=True)['input_ids']) for row in rows]
        assert max(lengths) + RULER_TASKS[task][0] <= args.sequence_length, f'{task}: total context budget exceeded'
        artifacts[task] = dict(sha256=sha256(path), rows=len(rows), maximum_prompt_tokens=max(lengths))
    write_json(output / 'manifest.json', dict(status='complete',
        protocol=dict(sequence_length=args.sequence_length, samples_per_task=args.samples, random_seed=args.seed,
                      tasks=tasks, model_template_type='base'),
        tokenizer_sha256=sha256(model / 'tokenizer.json'), prepare_script_sha256=sha256(script), artifacts=artifacts))


def longbench(args):
    from huggingface_hub import hf_hub_download
    revision = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
    archive = Path(hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset', revision=revision))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        files = [name for name in bundle.namelist() if name.endswith('.jsonl')]
        for name in files:
            # Only dataset JSONL files are copied; archive paths never become output paths.
            destination = output / Path(name).name
            assert not destination.exists(), str(destination)
            destination.write_bytes(bundle.read(name))
    write_json(output / 'manifest.json', dict(status='complete', dataset='THUDM/LongBench', revision=revision,
        archive_sha256=sha256(archive), files={Path(name).name: sha256(output / Path(name).name) for name in files}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='dataset', required=True)
    r = sub.add_parser('ruler')
    r.add_argument('--ruler-root', type=Path, required=True)
    r.add_argument('--model', type=Path, required=True)
    r.add_argument('--sequence-length', type=int, default=131072)
    r.add_argument('--samples', type=int, default=100)
    r.add_argument('--seed', type=int, default=42)
    r.add_argument('--tasks', default='all', help='generate all thirteen tasks, including the appendix subsets')
    r.add_argument('--output', type=Path, required=True)
    l = sub.add_parser('longbench')
    l.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (ruler if args.dataset == 'ruler' else longbench)(args)


if __name__ == '__main__':
    main()
