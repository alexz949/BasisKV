"""Main-paper RULER-128K quality: ten tasks, 100 examples per task, official base-completion protocol."""

import argparse
import json

from core.router_io import read_json
from eval.common import evaluate_generation, generation_arguments

# The official RULER v1 task table: name -> (tokens to generate, match type).
RULER_TASKS = {
    'niah_single_1': (128, 'all'), 'niah_single_2': (128, 'all'), 'niah_single_3': (128, 'all'),
    'niah_multikey_1': (128, 'all'), 'niah_multikey_2': (128, 'all'), 'niah_multikey_3': (128, 'all'),
    'niah_multivalue': (128, 'all'), 'niah_multiquery': (128, 'all'),
    'vt': (30, 'all'), 'cwe': (120, 'all'), 'fwe': (50, 'all'), 'qa_1': (32, 'part'), 'qa_2': (32, 'part'),
}
TASKS = ('niah_single_1', 'niah_single_2', 'niah_multikey_1', 'niah_multikey_2',
         'niah_multiquery', 'niah_multivalue', 'qa_1', 'qa_2', 'vt', 'fwe')


def sample_score(prediction, references, match_type):
    """Official RULER metric: case-insensitive substring matches, averaged ('all') or any ('part')."""
    normalized = prediction.strip().lower()
    matches = [reference.lower() in normalized for reference in references]
    return sum(matches) / len(matches) if match_type == 'all' else float(any(matches))


def read_samples(args):
    manifest = read_json(args.data / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['protocol']['sequence_length'] == args.sequence_length, (
        f'{args.data} is incomplete or was generated for another sequence length')
    samples = []
    for name in (RULER_TASKS if args.tasks == 'all' else args.tasks.split(',')):
        tokens, match_type = RULER_TASKS[name]
        path = args.data / name / 'validation.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(records) >= args.samples_per_task, f'{name} has {len(records)} samples, fewer than --samples-per-task'
        for ordinal, record in enumerate(records[:args.samples_per_task]):
            row = dict(index=len(samples), task=name, source_index=ordinal, answers=record['outputs'], match_type=match_type, max_new_tokens=tokens)
            if args.chat_template == 'auto':  # the answer prefix follows the generation prompt as assistant text
                row.update(prompt=record['input'], answer_prefix=record.get('answer_prefix', ''))
            else:
                row.update(prompt=record['input'] + record.get('answer_prefix', ''))
            samples.append(row)
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    generation_arguments(parser, routed_tokens=2048)
    parser.add_argument('--tasks', default=','.join(TASKS), help="comma list or 'all' (the thirteen official tasks)")
    parser.add_argument('--samples-per-task', type=int, default=100)
    parser.add_argument('--chat-template', choices=('none', 'auto'), default='none',
                        help='auto: chat-tuned models (Qwen3-8B in Table 2); the input is the user turn and the answer prefix follows the generation prompt')
    args = parser.parse_args(argv)
    samples = read_samples(args)
    evaluate_generation(args, 'ruler', samples,
        lambda row, prediction: sample_score(prediction, row['answers'], row['match_type']),
        dict(prompt_format='chat template with answer prefix' if args.chat_template == 'auto' else 'official RULER base completion', task_mean='unweighted',
             samples_per_task=args.samples_per_task))


if __name__ == '__main__':
    main()
