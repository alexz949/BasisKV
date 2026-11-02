"""Official LongBench-v1 nine-task quality on rendered inputs longer than 4K."""

import argparse
import json

from eval.common import evaluate_generation, generation_arguments

TASKS = ('narrativeqa', 'multifieldqa_en', 'hotpotqa', 'musique', 'dureader',
         'gov_report', 'samsum', 'passage_retrieval_en', 'lcc')


def read_samples(args):
    from eval.longbench_metrics import OFFICIAL_MAXLEN, OFFICIAL_PROMPT

    tasks = args.tasks.split(',')
    samples = []
    for task in tasks:
        records = [json.loads(line) for line in (args.data / f'{task}.jsonl').read_text().splitlines() if line.strip()]
        if args.limit_per_task is not None:
            records = records[:args.limit_per_task]
        for index, row in enumerate(records):
            samples.append(dict(index=len(samples), task=task, source_index=index, source_id=row.get('_id'),
                prompt=OFFICIAL_PROMPT[task].format(**row), answers=row['answers'],
                all_classes=row.get('all_classes'), max_new_tokens=OFFICIAL_MAXLEN[task]))
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    generation_arguments(parser, routed_tokens=256)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--min-input-tokens', type=int, default=4096, help='strict lower bound on the rendered raw prompt')
    parser.add_argument('--chat-template', choices=('auto', 'none'), default='auto')
    parser.add_argument('--limit-per-task', type=int, help='optional partial evaluation')
    args = parser.parse_args(argv)
    from eval.longbench_metrics import LONGBENCH_REPO_REVISION, longbench_score

    evaluate_generation(args, 'longbench', read_samples(args),
        lambda row, prediction: longbench_score(row['task'], prediction, row['answers'], row['all_classes']),
        dict(scorer_revision=LONGBENCH_REPO_REVISION, task_mean='unweighted',
             filter=f'official prompt tokens > {args.min_input_tokens}; evaluated tokenizer; no truncation'))


if __name__ == '__main__':
    main()
