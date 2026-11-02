"""Seven zero-shot commonsense tasks using the lm-evaluation-harness."""

import argparse

from eval.common import load_model, model_arguments, protocol, save_result

TASKS = ('arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'winogrande', 'boolq', 'openbookqa')


def task_scores(results, tasks):
    scores = {}
    for task in tasks:
        values = results[task]
        key = 'acc_norm,none' if 'acc_norm,none' in values else 'acc,none'
        scores[task] = dict(metric=key, accuracy=float(values[key]))
    return dict(tasks=scores, average_accuracy=sum(row['accuracy'] for row in scores.values()) / len(scores))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--limit', type=float, help='optional partial sample count or fraction')
    args = parser.parse_args(argv)
    tasks = args.tasks.split(',')
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    model, tokenizer = load_model(args)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    result = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=args.limit,
        random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed, fewshot_random_seed=args.seed)
    stamp = protocol(args, 'mcq')
    stamp.update(num_fewshot=0, metric_selection='acc_norm when present, otherwise acc')
    save_result(args, dict(protocol=stamp, metrics=task_scores(result['results'], tasks)))


if __name__ == '__main__':
    main()
