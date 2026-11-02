"""Table 6: structured-generation quality with the lm-evaluation-harness on the HF model.

Tasks and their paper settings (few-shot, generation cap): gsm8k 5-shot 512,
minerva_math500 4-shot 4096, mbpp_plus_full 0-shot 2048, ifeval 0-shot 1280.
Decoding is greedy. Qwen3 and Llama run as base completions; Qwen3.5 uses its
chat template with thinking disabled.

    python -m appendix.generation --model MODEL --value-factors FACTORS \\
        --tasks gsm8k,minerva_math500,mbpp_plus_full,ifeval --output OUT [--confirm-run-unsafe-code]
    python -m appendix.generation --model QWEN35 --gated-value-bank V_BANK --gdn-bank GDN_BANK \\
        --chat-template auto --thinking off --tasks gsm8k --output OUT

`mbpp_plus_full` executes generated code and needs `--confirm-run-unsafe-code`;
its task definition lives in `appendix/tasks/`.
"""

import argparse
from pathlib import Path

from eval.common import load_model, model_arguments, protocol, save_result

TASKS = {"gsm8k": (5, 512), "minerva_math500": (4, 4096), "mbpp_plus_full": (0, 2048), "ifeval": (0, 1280)}
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def task_metrics(results):
    """Every reported metric of the task, without standard errors."""
    return {key: float(value) for key, value in results.items()
            if "," in key and "stderr" not in key and isinstance(value, (int, float))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model_arguments(parser)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--limit", type=float, help="optional partial sample count or fraction")
    parser.add_argument("--chat-template", choices=("none", "auto"), default="none")
    parser.add_argument("--thinking", choices=("off", "on"), default="off", help="Qwen3.5 thinking mode under the chat template")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true", help="required by mbpp_plus_full")
    args = parser.parse_args(argv)
    tasks = args.tasks.split(",")
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    model, tokenizer = load_model(args)
    chat = args.chat_template == "auto" and tokenizer.chat_template is not None
    thinking = {"enable_thinking": args.thinking == "on", "think_end_token": "</think>" if args.thinking == "on" else None} if chat else {}
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=args.batch_size, max_length=args.max_length, **thinking)
    metrics = {}
    for task in tasks:
        shots, max_tokens = TASKS[task]
        evaluation = lm_eval.simple_evaluate(
            model=lm, tasks=[task], num_fewshot=shots, limit=args.limit, apply_chat_template=chat, fewshot_as_multiturn=False,
            gen_kwargs={"max_gen_toks": max_tokens, "do_sample": False, "temperature": 0.0},
            task_manager=TaskManager(include_path=str(TASKS_DIR)), confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed, fewshot_random_seed=args.seed)
        results = evaluation["results"].get(task) or evaluation["groups"][task]
        metrics[task] = dict(num_fewshot=shots, max_gen_toks=max_tokens, **task_metrics(results))
        print(f"[generation] {task}: {metrics[task]}", flush=True)
    stamp = protocol(args, "generation")
    stamp.update(chat_template=chat, thinking=args.thinking if chat else None, greedy=True,
                 tasks={task: dict(num_fewshot=TASKS[task][0], max_gen_toks=TASKS[task][1]) for task in tasks})
    save_result(args, dict(protocol=stamp, metrics=metrics))


if __name__ == "__main__":
    main()
