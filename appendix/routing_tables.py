"""Tables 8 to 11: per-task scores and paired bootstrap intervals from eval.ruler / eval.longbench runs.

Every arm is the output directory of one `eval.ruler` or `eval.longbench` run on
the same prompts. Scores are read from the per-sample records, so the arms may
have been evaluated in any order or sharding.

    python -m appendix.routing_tables --arms full=RUN_FULL b16r16=RUN_B16R16 r16=RUN_R16 \\
        --reference b16r16 --output TABLE
"""

import argparse
import logging
import random
import sys
from pathlib import Path

from core.router_io import read_json, write_json

log = logging.getLogger("basiskv.routing_tables")


def load_arm(directory):
    """{(task, index): score} of the included samples, plus the inputs fingerprint."""
    records = [read_json(path) for path in sorted((Path(directory) / "samples").glob("*.json"))]
    fingerprints = {row["protocol"]["inputs_sha256"] for row in records}
    assert all(row["status"] == "complete" for row in records) and len(fingerprints) == 1, (
        f"samples under {directory} are incomplete or come from different prompt sets")
    scores = {(row["task"], row["index"]): float(row["score"]) for row in records if not row["excluded"]}
    return scores, fingerprints.pop()


def task_means(scores, tasks):
    return {task: 100 * sum(value for (name, _), value in scores.items() if name == task)
            / sum(1 for (name, _) in scores if name == task) for task in tasks}


def unweighted_mean(scores, tasks):
    means = task_means(scores, tasks)
    return sum(means.values()) / len(means)


def paired_bootstrap(arm, reference, tasks, iterations, seed):
    """Percentile interval of the unweighted-mean difference, resampling prompts within each task."""
    rng = random.Random(seed)
    keys = {task: [key for key in arm if key[0] == task] for task in tasks}
    differences = []
    for _ in range(iterations):
        sample = {task: [rng.choice(keys[task]) for _ in keys[task]] for task in tasks}
        difference = sum(100 * sum(arm[key] - reference[key] for key in sample[task]) / len(keys[task]) for task in tasks)
        differences.append(difference / len(tasks))
    differences.sort()
    return {"low": differences[int(0.025 * iterations)], "high": differences[int(0.975 * iterations) - 1]}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, metavar="NAME=DIR", help="eval.ruler or eval.longbench output directories")
    parser.add_argument("--reference", help="arm the paired differences are taken against")
    parser.add_argument("--tasks", help="comma list; default: every task of the first arm")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    arms, fingerprints = {}, set()
    for item in args.arms:
        name, _, directory = item.partition("=")
        arms[name], fingerprint = load_arm(directory)
        fingerprints.add(fingerprint)
    assert len(fingerprints) == 1, "arms were evaluated on different prompt sets"
    first = next(iter(arms.values()))
    tasks = args.tasks.split(",") if args.tasks else sorted({task for task, _ in first})
    keys = {key for key in first if key[0] in tasks}
    for name, scores in arms.items():
        assert {key for key in scores if key[0] in tasks} == keys, f"arm {name} covers other prompts"
    table = {name: {"tasks": task_means(scores, tasks), "mean": unweighted_mean(scores, tasks)} for name, scores in arms.items()}
    if args.reference:
        for name in arms:
            if name != args.reference:
                table[name]["difference_to_reference"] = table[name]["mean"] - table[args.reference]["mean"]
                table[name]["bootstrap_95"] = paired_bootstrap(arms[name], arms[args.reference], tasks, args.bootstrap, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "result.json", dict(tasks=tasks, prompts=len(keys), reference=args.reference,
                                                 bootstrap=args.bootstrap, arms=table))
    lines = ["| Arm | " + " | ".join(tasks) + " | Avg. |" + (" Δ vs ref [95% CI] |" if args.reference else ""),
             "|---|" + "---:|" * (len(tasks) + 1) + ("---|" if args.reference else "")]
    for name, row in table.items():
        cells = [f"{row['tasks'][task]:.2f}" for task in tasks] + [f"{row['mean']:.2f}"]
        if args.reference and "bootstrap_95" in row:
            ci = row["bootstrap_95"]
            cells.append(f"{row['difference_to_reference']:+.2f} [{ci['low']:+.2f}, {ci['high']:+.2f}]")
        elif args.reference:
            cells.append("reference")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
