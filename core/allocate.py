"""Layerwise Value rank allocation under a global budget (Section 3.3, Appendix A.2).

Each layer's reconstruction-error curve e_l(r) is read from uniform-rank factor
banks. Two measured single-layer probes per layer (calibration.probe_kl) give
the one-sided sensitivities

    s-_l = max(0, D-_l / (e_l(r_low) - e_l(r0))),   s+_l = max(0, D+_l / (e_l(r_high) - e_l(r0))),

where D±_l is the output-KL change of moving layer l alone from the anchor rank
r0 to the probe rank. The predicted KL change of rank r at layer l is
s-_l (e_l(r) - e_l(r0)) below the anchor and s+_l (e_l(r) - e_l(r0)) above it.
Exact-budget dynamic programming picks one rank per layer minimizing the summed
prediction, and the chosen layers are assembled from the uniform banks.

    python -m core.allocate --banks 32=B32 48=B48 64=B64 80=B80 96=B96 112=B112 128=B128 \\
        --probes PROBES/probes.json --target-average-rank 96 --output FACTORS
"""

import argparse
import logging
import math
import sys
from pathlib import Path

from core.factor_io import assemble, read_results
from core.router_io import read_json

log = logging.getLogger("basiskv.allocate")
PROBES_FORMAT = "basiskv.rank_probes.v1"


def parse_banks(items):
    """'RANK=DIR' pairs to {rank: Path}."""
    banks = {}
    for item in items:
        rank, _, directory = item.partition("=")
        banks[int(rank)] = Path(directory).expanduser().resolve()
    return dict(sorted(banks.items()))


def error_curves(banks):
    """Held-out relative reconstruction error of every layer at every bank rank."""
    results = {rank: read_results(directory) for rank, directory in banks.items()}
    layers = next(iter(results.values()))["layers"]
    curves = {
        layer: {rank: float(results[rank]["records"][index]["heldout"]["factor_dtype_relative_mse"]) for rank in banks}
        for index, layer in enumerate(layers)
    }
    return curves


def sensitivities(curves, probes, *, anchor, low, high):
    """(s-, s+) per layer from the measured probe deltas and the error curves."""
    result = {}
    for layer, probe in probes.items():
        errors = curves[layer]
        below, above = errors[low] - errors[anchor], errors[high] - errors[anchor]
        assert below > 0 > above, f"layer {layer}: error curve is not decreasing across the probes"
        result[layer] = (max(0.0, probe["delta_low"] / below), max(0.0, probe["delta_high"] / above))
    return result


def predicted_costs(curves, slopes, anchor):
    """Predicted KL change of every candidate rank, zero at the anchor."""
    costs = {}
    for layer, errors in curves.items():
        minus, plus = slopes[layer]
        costs[layer] = {
            rank: (minus if rank < anchor else plus) * (error - errors[anchor]) if rank != anchor else 0.0
            for rank, error in errors.items()
        }
    return costs


def allocate(costs, budget):
    """One rank per layer with ranks summing exactly to `budget` and minimal total predicted cost."""
    layers = list(costs)
    ranks = sorted(next(iter(costs.values())))
    step = math.gcd(budget, *ranks)
    size = budget // step + 1
    best = [math.inf] * size  # best[b]: minimal cost of the layers so far with rank sum b * step
    best[0] = 0.0
    choices = []
    for layer in layers:
        following, choice = [math.inf] * size, [None] * size
        for total in range(size):
            for rank in ranks:  # ascending, so ties keep the smaller rank
                previous = total - rank // step
                if previous >= 0 and best[previous] + costs[layer][rank] < following[total]:
                    following[total], choice[total] = best[previous] + costs[layer][rank], rank
        best = following
        choices.append(choice)
    assert math.isfinite(best[-1]), f"rank budget {budget} is not reachable with ranks {ranks}"
    schedule, total = {}, budget // step
    for layer, choice in zip(reversed(layers), reversed(choices)):
        schedule[layer] = choice[total]
        total -= choice[total] // step
    return {layer: schedule[layer] for layer in layers}, best[-1]


def read_probes(path):
    probes = read_json(path)
    assert probes["format"] == PROBES_FORMAT, f"{path} is not a rank-probe record"
    anchor, low, high = probes["anchor_rank"], probes["low_rank"], probes["high_rank"]
    layers = {int(row["layer"]): row for row in probes["layers"]}
    return layers, anchor, low, high


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--banks", nargs="+", required=True, metavar="RANK=DIR", help="merged uniform core.fit_value outputs")
    parser.add_argument("--probes", type=Path, required=True, help="probes.json from calibration.probe_kl")
    parser.add_argument("--target-average-rank", type=int, required=True, help="mean Value rank per layer")
    parser.add_argument("--output", type=Path, required=True, help="assembled factor directory")
    args = parser.parse_args(argv)

    banks = parse_banks(args.banks)
    curves = error_curves(banks)
    probes, anchor, low, high = read_probes(args.probes)
    assert set(probes) == set(curves), "probes do not cover the fitted layers"
    slopes = sensitivities(curves, probes, anchor=anchor, low=low, high=high)
    costs = predicted_costs(curves, slopes, anchor)
    schedule, predicted = allocate(costs, args.target_average_rank * len(curves))
    layer_ranks = [schedule[layer] for layer in curves]
    log.info("layer ranks: %s", layer_ranks)
    log.info("mean rank %.3f, predicted KL change %.6g", sum(layer_ranks) / len(layer_ranks), predicted)
    allocation = {
        "method": "two_sided_kl_dynamic_programming",
        "anchor_rank": anchor,
        "low_rank": low,
        "high_rank": high,
        "target_average_rank": args.target_average_rank,
        "candidate_ranks": list(banks),
        "probes": str(args.probes.resolve()),
        "anchor_kl": probes[next(iter(probes))].get("anchor_kl"),
        "sensitivities": {str(layer): list(slopes[layer]) for layer in curves},
        "predicted_costs": {str(layer): {str(rank): cost for rank, cost in costs[layer].items()} for layer in curves},
        "predicted_kl_change": predicted,
    }
    assemble(args.output.expanduser().resolve(), banks, layer_ranks, allocation)
    log.info("assembled %d layers into %s", len(layer_ranks), args.output)


if __name__ == "__main__":
    main()
