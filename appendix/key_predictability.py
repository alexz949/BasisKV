"""Figure 6: how much of the pre-RoPE Key is affinely predictable from the Value.

The compressed model (Value factors installed) is replayed on calibration
windows. In every attention layer the dense Value v = W_v h and the pre-RoPE
Key k = W_k h (after the Key norm where the model has one) are accumulated as
FP64 sums and Grams of z = [v | k], separately for the fit and the held-out
windows of the window manifest. For every layer and KV group, four affine
predictors of K_g are fitted on the fit statistics and scored on the held-out
windows by centered explained energy 1 - ||K - K_hat||^2 / ||K - mean||^2:
local compressed V (v_g E_g), all-group compressed V, local raw V, all raw V.

    python -m appendix.key_predictability --model MODEL --value-factors FACTORS \\
        --windows WINDOWS/windows.safetensors --output OUT
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from core.factor_io import ENCODER_KEY, read_layer
from core.router_io import read_json, write_json
from inference.model import attention_layers, install_factors, load_model

log = logging.getLogger("basiskv.key_predictability")
VARIANTS = {"local_compressed_v": "local compressed V", "all_compressed_v": "all-group compressed V",
            "local_raw_v": "local raw V", "all_raw_v": "all-group raw V"}
CHUNK = 16384


@torch.inference_mode()
def accumulate(model, dense, windows, plan, device):
    """FP64 sums and Grams of [v | k] per (split, layer) over the planned windows."""
    layers = attention_layers(model)
    first = next(iter(layers.values()))
    groups, dim = first.groups, first.head_dim
    width = 2 * groups * dim
    stats = {(split, layer): {"gram": torch.zeros(width, width, dtype=torch.float64, device=device),
                              "sum": torch.zeros(width, dtype=torch.float64, device=device), "count": 0}
             for split in ("fit", "heldout") for layer in layers}
    active = {}

    def capture(module, positional, kwargs, layer):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else positional[0]
        n = hidden.shape[1]
        k = module.k_norm(module.k_proj(hidden).view(1, n, groups, dim))
        v = F.linear(hidden, *dense[layer]).view(1, n, groups, dim)
        z = torch.cat((v.reshape(n, groups * dim), k.reshape(n, groups * dim)), -1)
        entry = stats[(active["split"], layer)]
        for start in range(0, n, CHUNK):
            chunk = z[start : start + CHUNK].to(device, torch.float32)
            entry["gram"] += (chunk.mT @ chunk).double()
            entry["sum"] += chunk.sum(0).double()
        entry["count"] += n

    handles = [module.register_forward_pre_hook(lambda m, a, kw, layer=layer: capture(m, a, kw, layer), with_kwargs=True)
               for layer, module in layers.items()]
    embeddings = model.get_input_embeddings().weight.device
    for split, index in plan:
        active["split"] = split
        started = time.monotonic()
        output = model.model(input_ids=windows[index : index + 1].to(embeddings).long(), use_cache=False)
        assert torch.isfinite(output.last_hidden_state).all(), "non-finite replay"
        del output
        log.info("%s window %d: %.1fs", split, index, time.monotonic() - started)
    for handle in handles:
        handle.remove()
    return {key: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in entry.items()} for key, entry in stats.items()}


def explained_energy(fit, heldout, feature_map, key_slice, ridge):
    """Affine predictor k[key_slice] = W^T x + b with x = M v, fitted on `fit`; explained energy on both splits."""
    V = feature_map.shape[1]

    def blocks(entry):
        gram, total = entry["gram"], entry["sum"]
        return (feature_map @ gram[:V, :V] @ feature_map.mT, feature_map @ gram[:V, key_slice], gram[key_slice, key_slice],
                feature_map @ total[:V], total[key_slice], entry["count"])

    xx, xk, kk, sx, sk, n = blocks(fit)
    centered_xx = xx - torch.outer(sx, sx) / n
    centered_xx += ridge * centered_xx.diagonal().mean() * torch.eye(centered_xx.shape[0], dtype=centered_xx.dtype)
    weight = torch.linalg.solve(centered_xx, xk - torch.outer(sx, sk) / n)
    bias = sk / n - weight.mT @ (sx / n)

    def score(entry):
        xx, xk, kk, sx, sk, n = blocks(entry)
        sse = (kk.trace() - 2 * (weight.mT @ xk).trace() - 2 * bias @ sk + (weight.mT @ xx @ weight).trace()
               + 2 * bias @ (weight.mT @ sx) + n * bias @ bias)
        return float(1 - sse / (kk.trace() - sk @ sk / n))

    return {"fit": score(fit), "heldout": score(heldout)}


def solve(stats, encoders, groups, dim, ridge):
    V = groups * dim
    per_layer = {}
    for layer, encoder in encoders.items():
        encoder = encoder.double()
        rank = encoder.shape[-1]
        all_compressed = torch.zeros(groups * rank, V, dtype=torch.float64)
        for g in range(groups):
            all_compressed[g * rank : (g + 1) * rank, g * dim : (g + 1) * dim] = encoder[g].mT
        rows = {}
        for g in range(groups):
            local_raw = torch.zeros(dim, V, dtype=torch.float64)
            local_raw[:, g * dim : (g + 1) * dim] = torch.eye(dim, dtype=torch.float64)
            maps = {"local_compressed_v": all_compressed[g * rank : (g + 1) * rank], "all_compressed_v": all_compressed,
                    "local_raw_v": local_raw, "all_raw_v": torch.eye(V, dtype=torch.float64)}
            key_slice = slice(V + g * dim, V + (g + 1) * dim)
            rows[g] = {name: explained_energy(stats[("fit", layer)], stats[("heldout", layer)], m, key_slice, ridge)
                       for name, m in maps.items()}
        per_layer[layer] = {
            "rank": rank,
            "groups": rows,
            "mean": {name: {split: sum(rows[g][name][split] for g in rows) / groups for split in ("fit", "heldout")}
                     for name in VARIANTS},
        }
        log.info("layer %d heldout: %s", layer, {name: round(per_layer[layer]["mean"][name]["heldout"], 4) for name in VARIANTS})
    return per_layer


def summary(per_layer):
    layers = list(per_layer)
    lines = ["# Key predictability from Value (held-out centered explained energy, %)", "",
             "| layer | " + " | ".join(VARIANTS.values()) + " |", "|---:|" + "---:|" * len(VARIANTS)]
    for layer in layers:
        lines.append(f"| {layer} | " + " | ".join(f"{100 * per_layer[layer]['mean'][name]['heldout']:.2f}" for name in VARIANTS) + " |")
    lines.append("| mean | " + " | ".join(
        f"{100 * sum(per_layer[l]['mean'][name]['heldout'] for l in layers) / len(layers):.2f}" for name in VARIANTS) + " |")
    return "\n".join(lines) + "\n"


def plot(output, per_layer):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib is not installed; skipping the figure")
        return
    layers = list(per_layer)
    styles = {"local_compressed_v": ("tab:blue", "-"), "all_compressed_v": ("tab:blue", "--"),
              "local_raw_v": ("tab:gray", "-"), "all_raw_v": ("tab:gray", "--")}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, label in VARIANTS.items():
        color, style = styles[name]
        ax.plot(layers, [100 * per_layer[l]["mean"][name]["heldout"] for l in layers], color=color, linestyle=style,
                marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel("layer"); ax.set_ylabel("explained K energy (%)"); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "key_predictability.pdf"); plt.close(fig)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--value-factors", type=Path, required=True, help="merged core.fit_value output")
    parser.add_argument("--windows", type=Path, required=True, help="calibration windows with an adjacent manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="HF config for an explicitly extended context")
    parser.add_argument("--sequence-length", type=int, help="leading tokens per window; default: the stored length")
    parser.add_argument("--limit-windows", type=int, help="leading fit and held-out windows to use")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("none", "auto", "balanced"), default="none")
    args = parser.parse_args(argv)
    from transformers import AutoConfig

    manifest = read_json(args.windows.with_name("manifest.json"))
    assert manifest["status"] == "complete", "incomplete window manifest"
    windows = load_file(str(args.windows))["input_ids"][:, : args.sequence_length or None]
    fit_ids, heldout_ids = manifest["fit_ids"][: args.limit_windows], manifest["validation_ids"][: args.limit_windows]
    assert fit_ids and heldout_ids, "key predictability needs fit and held-out windows"
    config = AutoConfig.from_pretrained(args.config or args.model, local_files_only=True)
    model = load_model(args.model, config=config, dtype=args.dtype, device=args.device, device_map=args.device_map)
    dense = {layer: (module.v_proj.weight.detach().clone(), None if module.v_proj.bias is None else module.v_proj.bias.detach().clone())
             for layer, module in attention_layers(model).items()}
    install_factors(model, value_factors=args.value_factors)
    layers = attention_layers(model)
    encoders = {layer: load_file(str(args.value_factors / read_layer(args.value_factors, layer, None)["artifact"]["file"]))[ENCODER_KEY]
                for layer in layers}
    first = next(iter(layers.values()))
    device = torch.device(args.device if args.device_map == "none" else "cpu")
    stats = accumulate(model, dense, windows, [("fit", i) for i in fit_ids] + [("heldout", i) for i in heldout_ids], device)
    per_layer = solve(stats, encoders, first.groups, first.head_dim, args.ridge)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "result.json", dict(
        model=str(args.model), value_factors=str(args.value_factors), windows=str(args.windows),
        fit_ids=fit_ids, heldout_ids=heldout_ids, sequence_length=int(windows.shape[1]), ridge=args.ridge,
        variants=VARIANTS, per_layer={str(layer): row for layer, row in per_layer.items()}))
    (args.output / "summary.md").write_text(summary(per_layer))
    plot(args.output, per_layer)
    print(summary(per_layer))


if __name__ == "__main__":
    main()
