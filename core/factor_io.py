"""Write, verify, merge, and assemble fitted Value factors.

Each layer produces `layer_NNN.safetensors` holding `value_coordinate_encoders`
[G, d, r] and `head_output_decoders` [H, r, o], plus `layer_NNN.json` with the
fit configuration, losses, and solver diagnostics. `merge` collects the layers
of one uniform-rank fit into `results.json` and `summary.md`; `assemble` builds
the same layout from layers of several uniform fits under a per-layer rank
schedule.
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from safetensors.torch import save_file

from core.model_layout import ValueGeometry

FORMAT = "basiskv.value_factors.v1"
LAYER_FORMAT = "basiskv.value_factors.layer.v1"
ENCODER_KEY = "value_coordinate_encoders"
DECODER_KEY = "head_output_decoders"
# The fit_config entry that may differ between the layers of an allocated bank.
RANK_KEYS = ("cache_rank_per_head",)
log = logging.getLogger("basiskv.factor_io")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_safetensors(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def parse_layers(raw, geometry):
    """'all', or comma-separated indices and inclusive ranges such as '0-3,7'."""
    if raw.strip().lower() == "all":
        return geometry.attention_layers
    layers = set()
    for piece in filter(None, (piece.strip() for piece in raw.split(","))):
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            layers.update(range(first, last + 1))
        else:
            layers.add(int(piece))
    assert layers and layers <= set(geometry.attention_layers), f"no layers or layers outside {geometry.model_type}: {raw}"
    return tuple(sorted(layers))


def layer_paths(output_dir, layer):
    return output_dir / f"layer_{layer:03d}.json", output_dir / f"layer_{layer:03d}.safetensors"


def write_layer(output_dir, layer, encoder, decoder, record):
    """Write the factor artifact, then the record that describes it."""
    record_path, artifact_path = layer_paths(output_dir, layer)
    tensors = {ENCODER_KEY: encoder.contiguous(), DECODER_KEY: decoder.contiguous()}
    write_safetensors(artifact_path, tensors)
    complete = {
        "format": LAYER_FORMAT,
        "layer": layer,
        **record,
        "artifact": {
            "file": artifact_path.name,
            "sha256": sha256(artifact_path),
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in tensors.items()
            },
        },
    }
    write_json(record_path, complete)
    return complete


def read_layer(output_dir, layer, fit_config):
    """Verified layer record, or None when its output is missing or its checksum differs."""
    record_path, artifact_path = layer_paths(output_dir, layer)
    if not record_path.is_file() or not artifact_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record.get("format") == LAYER_FORMAT and record.get("layer") == layer, (
        f"incompatible layer record: {record_path}"
    )
    assert fit_config is None or record["fit_config"] == fit_config, (
        f"layer record was fitted with a different configuration: {record_path}"
    )
    if sha256(artifact_path) != record["artifact"]["sha256"]:
        log.warning("layer %d: factor checksum mismatch; refitting required", layer)
        return None
    return record


def cg_summary(updates, max_iterations):
    return {
        "max_iterations": max_iterations,
        "group_solves": len(updates),
        "rejected": sum(not step["accepted"] for step in updates),
        "total_iterations": sum(step["iterations"] for step in updates),
        "maximum_relative_residual": max((step["relative_residual"] for step in updates), default=0.0),
        "updates": list(updates),
    }


def shared_config(config):
    return {key: value for key, value in config.items() if key not in RANK_KEYS}


def _summary(records, layer_ranks, geometry):
    config = records[0]["fit_config"]
    mean_rank = sum(layer_ranks) / len(layer_ranks)
    heldout = [float(row["heldout"]["factor_dtype_relative_mse"]) for row in records]
    retained = (geometry.head_dim + mean_rank) / (2 * geometry.head_dim)
    uniform = len(set(layer_ranks)) == 1
    title = f"V{layer_ranks[0]}" if uniform else f"allocated mean V{mean_rank:g}"
    lines = [
        f"# {geometry.model_type} {geometry.attention_type.upper()} {title} joint fit",
        "",
        f"- K remains dense; each of {geometry.num_kv_heads} KV heads retains a mean V rank of "
        f"{mean_rank:g}/{geometry.head_dim}.",
        f"- Total KV-cache retention is {100 * retained:.6g}% ({100 * (1 - retained):.6g}% reduction).",
        f"- Fit windows: {config['fit_windows']}; held-out windows: {config['heldout_windows']}.",
        f"- Encoder sweeps: {config['encoder_sweeps']}; final decoder refitted.",
        f"- Mean held-out factor-dtype relative MSE: `{sum(heldout) / len(heldout):.9g}`.",
        "",
        "| Layer | Rank | Solver | Held-out relative MSE |",
        "|---:|---:|:---|---:|",
        *(
            f"| {row['layer']} | {rank} | {row['solver']['method']} | "
            f"{row['heldout']['factor_dtype_relative_mse']:.9g} |"
            for row, rank in zip(records, layer_ranks)
        ),
        "",
    ]
    return "\n".join(lines)


def _write_results(output_dir, layers, fit_config, layer_ranks, records, geometry, **extra):
    mean = lambda key: sum(float(row[key]["factor_dtype_relative_mse"]) for row in records) / len(records)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layers": list(layers),
        "fit_config": fit_config,
        "layer_ranks": list(layer_ranks),
        **extra,
        "artifacts": {str(row["layer"]): row["artifact"] for row in records},
        "records": records,
        "aggregate": {
            "mean_fit_factor_dtype_relative_mse": mean("fit"),
            "mean_heldout_factor_dtype_relative_mse": mean("heldout"),
        },
    }
    write_json(output_dir / "results.json", payload)
    (output_dir / "summary.md").write_text(_summary(records, layer_ranks, geometry), encoding="utf-8")
    return payload


def merge(output_dir, layers, geometry):
    assert not (output_dir / "results.json").exists(), f"already merged: {output_dir}"
    records = []
    for layer in layers:
        record = read_layer(output_dir, layer, records[0]["fit_config"] if records else None)
        assert record is not None, f"missing layer {layer} output"
        records.append(record)
    config = records[0]["fit_config"]
    ranks = [config["cache_rank_per_head"]] * len(records)
    return _write_results(output_dir, layers, config, ranks, records, geometry)


def read_results(directory):
    results = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
    assert results["format"] == FORMAT and results["status"] == "complete", f"incomplete factors: {directory}"
    return results


def assemble(output_dir, banks, layer_ranks, allocation):
    """Copy layer l from the uniform bank of rank layer_ranks[l]; record the allocation."""
    assert not (output_dir / "results.json").exists(), f"already assembled: {output_dir}"
    results = {rank: read_results(directory) for rank, directory in banks.items()}
    config = shared_config(next(iter(results.values()))["fit_config"])
    layers = next(iter(results.values()))["layers"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for layer, rank in zip(layers, layer_ranks):
        source = Path(banks[rank])
        record = read_layer(source, layer, results[rank]["fit_config"])
        assert record is not None, f"layer {layer} of {source} is missing or corrupt"
        for path in layer_paths(source, layer):
            shutil.copy2(path, output_dir / path.name)
        records.append(record)
    geometry = ValueGeometry.from_dict(config["geometry"])
    return _write_results(output_dir, layers, config, layer_ranks, records, geometry, allocation=allocation)
