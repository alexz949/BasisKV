"""Shared pieces of the calibration stages: process setup, the window manifest, and the teacher model."""

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from core.factor_io import parse_layers
from core.model_layout import ValueGeometry
from core.router_io import read_json
from inference.tp import chunk_tokenwise_prefill
from inference.model import attention_layers, install_factors, load_model


def setup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False


def add_model_arguments(parser):
    parser.add_argument("--model", type=Path, required=True, help="local Hugging Face model directory")
    parser.add_argument("--config", type=Path, help="HF config for an explicitly extended context, such as YaRN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("none", "balanced", "balanced_low_0"), default="none")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024, help="run the layer norms and MLPs this many tokens at a time during prefill; exact, bounds the 128K activation memory")


def add_window_arguments(parser):
    parser.add_argument("--windows", type=Path, required=True, help="windows.safetensors with its manifest.json")
    parser.add_argument("--sequence-length", type=int, help="leading tokens per window; default: the stored length")
    parser.add_argument("--fit-windows", type=int, help="leading fit windows; default: all")
    parser.add_argument("--heldout-windows", type=int, help="leading held-out windows; default: all")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--window-shard-index", type=int, default=0)
    parser.add_argument("--window-shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="combine the window shards written earlier")


def read_windows(args):
    """Token windows cut to --sequence-length, and the fit / held-out window ids."""
    manifest = read_json(args.windows.with_name("manifest.json"))
    assert manifest["status"] == "complete", "windows manifest is incomplete"
    windows = load_file(str(args.windows))["input_ids"]
    args.sequence_length = args.sequence_length or windows.shape[1]
    assert 0 < args.sequence_length <= windows.shape[1], "--sequence-length exceeds the stored windows"
    split_ids = {}
    for split, field, count in (("fit", "fit_ids", args.fit_windows), ("heldout", "validation_ids", args.heldout_windows)):
        available = list(map(int, manifest[field]))
        split_ids[split] = available if count is None else available[:count]
    assert split_ids["fit"] and not set(split_ids["fit"]) & set(split_ids["heldout"]), "fit and held-out windows must be disjoint"
    return windows[:, : args.sequence_length], split_ids


def load_teacher(args, **banks):
    """The model with the given Value banks installed, its text config, geometry, the selected
    attention modules, and the dense v_proj weights taken before installation."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.config or args.model, local_files_only=True)
    text = getattr(config, "text_config", config)
    assert args.sequence_length <= text.max_position_embeddings, "windows exceed the model context; pass --config for an extended one"
    model = load_model(args.model, config=config, dtype=args.model_dtype, device=args.device, device_map=args.device_map)
    if args.prefill_chunk_tokens:
        chunk_tokenwise_prefill(model, args.prefill_chunk_tokens)
    geometry = ValueGeometry.from_config(config, sorted(attention_layers(model)))
    layers = parse_layers(args.layers, geometry)
    dense = {index: (module.v_proj.weight.detach().clone(), None if module.v_proj.bias is None else module.v_proj.bias.detach().clone())
             for index, module in attention_layers(model).items() if index in layers}
    install_factors(model, **banks)
    attention = attention_layers(model)
    return model.eval(), text, geometry, {index: attention[index] for index in layers}, dense
