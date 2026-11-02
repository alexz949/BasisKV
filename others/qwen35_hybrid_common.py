"""Shared immutable artifacts and model loading for the Qwen3.5 hybrid run."""

import json
import os
from pathlib import Path
import tempfile

import torch
from inference.qwen35 import load_bank, verify_model_identity
from core.router_io import sha256


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Existing artifact: {path}'
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        if path.suffix == '.json':
            stream.write(json.dumps(value, indent=2, allow_nan=False).encode())
        else:
            torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def load_model(path, device):
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    return model


def full_layers(model):
    return {i: layer.self_attn for i, layer in enumerate(model.model.layers) if hasattr(layer, 'self_attn')}


def load_windows(root, split):
    payload = torch.load(Path(root) / 'windows.pt', weights_only=True, map_location='cpu')
    return payload[split].long()


