"""Model dimensions and head grouping shared by Value fitting and inference."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ValueGeometry:
    model_type: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    attention_layers: tuple[int, ...]

    @property
    def attention_type(self):
        return "mha" if self.num_heads == self.num_kv_heads else "gqa"

    @property
    def heads_per_group(self):
        return self.num_heads // self.num_kv_heads

    def head_to_kv_group(self, device="cpu"):
        return torch.arange(self.num_heads, device=device) // self.heads_per_group

    def to_dict(self):
        return {**asdict(self), "attention_layers": list(self.attention_layers)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "attention_layers": tuple(value["attention_layers"])})

    @classmethod
    def from_manifest(cls, manifest):
        model = manifest["model"]
        geometry = cls(
            model_type=model["model_type"],
            num_layers=int(model["num_hidden_layers"]),
            num_heads=int(model["num_attention_heads"]),
            num_kv_heads=int(model["num_key_value_heads"]),
            head_dim=int(model["head_dim"]),
            hidden_size=int(model["hidden_size"]),
            attention_layers=tuple(map(int, manifest["layers"])),
        )
        return geometry

    @classmethod
    def from_config(cls, config, attention_layers):
        """Geometry of a loaded Hugging Face config; multimodal wrappers contribute their text config."""
        text = getattr(config, "text_config", config)
        kind = "qwen3_5" if text.model_type.startswith("qwen3_5") else text.model_type
        head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        return cls(kind, text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads, head_dim,
                   text.hidden_size, tuple(attention_layers))

    @classmethod
    def from_factors(cls, directory):
        records = sorted(Path(directory).glob("layer_*.json"))
        record = json.loads(records[0].read_text())
        return cls.from_dict(record["fit_config"]["geometry"])


@dataclass(frozen=True)
class GQAVOLayout:
    """Per-layer V/O widths at one rank, used by the inference-side adapters."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rank: int

    @property
    def query_heads_per_kv_group(self):
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def query_width(self):
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self):
        return self.num_key_value_heads * self.head_dim

    @property
    def compressed_query_width(self):
        return self.num_attention_heads * self.rank

    @property
    def compressed_kv_width(self):
        return self.num_key_value_heads * self.rank

    def query_head_indices(self, group_index):
        start = group_index * self.query_heads_per_kv_group
        return tuple(range(start, start + self.query_heads_per_kv_group))
