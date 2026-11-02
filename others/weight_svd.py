"""Ordinary per-head weight-SVD reference using PyTorch layers."""

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from others.quality_support import _decoder_layers


@torch.no_grad()
def factorize_weight_svd_projection(
    weight: Tensor,
    *,
    ranks: Sequence[int],
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    assert ranks and len(set(ranks)) == 1
    assert weight.shape[0] % len(ranks) == 0
    groups = (
        weight.detach()
        .cpu()
        .float()
        .reshape(len(ranks), weight.shape[0] // len(ranks), weight.shape[1])
    )
    writers, decoders = [], []
    for block, rank in zip(groups, ranks):
        assert 0 < rank <= min(block.shape)
        left, singular, right = torch.linalg.svd(block, full_matrices=False)
        scale = singular[:rank].sqrt()
        writers.append((scale[:, None] * right[:rank]).to(output_dtype))
        decoders.append((left[:, :rank] * scale[None, :]).to(output_dtype))
    writer = torch.cat(writers).contiguous()
    decoder = torch.stack(decoders).contiguous()
    reconstructed = (
        decoder.double()
        @ writer.reshape(len(ranks), ranks[0], weight.shape[1]).double()
    )
    reference = weight.detach().cpu().double().reshape_as(reconstructed)
    error = (reconstructed - reference).square().sum() / reference.square().sum()
    return writer, decoder, {"relative_frobenius_error": float(error.sqrt())}


class HeadwiseSVDProjection(nn.Module):
    """Apply a shared writer followed by one decoder per physical V head."""

    def __init__(self, writer: Tensor, decoder: Tensor):
        super().__init__()
        heads, width, rank = decoder.shape
        assert writer.shape[0] == heads * rank
        self.rank = rank
        self.writer = nn.Linear(
            writer.shape[1],
            writer.shape[0],
            bias=False,
            device=writer.device,
            dtype=writer.dtype,
        )
        self.decoders = nn.ModuleList(
            [
                nn.Linear(
                    rank, width, bias=False, device=writer.device, dtype=writer.dtype
                )
                for _ in range(heads)
            ]
        )
        with torch.no_grad():
            self.writer.weight.copy_(writer)
            for module, block in zip(self.decoders, decoder):
                module.weight.copy_(block)

    def forward(self, hidden: Tensor) -> Tensor:
        pieces = self.writer(hidden).split(self.rank, dim=-1)
        return torch.cat(
            [module(piece) for module, piece in zip(self.decoders, pieces)], dim=-1
        )


@torch.no_grad()
def install_weight_svd_factors(
    model: nn.Module,
    payload: Mapping[str, Tensor],
    *,
    layer_ranks: Sequence[Sequence[int]],
    head_dim: int,
    require_cuda_resident: bool = False,
) -> list[dict[str, Any]]:
    layers = _decoder_layers(model)
    assert len(layers) == len(layer_ranks)
    records = []
    for index, (layer, ranks) in enumerate(zip(layers, layer_ranks)):
        dense = layer.self_attn.v_proj
        assert isinstance(dense, nn.Linear) and dense.bias is None
        assert ranks and len(set(ranks)) == 1 and min(ranks) > 0
        assert not require_cuda_resident or dense.weight.device.type == "cuda"
        writer = payload[f"layers.{index}.v_writer.weight"]
        decoder = payload[f"layers.{index}.v_decoder.weight"]
        assert writer.shape == (sum(ranks), dense.in_features)
        assert decoder.shape == (len(ranks), head_dim, ranks[0])
        assert dense.out_features == len(ranks) * head_dim
        replacement = HeadwiseSVDProjection(
            writer.to(dense.weight), decoder.to(dense.weight)
        )
        replacement.train(dense.training)
        layer.self_attn.v_proj = replacement
        records.append(
            {
                "layer": index,
                "ranks": list(ranks),
                "projection": "v",
                "writer_shape": list(writer.shape),
                "decoder_shape": list(decoder.shape),
                "runtime": "per-head weight-SVD writer and decoder",
                "heads_per_group": 1,
            }
        )
    return records
