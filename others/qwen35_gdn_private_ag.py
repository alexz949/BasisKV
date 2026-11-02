"""Decoder-closed C1 ALS factors for Qwen3.5 GDN output wires."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from others.tp_source_wo_c1 import TPSourceWOLayout
from others.tp_source_wo_fit import TPSourceWOFitConfig, fit_tp_source_wo_c1


@dataclass(frozen=True)
class Qwen35PrivateAGJointFactors:
    """One layer of source-private encoders and a decoder-closed joint map."""

    private_encoders: Tensor
    joint_decoder_weight: Tensor
    metrics: dict[str, Any]


@torch.no_grad()
def fit_qwen35_private_ag_joint_factors(
    weight: Tensor,
    fit_second_moment: Tensor,
    heldout_second_moment: Tensor,
    *,
    tp_size: int,
    local_rank: int,
    encoder_sweeps: int = 5,
    covariance_damping: float = 1.0e-5,
    work_dtype: torch.dtype = torch.float32,
    factor_dtype: torch.dtype = torch.bfloat16,
) -> Qwen35PrivateAGJointFactors:
    """Fit one Qwen3.5 output projection with decoder-closed C1 ALS.

    The solver starts from activation-weighted source-local encoders, solves a
    joint decoder using the complete cross-source covariance, alternates exact
    decoder solves with per-source CG encoder updates. It returns the final
    decoder-refitted factors; held-out error is evaluated after fitting.
    """

    if weight.ndim != 2:
        raise ValueError("Qwen3.5 output-projection weight must be a matrix")
    output_width, input_width = map(int, weight.shape)
    layout = TPSourceWOLayout(
        input_width=input_width,
        output_width=output_width,
        tp_size=tp_size,
        source_rank=local_rank,
    )
    fit_config = TPSourceWOFitConfig(
        encoder_sweeps=encoder_sweeps,
        covariance_damping=covariance_damping,
    )
    fitted = fit_tp_source_wo_c1(
        weight,
        fit_second_moment,
        heldout_second_moment,
        layout,
        config=fit_config,
        work_device=weight.device,
        work_dtype=work_dtype,
        factor_dtype=factor_dtype,
    )
    joint_decoder_weight = (
        fitted.source_decoders.reshape(layout.gathered_width, output_width)
        .transpose(0, 1)
        .contiguous()
    )
    metrics = {
        **layout.accounting(),
        "algorithm": "activation_weighted_initialization_plus_decoder_closed_c1_als",
        "fit_config": asdict(fit_config),
        "sweeps": fitted.sweeps,
        "fit_relative_output_mse": fitted.fit_relative_mse,
        "heldout_relative_output_mse": fitted.heldout_relative_mse,
        "quantized_fit_relative_output_mse": fitted.quantized_fit_relative_mse,
        "quantized_heldout_relative_output_mse": (
            fitted.quantized_heldout_relative_mse
        ),
        "diagnostics": fitted.diagnostics,
    }
    return Qwen35PrivateAGJointFactors(
        private_encoders=fitted.source_encoders,
        joint_decoder_weight=joint_decoder_weight,
        metrics=metrics,
    )


__all__ = [
    "Qwen35PrivateAGJointFactors",
    "fit_qwen35_private_ag_joint_factors",
]
