"""Generic activation-aware C1 fitting for TP-source output projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from core.als import damp_covariance, evaluate_loss, fit_joint, initialize_weighted_svd, quadratic_from_target
from others.tp_source_wo_c1 import TPSourceWOLayout, covariance_to_source_blocks, identity_factors, weight_to_source_targets


@dataclass(frozen=True)
class TPSourceWOFitConfig:
    """Numerical controls for one rectangular TP-source C1 fit."""

    encoder_sweeps: int = 1
    covariance_damping: float = 1.0e-5
    encoder_cg_iterations: int = 16

    def validate(self) -> None:
        if self.encoder_sweeps < 0:
            raise ValueError("invalid encoder sweep range")
        if self.covariance_damping < 0:
            raise ValueError("C1 damping and tolerances must be nonnegative")
        if self.encoder_cg_iterations <= 0:
            raise ValueError("C1 iteration controls must be positive")


@dataclass(frozen=True)
class TPSourceWOFitResult:
    source_encoders: Tensor
    source_decoders: Tensor
    fit_relative_mse: float
    heldout_relative_mse: float
    quantized_fit_relative_mse: float
    quantized_heldout_relative_mse: float
    sweeps: int
    diagnostics: dict[str, Any]


def _relative(loss: float, constant: Tensor) -> float:
    return float(loss) / max(abs(float(constant)), 1.0e-300)


@torch.no_grad()
def fit_tp_source_wo_c1(
    weight: Tensor,
    fit_covariance: Tensor,
    heldout_covariance: Tensor,
    layout: TPSourceWOLayout,
    *,
    config: TPSourceWOFitConfig = TPSourceWOFitConfig(),
    work_device: torch.device | str | None = None,
    work_dtype: torch.dtype = torch.float32,
    factor_dtype: torch.dtype = torch.bfloat16,
) -> TPSourceWOFitResult:
    """Fit one post-gate/output-projection C1 map and select on held-out MSE."""

    config.validate()
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("C1 work dtype must be float32 or float64")
    if factor_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("unsupported C1 factor dtype")
    device = torch.device(weight.device if work_device is None else work_device)
    if tuple(weight.shape) != (layout.output_width, layout.input_width):
        raise ValueError("C1 weight does not match its layout")
    expected_covariance = (layout.input_width, layout.input_width)
    if tuple(fit_covariance.shape) != expected_covariance or tuple(
        heldout_covariance.shape
    ) != expected_covariance:
        raise ValueError("C1 covariance does not match its layout")

    if layout.source_rank == layout.source_width:
        encoders, decoders = identity_factors(
            weight.to(device=device), layout, dtype=factor_dtype
        )
        return TPSourceWOFitResult(
            source_encoders=encoders.cpu().contiguous(),
            source_decoders=decoders.cpu().contiguous(),
            fit_relative_mse=0.0,
            heldout_relative_mse=0.0,
            quantized_fit_relative_mse=0.0,
            quantized_heldout_relative_mse=0.0,
            sweeps=0,
            diagnostics={"method": "analytic_identity"},
        )

    fit_raw = covariance_to_source_blocks(
        fit_covariance, layout, device=device, dtype=torch.float64
    )
    heldout_blocks = covariance_to_source_blocks(
        heldout_covariance, layout, device=device, dtype=torch.float64
    )
    fit_blocks, absolute_damping = damp_covariance(fit_raw, config.covariance_damping)
    target = weight_to_source_targets(
        weight, layout, device=device, dtype=torch.float64
    )
    fit_objective = quadratic_from_target(fit_blocks, target)
    heldout_objective = quadratic_from_target(heldout_blocks, target)
    del fit_raw, fit_blocks, heldout_blocks
    mapping = torch.arange(layout.tp_size, device=device, dtype=torch.long)
    initial_A, _ = initialize_weighted_svd(
        fit_objective.covariance.to(dtype=work_dtype),
        target.to(dtype=work_dtype),
        mapping,
        layout.source_rank,
    )
    final_A, final_D, info = fit_joint(
        fit_objective,
        initial_A,
        mapping,
        sweeps=config.encoder_sweeps,
        cg_iterations=config.encoder_cg_iterations,
        work_dtype=work_dtype,
    )
    final_A = final_A.to(device=device, dtype=work_dtype)
    final_D = final_D.to(device=device, dtype=work_dtype)

    fit_loss = evaluate_loss(fit_objective, final_A, final_D, mapping)[0]
    heldout_loss = evaluate_loss(heldout_objective, final_A, final_D, mapping)[0]
    stored_A = final_A.to(dtype=factor_dtype)
    stored_D = final_D.to(dtype=factor_dtype)
    quantized_A = stored_A.to(device=device, dtype=work_dtype)
    quantized_D = stored_D.to(device=device, dtype=work_dtype)
    quantized_fit = evaluate_loss(fit_objective, quantized_A, quantized_D, mapping)[0]
    quantized_heldout = evaluate_loss(heldout_objective, quantized_A, quantized_D, mapping)[0]
    encoder_solves = [
        {
            "sweep": int(step["sweep"]),
            "source": int(step["group_index"]),
            "solver": "fixed_iteration_conjugate_gradient",
            "cg_iterations": int(step["iterations"]),
            "relative_residual": float(step["relative_residual"]),
            "two_sided_solve_relative_residual": None,
            "accepted": bool(step["accepted"]),
            "loss": float(step["loss"]),
        }
        for step in info["updates"]
    ]
    accepted_encoder_solves = [row for row in encoder_solves if row["accepted"]]
    diagnostics = {
        "method": "activation_weighted_initialization_plus_joint_c1_als_with_fixed_iteration_conjugate_gradient_encoder_solves",
        "absolute_covariance_damping": absolute_damping,
        "encoder_sweeps_completed": info["sweeps"],
        "initial_relative_mse": _relative(info["initial_loss"], fit_objective.constant),
        "decoder_only_relative_mse": _relative(
            info["decoder_only_loss"], fit_objective.constant
        ),
        "final_relative_mse": _relative(info["final_loss"], fit_objective.constant),
        "encoder_group_solves": len(encoder_solves),
        "encoder_maximum_relative_residual": max(
            (float(row["relative_residual"]) for row in encoder_solves), default=0.0
        ),
        "encoder_maximum_accepted_relative_residual": max(
            (float(row["relative_residual"]) for row in accepted_encoder_solves),
            default=0.0,
        ),
        "encoder_solves": encoder_solves,
    }
    return TPSourceWOFitResult(
        source_encoders=stored_A.cpu().contiguous(),
        source_decoders=stored_D.cpu().contiguous(),
        fit_relative_mse=_relative(fit_loss, fit_objective.constant),
        heldout_relative_mse=_relative(heldout_loss, heldout_objective.constant),
        quantized_fit_relative_mse=_relative(quantized_fit, fit_objective.constant),
        quantized_heldout_relative_mse=_relative(
            quantized_heldout, heldout_objective.constant
        ),
        sweeps=info["sweeps"],
        diagnostics=diagnostics,
    )


__all__ = [
    "TPSourceWOFitConfig",
    "TPSourceWOFitResult",
    "fit_tp_source_wo_c1",
]
