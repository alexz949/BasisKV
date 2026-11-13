"""Matrix-shaped preconditioned conjugate gradients for gated Value fitting."""
from dataclasses import dataclass
from typing import Callable
import torch

@dataclass(frozen=True)
class CGDiagnostics:
    iterations: int
    converged: bool
    relative_residual: float
    absolute_damping: float
    negative_curvature: bool
    initial_residual_norm: float
    final_residual_norm: float
    fixed_iterations: bool

def conjugate_gradient_matrix(operator: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, *, relative_tolerance: float, max_iterations: int, absolute_damping: float=0.0, fixed_iterations: bool=False, preconditioner: Callable[[torch.Tensor], torch.Tensor] | None=None) -> tuple[torch.Tensor, CGDiagnostics]:
    assert not (relative_tolerance <= 0 or max_iterations <= 0 or absolute_damping < 0), 'invalid CG tolerance, iteration count, or damping'
    solution = torch.zeros_like(rhs)

    def damped(value: torch.Tensor) -> torch.Tensor:
        result = operator(value)
        if absolute_damping:
            result = result + absolute_damping * value
        return result

    def precondition(value: torch.Tensor) -> torch.Tensor:
        result = value.clone() if preconditioner is None else preconditioner(value)
        assert not result.shape != value.shape, 'CG preconditioner must preserve the matrix shape'
        assert not not torch.isfinite(result).all(), 'CG preconditioner returned non-finite values'
        return result
    residual = rhs.clone()
    preconditioned_residual = precondition(residual)
    direction = preconditioned_residual.clone()
    residual_square = torch.sum(residual * residual)
    residual_inner_product = torch.sum(residual * preconditioned_residual)
    rhs_norm = torch.sqrt(residual_square).clamp_min(torch.finfo(rhs.dtype).tiny)
    initial_residual_norm = float(torch.sqrt(residual_square))
    negative_curvature = False
    converged = initial_residual_norm == 0.0
    iterations = 0
    if converged:
        return (solution, CGDiagnostics(iterations=0, converged=True, relative_residual=0.0, absolute_damping=float(absolute_damping), negative_curvature=False, initial_residual_norm=0.0, final_residual_norm=0.0, fixed_iterations=bool(fixed_iterations)))
    assert not float(residual_inner_product) <= 0.0, 'CG preconditioner must be positive definite'
    for iterations in range(1, max_iterations + 1):
        product = damped(direction)
        curvature = torch.sum(direction * product)
        tolerance = torch.finfo(rhs.dtype).eps * torch.linalg.vector_norm(direction).square()
        if float(curvature) <= -float(tolerance):
            negative_curvature = True
            break
        if float(curvature) <= float(tolerance):
            break
        step = residual_inner_product / curvature
        solution.add_(step * direction)
        residual.add_(-step * product)
        new_square = torch.sum(residual * residual)
        relative = float(torch.sqrt(new_square) / rhs_norm)
        if relative <= relative_tolerance:
            residual_square = new_square
            converged = True
            if not fixed_iterations or float(new_square) == 0.0:
                break
        preconditioned_residual = precondition(residual)
        new_inner_product = torch.sum(residual * preconditioned_residual)
        assert not float(new_inner_product) <= 0.0, 'CG preconditioner must be positive definite'
        direction = preconditioned_residual + new_inner_product / residual_inner_product * direction
        residual_square = new_square
        residual_inner_product = new_inner_product
    relative = float(torch.sqrt(residual_square) / rhs_norm)
    converged = relative <= relative_tolerance
    return (solution, CGDiagnostics(iterations=iterations, converged=converged, relative_residual=relative, absolute_damping=float(absolute_damping), negative_curvature=negative_curvature, initial_residual_norm=initial_residual_norm, final_residual_norm=float(torch.sqrt(residual_square)), fixed_iterations=bool(fixed_iterations)))
