"""Full-layer V/O joint ALS.

Each KV group g shares one Value encoder A[g] (d x r); each query head h owns a
free output decoder D[h] (r x o). With the cross-head input covariance
C[h, k] (d x d) and the dense per-head target W[h] (d x o), the layer objective
is the quadratic

    loss(A, D) = constant - 2 <D, rhs(A)> + <D, normal(A) D>,

where normal[h, k] = A[g(h)]^T C[h, k] A[g(k)] and rhs[h] = A[g(h)]^T B[h] with
B[h] = sum_k C[h, k] W[k]. ALS alternates a closed-form decoder solve with one
CG solve per group on the shared encoder, then re-orthonormalizes A by QR.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor

# Predicted decreases above this relative size are treated as roundoff.
_ROUNDOFF = 1e-10


class Quadratic(NamedTuple):
    """Sufficient statistics of one layer's objective."""

    covariance: Tensor  # [H, H, d, d]; symmetric under (h, k, i, j) -> (k, h, j, i)
    cross: Tensor  # [H, d, o]; B[h] = sum_k C[h, k] W[k]
    constant: Tensor  # scalar; <W, B>

    def to(self, dtype):
        return Quadratic(self.covariance.to(dtype), self.cross.to(dtype), self.constant)


def symmetrize(covariance):
    return 0.5 * (covariance + covariance.permute(1, 0, 3, 2))


def quadratic_from_target(covariance, target):
    covariance = symmetrize(covariance)
    cross = torch.einsum("hkij,kjo->hio", covariance, target)
    constant = (target.double() * cross.double()).sum()
    return Quadratic(covariance, cross, constant)


def damp_covariance(covariance, relative_damping):
    """Add `relative_damping` times the mean diagonal entry to every diagonal block."""
    work = symmetrize(covariance)
    heads, _, dim, _ = work.shape
    index = torch.arange(heads, device=work.device)
    trace = work[index, index].diagonal(dim1=-2, dim2=-1).sum()
    absolute = relative_damping * float(trace) / (heads * dim)
    if absolute:
        work[index, index] += absolute * torch.eye(dim, device=work.device, dtype=work.dtype)
    return work, absolute


def _thin_qr(value):
    """Batched reduced QR with a non-negative diagonal in R."""
    q, r = torch.linalg.qr(value, mode="reduced")
    signs = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(-2), r * signs.unsqueeze(-1)


def _cholesky(matrix, relative_jitter=0.0):
    """Cholesky factor with the smallest relative jitter (at least `relative_jitter`) that succeeds."""
    matrix = 0.5 * (matrix + matrix.T)
    scale = float(matrix.diagonal().abs().mean().clamp_min(torch.finfo(matrix.dtype).tiny))
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    escalation = (10.0**power for power in range(-12, -1))
    for relative in (relative_jitter, *(value for value in escalation if value > relative_jitter)):
        absolute = relative * scale
        factor, info = torch.linalg.cholesky_ex(matrix + absolute * identity, check_errors=False)
        # CUDA may report success for a non-finite factor; keep searching in that case.
        succeeded = int(info) == 0 and bool(torch.isfinite(factor).all())
        if succeeded:
            break
    assert succeeded, "matrix is not positive definite even with 1e-2 relative jitter"
    return factor, absolute


@torch.no_grad()
def initialize_weighted_svd(covariance, target, mapping, rank):
    """Leading subspace of L^T [W_h]_{h in g} per group, where L L^T is the pooled covariance."""
    groups = int(mapping.max()) + 1
    dim = covariance.shape[-1]
    encoder = covariance.new_empty(groups, dim, rank)
    diagnostics = []
    for group in range(groups):
        heads = torch.nonzero(mapping == group).flatten()
        factor, jitter = _cholesky(covariance[heads, heads].mean(dim=0))
        weighted = factor.T @ target[heads].permute(1, 0, 2).reshape(dim, -1)
        gram = weighted @ weighted.T
        # eigh sorts ascending; the last `rank` columns span the leading subspace.
        _, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
        leading = vectors[:, -rank:].flip(1)
        encoder[group] = torch.linalg.solve_triangular(factor.T, leading, upper=True)
        diagnostics.append({"group_index": group, "absolute_jitter": jitter})
    return _thin_qr(encoder)[0], diagnostics


def _normal_equations(objective, encoder, mapping):
    by_head = encoder[mapping]
    normal = torch.einsum("hia,hkij,kjb->hkab", by_head, objective.covariance, by_head)
    rhs = torch.einsum("hia,hio->hao", by_head, objective.cross)
    return normal, rhs


def solve_decoder(objective, encoder, mapping, relative_jitter=0.0):
    """Closed-form full-layer decoder for fixed encoders."""
    normal, rhs = _normal_equations(Quadratic(*objective), encoder, mapping)
    heads, rank, width = rhs.shape
    normal = normal.permute(0, 2, 1, 3).reshape(heads * rank, heads * rank)
    rhs = rhs.reshape(heads * rank, width)
    factor, jitter = _cholesky(normal, relative_jitter)
    decoder = torch.cholesky_solve(rhs, factor)
    residual = torch.linalg.vector_norm(normal @ decoder - rhs)
    residual = residual / torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).tiny)
    info = {"absolute_jitter": jitter, "relative_residual": float(residual)}
    return decoder.reshape(heads, rank, width), info


def evaluate_loss(objective, encoder, decoder, mapping):
    """Loss in FP64; returns (loss, loss / constant, magnitude of the largest term)."""
    roundoff = max(1024 * torch.finfo(decoder.dtype).eps, _ROUNDOFF)
    exact = Quadratic(*objective).to(torch.float64)
    normal, rhs = _normal_equations(exact, encoder.double(), mapping)
    decoder = decoder.double()
    quadratic = float(torch.einsum("hao,hkab,kbo->", decoder, normal, decoder))
    linear = 2 * float((decoder * rhs).sum())
    constant = float(exact.constant)
    loss = constant - linear + quadratic
    scale = max(abs(constant), abs(linear), abs(quadratic), abs(loss), 1.0)
    assert math.isfinite(loss), "joint objective became non-finite"
    if loss < 0:
        assert -loss <= roundoff * scale, "PSD joint objective is materially negative"
        loss = 0.0
    return loss, loss / max(abs(constant), 1e-300), scale


def _encoder_gradient(objective, encoder, decoder, mapping, heads, grams):
    """Half gradient of the loss with respect to one group's shared encoder."""
    by_head = encoder[mapping]
    quadratic = torch.einsum("hkij,kja,khab->hib", objective.covariance[heads], by_head, grams[:, heads])
    linear = torch.einsum("hio,hbo->hib", objective.cross[heads], decoder[heads])
    return (quadratic - linear).sum(dim=0)


def _encoder_hessian(covariance, grams, heads):
    """Hessian-vector product restricted to one group's encoder block."""
    block = covariance[heads][:, heads]
    gram_block = grams[heads][:, heads]

    def operator(delta):
        return torch.einsum("hkij,ja,khab->ib", block, delta, gram_block)

    return operator


def encoder_hessian_vector_product(covariance, decoder, heads, delta):
    """Standalone form of the encoder Hessian-vector product for one group."""
    heads = torch.as_tensor(heads, device=covariance.device)
    grams = torch.einsum("kao,hbo->khab", decoder, decoder)
    return _encoder_hessian(covariance, grams, heads)(delta)


def conjugate_gradient(operator, rhs, *, iterations, tolerance=0.0, preconditioner=lambda r: r):
    """Preconditioned CG on an SPD operator with matrix-shaped unknowns.

    Stops after `iterations` steps or once the relative residual is at most
    `tolerance`; `tolerance=0` runs the full budget unless the solve is exact.
    """
    rhs_norm = max(float(torch.linalg.vector_norm(rhs)), torch.finfo(rhs.dtype).tiny)
    solution, residual = torch.zeros_like(rhs), rhs.clone()
    z = preconditioner(residual)
    direction, inner = z.clone(), (residual * z).sum()
    relative, completed = float(torch.linalg.vector_norm(residual)) / rhs_norm, 0
    while completed < iterations and relative > tolerance:
        product = operator(direction)
        curvature = float((direction * product).sum())
        if curvature <= torch.finfo(rhs.dtype).eps * float(direction.square().sum()):
            break  # exact solution or lost positive definiteness
        step = inner / curvature
        solution += step * direction
        residual -= step * product
        z = preconditioner(residual)
        new_inner = (residual * z).sum()
        direction = z + (new_inner / inner) * direction
        inner = new_inner
        relative = float(torch.linalg.vector_norm(residual)) / rhs_norm
        completed += 1
    return solution, {"iterations": completed, "relative_residual": relative, "converged": relative <= tolerance}


def _encoder_sweep(work, encoder, decoder, mapping, groups, loss, *, cg_iterations):
    """One CG update per group on the shared encoders, then QR normalization."""
    grams = torch.einsum("kao,hbo->khab", decoder, decoder)
    records = []
    for group, heads in enumerate(groups):
        gradient = _encoder_gradient(work, encoder, decoder, mapping, heads, grams)
        hessian = _encoder_hessian(work.covariance, grams, heads)
        delta, cg = conjugate_gradient(hessian, -gradient, iterations=cg_iterations)
        predicted = float(2 * (gradient * delta).sum() + (delta * hessian(delta)).sum())
        accepted = predicted <= _ROUNDOFF * max(abs(loss), 1.0)
        if accepted:
            encoder[group] += delta
            loss += predicted
        records.append({
            "group_index": group,
            "accepted": accepted,
            "predicted_change": predicted if accepted else 0.0,
            "loss": loss,
            **cg,
        })
    # Absorb the QR scale into D so that A[g(h)] @ D[h] is unchanged.
    encoder, triangular = _thin_qr(encoder)
    decoder = torch.einsum("hab,hbo->hao", triangular[mapping], decoder)
    return encoder, decoder, loss, records


@torch.no_grad()
def fit_joint(objective, encoder, mapping, *, sweeps=6, cg_iterations=16, work_dtype=torch.float32):
    """ALS from an initial encoder. Losses use the FP64 objective; solves use `work_dtype`."""
    exact = Quadratic(*objective).to(torch.float64)
    work = exact.to(work_dtype)
    groups = [torch.nonzero(mapping == group).flatten() for group in range(encoder.shape[0])]
    encoder, _ = _thin_qr(encoder.to(work_dtype))
    initial_loss = float(exact.constant)
    slack = max(1e-8, 1024 * torch.finfo(work_dtype).eps)

    def tolerance(previous, scale=1.0):
        return slack * max(1.0, abs(initial_loss), abs(previous), scale)

    def refit_decoder(encoder, previous):
        decoder, _ = solve_decoder(work, encoder, mapping)
        loss, _, scale = evaluate_loss(exact, encoder, decoder, mapping)
        assert loss <= previous + tolerance(previous, scale), "decoder solve increased the objective"
        return decoder, loss

    decoder, decoder_only_loss = refit_decoder(encoder, initial_loss)
    loss, updates = decoder_only_loss, []
    for sweep in range(1, sweeps + 1):
        if sweep > 1:
            decoder, loss = refit_decoder(encoder, loss)
        before = loss
        encoder, decoder, _, records = _encoder_sweep(
            work, encoder, decoder, mapping, groups, loss, cg_iterations=cg_iterations
        )
        loss, _, _ = evaluate_loss(exact, encoder, decoder, mapping)
        assert loss <= before + tolerance(before), "encoder sweep increased the objective"
        updates.extend({"sweep": sweep, **record} for record in records)
    if sweeps:
        decoder, loss = refit_decoder(encoder, loss)
    return encoder.cpu(), decoder.cpu(), {
        "initial_loss": initial_loss,
        "decoder_only_loss": decoder_only_loss,
        "final_loss": loss,
        "sweeps": sweeps,
        "updates": updates,
    }
