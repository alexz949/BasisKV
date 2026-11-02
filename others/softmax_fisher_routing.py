"""Softmax-Fisher routing objectives of the historical S80 experiments.

These operate on joint [V, K] token rows, explicit target maps, and active row
subsets. The release router (core.page_fisher) is the special case with
residual-key features only, an identity target, and every row active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from others.gqa_joint_routing_payload_s80_stats import S80DirectResidualData

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from core.als import conjugate_gradient
from core.page_fisher import unpack_grams
from core.router import AffineMap, fit_affine_map


@dataclass(frozen=True)
class PageFisherRouterSweep:
    sweep: int
    loss_before: float
    loss_after_queries: float
    loss_after_encoder: float


@dataclass(frozen=True)
class PageFisherRouterFit:
    routing_encoders: torch.Tensor
    routing_query_factors: torch.Tensor
    sweeps: tuple[PageFisherRouterSweep, ...]
    final_query_diagnostics: tuple[dict[str, Any], ...]
    encoder_diagnostics: tuple[tuple[dict[str, Any], ...], ...]


def _selector(statistics: S80CompactSoftmaxFisherRouting) -> torch.Tensor:
    result = statistics.queries_by_head.new_zeros(
        statistics.key_dim,
        statistics.joint_dim,
    )
    result[:, statistics.value_dim :] = torch.eye(
        statistics.key_dim,
        device=result.device,
        dtype=result.dtype,
    )
    return result


def _target_maps(
    statistics: S80CompactSoftmaxFisherRouting,
    target_maps: torch.Tensor | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if target_maps is None:
        return _selector(statistics).to(like).unsqueeze(0).expand(
            statistics.queries_by_head.shape[0],
            -1,
            -1,
        )
    return target_maps.to(like)


def _relative_damping(diagonal: torch.Tensor, relative: float) -> float:
    return float(relative) * max(
        float(diagonal.abs().mean()),
        torch.finfo(diagonal.dtype).tiny,
    )


def _diagonal_preconditioner(
    diagonal: torch.Tensor,
    *,
    damping: float,
):
    shifted = diagonal.clamp_min(0).add(float(damping))
    shifted.clamp_min_(torch.finfo(shifted.dtype).tiny)

    def apply(value: torch.Tensor) -> torch.Tensor:
        return value / shifted

    return apply


def refit_page_fisher_query_factors(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    routing_encoders: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...]]:
    """Solve unrestricted per-head query maps for fixed routing encoders."""

    mapping = statistics.head_to_kv_group.to(
        device=routing_encoders.device,
        dtype=torch.long,
    )
    queries = statistics.queries_by_head.to(routing_encoders)
    grams = statistics.fisher_grams_by_head.to(routing_encoders)
    targets = _target_maps(
        statistics,
        target_maps,
        like=routing_encoders,
    )
    fitted = routing_encoders.new_empty(
        statistics.queries_by_head.shape[0],
        statistics.key_dim,
        routing_encoders.shape[-1],
    )
    diagnostics: list[dict[str, Any]] = []
    for head, group in enumerate(mapping.tolist()):
        query = queries[head]
        gram = grams[head]
        encoder = routing_encoders[group]
        right_factors = torch.einsum(
            "ir,dij,js->drs",
            encoder,
            gram,
            encoder,
        )
        target = query @ targets[head]
        weighted_target = torch.einsum(
            "di,dij,jr->dr",
            target,
            gram,
            encoder,
        )
        rhs = query.mT @ weighted_target

        def operator(value: torch.Tensor) -> torch.Tensor:
            codes = query @ value
            transformed = torch.einsum("dr,drs->ds", codes, right_factors)
            return query.mT @ transformed

        diagonal = torch.einsum(
            "di,dr->ir",
            query.square(),
            right_factors.diagonal(dim1=-2, dim2=-1),
        )
        damping = _relative_damping(diagonal, relative_damping)
        solution, record = conjugate_gradient(
            lambda value: operator(value) + damping * value,
            rhs,
            iterations=max_iterations,
            tolerance=relative_tolerance,
            preconditioner=_diagonal_preconditioner(diagonal, damping=damping),
        )
        fitted[head] = solution
        diagnostics.append(record)
    return fitted, tuple(diagnostics)


def refit_page_fisher_routing_encoders(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    routing_query_factors: torch.Tensor,
    active_joint_rows: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...]]:
    """Solve group encoders with support restricted to selected joint rows."""

    queries = statistics.queries_by_head.to(routing_query_factors)
    grams = statistics.fisher_grams_by_head.to(routing_query_factors)
    mapping = statistics.head_to_kv_group.to(
        device=routing_query_factors.device,
        dtype=torch.long,
    )
    active = active_joint_rows.to(
        device=routing_query_factors.device,
        dtype=torch.long,
    )
    targets = _target_maps(
        statistics,
        target_maps,
        like=routing_query_factors,
    )
    encoders = routing_query_factors.new_zeros(
        int(mapping.max()) + 1,
        statistics.joint_dim,
        routing_query_factors.shape[-1],
    )
    diagnostics: list[dict[str, Any]] = []
    for group in range(encoders.shape[0]):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        group_queries = queries.index_select(0, heads)
        group_grams = grams.index_select(0, heads)
        query_factors = routing_query_factors.index_select(0, heads)
        query_codes = torch.einsum(
            "hdk,hkr->hdr",
            group_queries,
            query_factors,
        )
        active_grams = group_grams.index_select(-2, active).index_select(-1, active)
        active_to_joint = group_grams.index_select(-2, active)
        group_targets = targets.index_select(0, heads)
        target = torch.einsum("hdk,hki->hdi", group_queries, group_targets)
        weighted_target = torch.einsum(
            "hdij,hdj->hdi",
            active_to_joint,
            target,
        )
        rhs = torch.einsum("hdi,hdr->ir", weighted_target, query_codes)

        def operator(value: torch.Tensor) -> torch.Tensor:
            # Associate G @ (E @ u) before the final outer product with u.
            # This avoids a [heads, documents, features, rank] intermediate
            # and applies each Fisher Gram to one vector instead of rank vectors.
            projected = torch.einsum("ir,hdr->hdi", value, query_codes)
            transformed = torch.einsum("hdij,hdj->hdi", active_grams, projected)
            return torch.einsum("hdi,hdr->ir", transformed, query_codes)

        diagonal = torch.einsum(
            "hdi,hdr->ir",
            active_grams.diagonal(dim1=-2, dim2=-1),
            query_codes.square(),
        )
        damping = _relative_damping(diagonal, relative_damping)
        solution, record = conjugate_gradient(
            lambda value: operator(value) + damping * value,
            rhs,
            iterations=max_iterations,
            tolerance=relative_tolerance,
            preconditioner=_diagonal_preconditioner(diagonal, damping=damping),
        )
        encoders[group].index_copy_(0, active, solution)
        diagnostics.append(record)
    return encoders, tuple(diagnostics)


def fit_page_fisher_router(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    initial_routing_encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    active_joint_rows: torch.Tensor,
    sweeps: int,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> PageFisherRouterFit:
    """Alternately close unrestricted query maps and supported encoders."""

    encoders = initial_routing_encoders.to(
        device=statistics.queries_by_head.device,
        dtype=statistics.queries_by_head.dtype,
    ).clone()
    query_factors = initial_query_factors.to(encoders).clone()
    records: list[PageFisherRouterSweep] = []
    encoder_diagnostics: list[tuple[dict[str, Any], ...]] = []
    query_diagnostics: tuple[dict[str, Any], ...] = ()

    def loss() -> float:
        return compact_softmax_fisher_loss(
            statistics,
            routing_payload_encoders=encoders,
            routing_query_factors=query_factors,
            target_maps=target_maps,
        )

    for sweep in range(1, int(sweeps) + 1):
        before = loss()
        query_factors, query_diagnostics = refit_page_fisher_query_factors(
            statistics,
            routing_encoders=encoders,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            target_maps=target_maps,
        )
        after_queries = loss()
        encoders, encoder_step = refit_page_fisher_routing_encoders(
            statistics,
            routing_query_factors=query_factors,
            active_joint_rows=active_joint_rows,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            target_maps=target_maps,
        )
        encoder_diagnostics.append(encoder_step)
        records.append(
            PageFisherRouterSweep(
                sweep=sweep,
                loss_before=before,
                loss_after_queries=after_queries,
                loss_after_encoder=loss(),
            )
        )
        print(
            f"Page-Fisher sweep {sweep}/{sweeps}: "
            f"loss {before:.6g} -> {records[-1].loss_after_encoder:.6g}",
            flush=True,
        )
    query_factors, query_diagnostics = refit_page_fisher_query_factors(
        statistics,
        routing_encoders=encoders,
        relative_damping=relative_damping,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        target_maps=target_maps,
    )
    return PageFisherRouterFit(
        routing_encoders=encoders,
        routing_query_factors=query_factors,
        sweeps=tuple(records),
        final_query_diagnostics=query_diagnostics,
        encoder_diagnostics=tuple(encoder_diagnostics),
    )


@dataclass(frozen=True)
class S80SoftmaxFisherRouting:
    queries_by_head: torch.Tensor
    joint_rows_by_group: torch.Tensor
    exact_scores_by_head: torch.Tensor
    probabilities_by_head: torch.Tensor
    head_to_kv_group: torch.Tensor
    value_dim: int
    key_dim: int
    scaling: float
    teacher_fisher_energy: float

    @property
    def documents(self) -> int:
        return int(self.queries_by_head.shape[1])

    @property
    def tokens(self) -> int:
        return int(self.joint_rows_by_group.shape[2])


@dataclass(frozen=True)
class S80CompactSoftmaxFisherRouting:
    """Per-query Fisher sufficient statistics without raw token rows."""

    queries_by_head: torch.Tensor
    fisher_grams_by_head: torch.Tensor
    head_to_kv_group: torch.Tensor
    value_dim: int
    key_dim: int
    scaling: float
    teacher_fisher_energy: float

    @property
    def documents(self) -> int:
        return int(self.queries_by_head.shape[1])

    @property
    def joint_dim(self) -> int:
        return int(self.fisher_grams_by_head.shape[-1])

    def validate(self) -> None:
        return None


def prepare_compact_softmax_fisher_routing(
    *,
    queries_by_head: torch.Tensor,
    fisher_grams_packed_by_head: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    value_dim: int,
    key_dim: int,
    scaling: float,
    teacher_fisher_energy: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> S80CompactSoftmaxFisherRouting:
    """Materialize packed per-query Fisher statistics for one layer."""

    target_device = torch.device(device)
    joint_dim = int(value_dim) + int(key_dim)
    packed = fisher_grams_packed_by_head.to(
        device=target_device,
        dtype=dtype,
    )
    result = S80CompactSoftmaxFisherRouting(
        queries_by_head=queries_by_head.to(
            device=target_device,
            dtype=dtype,
        ),
        fisher_grams_by_head=unpack_grams(packed, joint_dim),
        head_to_kv_group=head_to_kv_group.to(
            device=target_device,
            dtype=torch.long,
        ),
        value_dim=int(value_dim),
        key_dim=int(key_dim),
        scaling=float(scaling),
        teacher_fisher_energy=float(teacher_fisher_energy),
    )
    result.validate()
    return result


def softmax_fisher_gram(
    queries: torch.Tensor,
    joint_rows: torch.Tensor,
    *,
    value_dim: int,
    scaling: float,
) -> tuple[torch.Tensor, float]:
    """Return exact per-query ``X.T J_p X`` and teacher Fisher energy."""

    scores = scaling * queries @ joint_rows[:, value_dim:].mT
    probabilities = torch.softmax(scores, dim=-1)
    means = probabilities @ joint_rows
    centered = joint_rows.unsqueeze(0) - means.unsqueeze(1)
    centered.mul_(torch.sqrt(probabilities).unsqueeze(-1))
    grams = torch.bmm(centered.mT, centered)
    grams = 0.5 * (grams + grams.mT)
    score_means = torch.sum(probabilities * scores, dim=-1, keepdim=True)
    teacher_energy = float(
        0.5 * torch.sum(probabilities * (scores - score_means).square())
    )
    return grams, teacher_energy


def page_softmax_fisher_gram(
    queries: torch.Tensor,
    joint_rows: torch.Tensor,
    *,
    value_dim: int,
    scaling: float,
    page_size: int,
) -> tuple[torch.Tensor, float]:
    """Return Page-Fisher Grams over teacher-weighted page representatives.

    Each page is represented by the conditional teacher expectation of the
    joint ``[V, K_post]`` rows inside that page.  The outer Fisher distribution
    is the teacher attention mass assigned to each page.
    """

    tokens = int(joint_rows.shape[0])
    pages = (tokens + int(page_size) - 1) // int(page_size)
    padded_tokens = pages * int(page_size)
    padding = padded_tokens - tokens
    scores = scaling * queries @ joint_rows[:, value_dim:].mT
    probabilities = torch.softmax(scores, dim=-1)
    if padding:
        probabilities = torch.nn.functional.pad(probabilities, (0, padding))
        joint_rows = torch.nn.functional.pad(joint_rows, (0, 0, 0, padding))
    probabilities_by_page = probabilities.reshape(
        queries.shape[0],
        pages,
        int(page_size),
    )
    rows_by_page = joint_rows.reshape(
        pages,
        int(page_size),
        joint_rows.shape[-1],
    )
    page_mass = probabilities_by_page.sum(dim=-1)
    page_numerators = torch.einsum(
        "hps,psd->hpd",
        probabilities_by_page,
        rows_by_page,
    )
    page_rows = page_numerators / page_mass.clamp_min(
        torch.finfo(page_mass.dtype).tiny
    ).unsqueeze(-1)
    page_mean = torch.einsum("hp,hpd->hd", page_mass, page_rows)
    centered = page_rows - page_mean.unsqueeze(1)
    centered.mul_(torch.sqrt(page_mass).unsqueeze(-1))
    grams = torch.bmm(centered.mT, centered)
    grams = 0.5 * (grams + grams.mT)

    page_key_rows = page_rows[..., value_dim:]
    page_scores = scaling * torch.einsum(
        "hk,hpk->hp",
        queries,
        page_key_rows,
    )
    page_score_mean = torch.sum(page_mass * page_scores, dim=-1, keepdim=True)
    teacher_energy = float(
        0.5 * torch.sum(page_mass * (page_scores - page_score_mean).square())
    )
    return grams, teacher_energy


def compact_softmax_fisher_roots(
    routing: S80CompactSoftmaxFisherRouting,
) -> torch.Tensor:
    """Return roots ``L`` satisfying ``L L.T = X.T J_p X``."""

    grams = 0.5 * (
        routing.fisher_grams_by_head
        + routing.fisher_grams_by_head.transpose(-1, -2)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(grams)
    eigenvalues = eigenvalues.clamp_min(0)
    return eigenvectors * torch.sqrt(eigenvalues).unsqueeze(-2)


def compact_softmax_fisher_map_loss(
    routing: S80CompactSoftmaxFisherRouting,
    *,
    proxy_maps: torch.Tensor,
    target_maps: torch.Tensor | None = None,
) -> float:
    """Evaluate fixed-teacher Fisher loss between explicit proxy maps."""

    if target_maps is None:
        target_maps = proxy_maps.new_zeros(
            proxy_maps.shape[0],
            routing.key_dim,
            routing.joint_dim,
        )
        target_maps[:, :, routing.value_dim :] = torch.eye(
            routing.key_dim,
            device=proxy_maps.device,
            dtype=proxy_maps.dtype,
        )
    else:
        target_maps = target_maps.to(proxy_maps)
    error_maps = proxy_maps - target_maps
    projected_errors = torch.einsum(
        "hdk,hki->hdi",
        routing.queries_by_head,
        error_maps,
    )
    value = 0.5 * routing.scaling**2 * torch.einsum(
        "hdi,hdij,hdj->",
        projected_errors,
        routing.fisher_grams_by_head,
        projected_errors,
    )
    return float(value)


def compact_softmax_fisher_loss(
    routing: S80CompactSoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
    target_maps: torch.Tensor | None = None,
) -> float:
    """Evaluate the exact fixed-teacher Fisher quadratic from compact Grams."""

    mapping = routing.head_to_kv_group.to(
        device=routing_payload_encoders.device,
        dtype=torch.long,
    )
    encoders = routing_payload_encoders.index_select(0, mapping)
    proxy_maps = torch.bmm(
        routing_query_factors,
        encoders.transpose(1, 2),
    )
    return compact_softmax_fisher_map_loss(
        routing,
        proxy_maps=proxy_maps,
        target_maps=target_maps,
    )


def compact_softmax_fisher_encoder_diagonal(
    routing: S80CompactSoftmaxFisherRouting,
    *,
    head_indices: torch.Tensor,
    projected_queries: torch.Tensor,
) -> torch.Tensor:
    """Return one Route32 encoder block's exact Fisher Hessian diagonal."""

    grams = routing.fisher_grams_by_head.index_select(0, head_indices)
    diagonals = grams.diagonal(dim1=-2, dim2=-1).permute(1, 0, 2)
    return 0.5 * routing.scaling**2 * torch.einsum(
        "dhi,dhr->ir",
        diagonals,
        projected_queries.square(),
    )


def compact_softmax_fisher_adapter_system(
    routing: S80CompactSoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact compact Kronecker system for adapter-U."""

    mapping = routing.head_to_kv_group.to(
        device=routing_payload_encoders.device,
        dtype=torch.long,
    )
    encoders = routing_payload_encoders.index_select(0, mapping)
    query_codes = torch.einsum(
        "hdk,hkr->hdr",
        routing.queries_by_head,
        routing_query_factors,
    )
    proxy_maps = torch.bmm(
        routing_query_factors,
        encoders.transpose(1, 2),
    )
    selector = proxy_maps.new_zeros(routing.key_dim, routing.joint_dim)
    selector[:, routing.value_dim :] = torch.eye(
        routing.key_dim,
        device=selector.device,
        dtype=selector.dtype,
    )
    error_maps = proxy_maps - selector.unsqueeze(0)
    projected_errors = torch.einsum(
        "hdk,hki->hdi",
        routing.queries_by_head,
        error_maps,
    )
    error_gram_encoders = torch.einsum(
        "hdi,hdij,hjr->hdr",
        projected_errors,
        routing.fisher_grams_by_head,
        encoders,
    )
    scale = 0.5 * routing.scaling**2
    left = torch.einsum(
        "hdr,hds->dhrs",
        query_codes,
        query_codes,
    )
    right = scale * torch.einsum(
        "hir,hdij,hjs->dhrs",
        encoders,
        routing.fisher_grams_by_head,
        encoders,
    )
    gradient = scale * torch.einsum(
        "hdr,hds->hrs",
        query_codes,
        error_gram_encoders,
    )
    return left, right, -gradient


def softmax_fisher_transform(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return a residual whose squared norm is ``0.5 delta^T J_p delta``."""

    mean = torch.sum(probabilities * scores, dim=-1, keepdim=True)
    centered = scores - mean
    return torch.sqrt(0.5 * probabilities) * centered


def softmax_fisher_adjoint(
    residual: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Apply the transpose of :func:`softmax_fisher_transform`."""

    weighted = torch.sqrt(0.5 * probabilities) * residual
    return weighted - probabilities * weighted.sum(dim=-1, keepdim=True)


def prepare_softmax_fisher_routing(
    direct: S80DirectResidualData,
    *,
    head_to_kv_group: torch.Tensor,
    value_dim: int,
    key_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> S80SoftmaxFisherRouting:
    """Materialize teacher logits and probabilities from paired direct capture."""

    target_device = torch.device(device)
    queries = (
        direct.routing_queries.permute(1, 0, 2)
        .contiguous()
        .to(device=target_device, dtype=dtype)
    )
    rows = (
        direct.routing_joint_rows.permute(2, 0, 1, 3)
        .contiguous()
        .to(device=target_device, dtype=dtype)
    )
    mapping = head_to_kv_group.to(device=target_device, dtype=torch.long)
    heads = int(queries.shape[0])
    exact = torch.empty(
        heads,
        int(queries.shape[1]),
        int(rows.shape[2]),
        device=target_device,
        dtype=dtype,
    )
    scaling = key_dim**-0.5
    for group in range(int(rows.shape[0])):
        head_indices = torch.nonzero(mapping == group, as_tuple=False).flatten()
        group_scores = torch.einsum(
            "hdk,dtk->hdt",
            queries.index_select(0, head_indices),
            rows[group, ..., value_dim : value_dim + key_dim],
        )
        exact.index_copy_(0, head_indices, scaling * group_scores)
    probabilities = torch.softmax(exact, dim=-1)
    teacher_energy = float(
        softmax_fisher_transform(exact, probabilities).square().sum()
    )
    return S80SoftmaxFisherRouting(
        queries_by_head=queries,
        joint_rows_by_group=rows,
        exact_scores_by_head=exact,
        probabilities_by_head=probabilities,
        head_to_kv_group=mapping,
        value_dim=int(value_dim),
        key_dim=int(key_dim),
        scaling=float(scaling),
        teacher_fisher_energy=teacher_energy,
    )


def softmax_fisher_proxy_scores(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> torch.Tensor:
    """Compute every Store80 Route32 proxy score in head-major layout."""

    proxy = torch.empty_like(routing.exact_scores_by_head)
    groups = int(routing.joint_rows_by_group.shape[0])
    for group in range(groups):
        head_indices = torch.nonzero(
            routing.head_to_kv_group == group,
            as_tuple=False,
        ).flatten()
        query_code = torch.einsum(
            "hdk,hkr->hdr",
            routing.queries_by_head.index_select(0, head_indices),
            routing_query_factors.index_select(0, head_indices),
        )
        token_code = torch.einsum(
            "dti,ir->dtr",
            routing.joint_rows_by_group[group],
            routing_payload_encoders[group],
        )
        scores = torch.einsum("hdr,dtr->hdt", query_code, token_code)
        proxy.index_copy_(0, head_indices, routing.scaling * scores)
    return proxy


def softmax_fisher_routing_loss(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> float:
    proxy = softmax_fisher_proxy_scores(
        routing,
        routing_payload_encoders=routing_payload_encoders,
        routing_query_factors=routing_query_factors,
    )
    residual = softmax_fisher_transform(
        proxy - routing.exact_scores_by_head,
        routing.probabilities_by_head,
    )
    return float(residual.square().sum())


def softmax_fisher_encoder_diagonal(
    *,
    joint_rows: torch.Tensor,
    projected_queries: torch.Tensor,
    probabilities: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Return the exact Fisher Hessian diagonal for one Route32 encoder block."""

    mean = torch.einsum("dht,dti->dhi", probabilities, joint_rows)
    second = torch.einsum(
        "dht,dti->dhi",
        probabilities,
        joint_rows.square(),
    )
    variance = torch.clamp(second - mean.square(), min=0)
    return 0.5 * scaling**2 * torch.einsum(
        "dhi,dhr->ir",
        variance,
        projected_queries.square(),
    )


def softmax_fisher_adapter_system(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Kronecker factors and normal-equation RHS for adapter-U."""

    heads = int(routing.queries_by_head.shape[0])
    documents = routing.documents
    rank = int(routing_query_factors.shape[-1])
    left = routing_query_factors.new_empty(documents, heads, rank, rank)
    right = routing_query_factors.new_empty(documents, heads, rank, rank)
    gradient = routing_query_factors.new_zeros(heads, rank, rank)
    for group in range(int(routing.joint_rows_by_group.shape[0])):
        head_indices = torch.nonzero(
            routing.head_to_kv_group == group,
            as_tuple=False,
        ).flatten()
        query_code = torch.einsum(
            "hdk,hkr->hdr",
            routing.queries_by_head.index_select(0, head_indices),
            routing_query_factors.index_select(0, head_indices),
        )
        latent = torch.einsum(
            "dti,ir->dtr",
            routing.joint_rows_by_group[group],
            routing_payload_encoders[group],
        )
        exact = routing.exact_scores_by_head.index_select(0, head_indices)
        probabilities = routing.probabilities_by_head.index_select(0, head_indices)
        proxy = routing.scaling * torch.einsum("hdr,dtr->hdt", query_code, latent)
        delta = proxy - exact
        centered = delta - torch.sum(
            probabilities * delta,
            dim=-1,
            keepdim=True,
        )
        fisher_score = 0.5 * probabilities * centered
        weighted_latent = torch.einsum("hdt,dtr->hdr", fisher_score, latent)
        group_gradient = routing.scaling * torch.einsum(
            "hdr,hds->hrs",
            query_code,
            weighted_latent,
        )
        gradient.index_copy_(0, head_indices, group_gradient)
        group_left = torch.einsum("hdr,hds->dhrs", query_code, query_code)
        left[:, head_indices] = group_left
        for local, head in enumerate(head_indices.tolist()):
            probability = probabilities[local]
            mean = torch.einsum("dt,dtr->dr", probability, latent)
            second = torch.einsum(
                "dtr,dt,dts->drs",
                latent,
                probability,
                latent,
            )
            covariance = second - torch.einsum("dr,ds->drs", mean, mean)
            covariance = 0.5 * (covariance + covariance.mT)
            right[:, head] = 0.5 * routing.scaling**2 * covariance
    return left, right, -gradient


def softmax_fisher_routing_diagnostics(
    routing: S80SoftmaxFisherRouting,
    *,
    selection_routing_encoders: torch.Tensor,
    payload_routing_encoders: torch.Tensor,
    payload_only_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
    payload_decoders: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
) -> dict[str, float | int]:
    """Measure score, physical-page, mass, and exact-refined output quality."""
    from others.exact_qk_v_offload import gqa_union_page_mass_mask

    proxy = softmax_fisher_proxy_scores(
        routing,
        routing_payload_encoders=selection_routing_encoders,
        routing_query_factors=routing_query_factors,
    )
    delta = proxy - routing.exact_scores_by_head
    raw_nmse = float(delta.square().sum() / routing.exact_scores_by_head.square().sum())
    fisher_loss = float(
        softmax_fisher_transform(delta, routing.probabilities_by_head).square().sum()
    )
    joint_encoders = torch.cat(
        (payload_routing_encoders, payload_only_encoders),
        dim=-1,
    )
    page_recalls: list[float] = []
    mass_recalls: list[float] = []
    selected_fractions: list[float] = []
    output_error = 0.0
    output_energy = 0.0
    pages_per_head = math.ceil(exact_token_budget / page_size)
    heads_per_group = int(routing.queries_by_head.shape[0]) // int(
        routing.joint_rows_by_group.shape[0]
    )
    for document in range(routing.documents):
        exact_scores = routing.exact_scores_by_head[:, document]
        proxy_scores = proxy[:, document]
        proxy_tokens, proxy_pages = gqa_union_page_mass_mask(
            proxy_scores,
            num_kv_heads=int(routing.joint_rows_by_group.shape[0]),
            page_size=page_size,
            pages_per_query_head=pages_per_head,
        )
        _, teacher_pages = gqa_union_page_mass_mask(
            exact_scores,
            num_kv_heads=int(routing.joint_rows_by_group.shape[0]),
            page_size=page_size,
            pages_per_query_head=pages_per_head,
        )
        page_recalls.extend(
            (
                (proxy_pages & teacher_pages).sum(dim=-1)
                / teacher_pages.sum(dim=-1).clamp_min(1)
            )
            .tolist()
        )
        query_mask = proxy_tokens.repeat_interleave(heads_per_group, dim=0)
        probabilities = routing.probabilities_by_head[:, document]
        mass_recalls.extend((probabilities * query_mask).sum(dim=-1).tolist())
        selected_fractions.extend(proxy_tokens.float().mean(dim=-1).tolist())

        latent = torch.einsum(
            "gti,gir->gtr",
            routing.joint_rows_by_group[:, document],
            joint_encoders,
        )
        expanded_latent = latent.index_select(0, routing.head_to_kv_group)
        dense_latent = torch.einsum("ht,htr->hr", probabilities, expanded_latent)
        sparse_probability = torch.softmax(
            exact_scores.masked_fill(~query_mask, -torch.inf),
            dim=-1,
        )
        sparse_latent = torch.einsum(
            "ht,htr->hr",
            sparse_probability,
            expanded_latent,
        )
        dense_output = torch.einsum("hr,hro->o", dense_latent, payload_decoders)
        sparse_output = torch.einsum("hr,hro->o", sparse_latent, payload_decoders)
        output_error += float((sparse_output - dense_output).square().sum())
        output_energy += float(dense_output.square().sum())
    return {
        "documents": routing.documents,
        "tokens": routing.tokens,
        "raw_score_nmse": raw_nmse,
        "softmax_fisher_loss": fisher_loss,
        "softmax_fisher_nmse": fisher_loss / routing.teacher_fisher_energy,
        "teacher_fisher_energy": routing.teacher_fisher_energy,
        "physical_page_recall_mean": sum(page_recalls) / len(page_recalls),
        "physical_page_recall_minimum": min(page_recalls),
        "attention_mass_recall_mean": sum(mass_recalls) / len(mass_recalls),
        "attention_mass_recall_minimum": min(mass_recalls),
        "selected_token_fraction_mean": (
            sum(selected_fractions) / len(selected_fractions)
        ),
        "exact_refined_output_relative_mse": output_error / output_energy,
        "page_size": page_size,
        "exact_token_budget": exact_token_budget,
    }


def _initial_residual_factors(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = statistics.head_to_kv_group
    groups = int(mapping.max()) + 1
    encoders = []
    for group in range(groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        gram = statistics.fisher_grams_by_head.index_select(0, heads).sum(
            dim=(0, 1)
        )
        _, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        encoders.append(eigenvectors[:, -rank:])
    encoder = torch.stack(encoders).to(device=device, dtype=torch.float32)
    query = encoder.index_select(0, mapping.to(device=device))
    return encoder, query


def _fit_residual_grid(
    fit_statistics: dict[int, S80CompactSoftmaxFisherRouting],
    validation_statistics: dict[int, S80CompactSoftmaxFisherRouting],
    *,
    residual_ranks: tuple[int, ...],
    sweeps: int,
    relative_damping: float,
    iterative_tolerance: float,
    iterative_max_iterations: int,
    device: torch.device,
) -> tuple[
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    factor_bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for base_rank, fit_cpu in fit_statistics.items():
        fit = S80CompactSoftmaxFisherRouting(
            queries_by_head=fit_cpu.queries_by_head.to(device),
            fisher_grams_by_head=fit_cpu.fisher_grams_by_head.to(device),
            head_to_kv_group=fit_cpu.head_to_kv_group.to(device),
            value_dim=0,
            key_dim=fit_cpu.key_dim,
            scaling=fit_cpu.scaling,
            teacher_fisher_energy=fit_cpu.teacher_fisher_energy,
        )
        validation_cpu = validation_statistics[base_rank]
        validation = S80CompactSoftmaxFisherRouting(
            queries_by_head=validation_cpu.queries_by_head.to(device),
            fisher_grams_by_head=validation_cpu.fisher_grams_by_head.to(device),
            head_to_kv_group=validation_cpu.head_to_kv_group.to(device),
            value_dim=0,
            key_dim=validation_cpu.key_dim,
            scaling=validation_cpu.scaling,
            teacher_fisher_energy=validation_cpu.teacher_fisher_energy,
        )
        diagnostics[f"b{base_rank}_r0"] = {
            "base_rank": base_rank,
            "residual_rank": 0,
            "fit_page_fisher_nmse": 1.0,
            "validation_page_fisher_nmse": 1.0,
        }
        for residual_rank in residual_ranks:
            if residual_rank == 0:
                continue
            print(
                f"    fit residual router base={base_rank} rank={residual_rank}",
                flush=True,
            )
            initial_encoder, initial_query = _initial_residual_factors(
                fit,
                rank=residual_rank,
                device=device,
            )
            fitted = fit_page_fisher_router(
                fit,
                initial_routing_encoders=initial_encoder,
                initial_query_factors=initial_query,
                active_joint_rows=torch.arange(fit.key_dim, device=device),
                sweeps=sweeps,
                relative_damping=relative_damping,
                relative_tolerance=iterative_tolerance,
                max_iterations=iterative_max_iterations,
            )
            fit_loss = compact_softmax_fisher_loss(
                fit,
                routing_payload_encoders=fitted.routing_encoders,
                routing_query_factors=fitted.routing_query_factors,
            )
            validation_loss = compact_softmax_fisher_loss(
                validation,
                routing_payload_encoders=fitted.routing_encoders,
                routing_query_factors=fitted.routing_query_factors,
            )
            factor_bank[(base_rank, residual_rank)] = (
                fitted.routing_encoders.float().cpu(),
                fitted.routing_query_factors.float().cpu(),
            )
            diagnostics[f"b{base_rank}_r{residual_rank}"] = {
                "base_rank": base_rank,
                "residual_rank": residual_rank,
                "fit_page_fisher_nmse": fit_loss / fit.teacher_fisher_energy,
                "validation_page_fisher_nmse": (
                    validation_loss / validation.teacher_fisher_energy
                ),
                "sweeps": [asdict(item) for item in fitted.sweeps],
                "final_query_maximum_iterations": max(
                    item["iterations"] for item in fitted.final_query_diagnostics
                ),
                "final_query_maximum_relative_residual": max(
                    item["relative_residual"] for item in fitted.final_query_diagnostics
                ),
            }
        del fit, validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return factor_bank, diagnostics


def fit_affine_metric_reduced_rank_map(
    *,
    row_count: int,
    input_sum: torch.Tensor,
    target_sum: torch.Tensor,
    input_gram: torch.Tensor,
    input_target_gram: torch.Tensor,
    target_metric: torch.Tensor,
    rank: int,
) -> AffineMap:
    """Fit affine reduced-rank regression in a target-side PSD metric.

    For row residuals ``E = K - C M - b``, this solves

    ``min rank(M)<=r ||E H^(1/2)||_F^2``

    for the supplied positive-semidefinite target metric ``H``.  The target
    is transformed by the symmetric square root of ``H``, fitted with the
    ordinary exact RRR solver, and mapped back with the numerical
    Moore-Penrose inverse square root.  A query covariance can therefore
    reorder the retained predictive directions without changing the
    unrestricted linear conditional mean when the metric is nonsingular.
    """

    metric = 0.5 * (target_metric + target_metric.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    scale = eigenvalues.abs().amax()
    cutoff = torch.finfo(metric.dtype).eps * metric.shape[0] * scale
    positive = eigenvalues > cutoff
    roots = torch.where(
        positive,
        torch.sqrt(eigenvalues.clamp_min(0)),
        torch.zeros_like(eigenvalues),
    )
    inverse_roots = torch.where(
        positive,
        torch.rsqrt(eigenvalues.clamp_min(torch.finfo(metric.dtype).tiny)),
        torch.zeros_like(eigenvalues),
    )
    square_root = (eigenvectors * roots.unsqueeze(0)) @ eigenvectors.mT
    inverse_square_root = (
        eigenvectors * inverse_roots.unsqueeze(0)
    ) @ eigenvectors.mT
    transformed = fit_affine_map(
        row_count=row_count,
        input_sum=input_sum,
        target_sum=target_sum @ square_root,
        input_gram=input_gram,
        input_target_gram=input_target_gram @ square_root,
        rank=rank,
    )
    return AffineMap(
        left=transformed.left,
        right=transformed.right @ inverse_square_root,
        bias=transformed.bias @ inverse_square_root,
    )
