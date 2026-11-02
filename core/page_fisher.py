"""Page-Fisher residual router: statistics, objective, and alternating solves.

A routing query q (one row per selected query position and calibration
window) scores the residual-key feature x of every routable page. The
exact-teacher Page-Fisher Gram G_{h,n} of head h and query n summarizes how the
teacher's page distribution responds to perturbations of that score. A rank-r
router stores E_g^T x per token (one encoder per KV group g) and applies U_h^T
to the query, so the proxy score is (q U_h)(E_g^T x) and the score-error map is
M_h = U_h E_g^T - I. The fitted objective is the Fisher quadratic

    loss(E, U) = 0.5 s^2 sum_{h,n} (q_n M_h) G_{h,n} (q_n M_h)^T,

minimized by alternating preconditioned-CG solves of U (per head) and E (per
group) from a spectral initialization.
"""

import logging
from typing import NamedTuple

import torch

from core.als import conjugate_gradient

log = logging.getLogger("basiskv.page_fisher")


class FisherStatistics(NamedTuple):
    queries: torch.Tensor  # [H, N, d]
    grams: torch.Tensor  # [H, N, d, d], symmetric PSD
    head_to_kv_group: torch.Tensor  # [H]
    scaling: float  # score scale, d ** -0.5
    teacher_energy: float  # Page-Fisher energy of the exact residual score

    @property
    def heads(self):
        return int(self.queries.shape[0])

    @property
    def dim(self):
        return int(self.queries.shape[-1])

    @property
    def groups(self):
        return int(self.head_to_kv_group.max()) + 1

    def to(self, device):
        return self._replace(
            queries=self.queries.to(device),
            grams=self.grams.to(device),
            head_to_kv_group=self.head_to_kv_group.to(device),
        )


def pack_grams(grams):
    """Upper triangle of the last two axes."""
    rows, cols = torch.triu_indices(grams.shape[-1], grams.shape[-1], device=grams.device)
    return grams[..., rows, cols]


def unpack_grams(packed, dim):
    rows, cols = torch.triu_indices(dim, dim, device=packed.device)
    grams = packed.new_zeros(*packed.shape[:-1], dim, dim)
    grams[..., rows, cols] = packed
    grams[..., cols, rows] = packed
    return grams


def page_teacher(queries, exact_keys, *, scaling, page_size, excluded_prefix_pages):
    """Teacher page distribution over the routable prefix.

    Returns token probabilities grouped by page [heads, pages, page_size] (zero
    padded to whole pages), the page masses [heads, pages], and the first
    routable token. The pinned prefix pages are removed before normalization.
    """
    first = excluded_prefix_pages * page_size
    exact_keys = exact_keys[first:]
    pages = -(-exact_keys.shape[0] // page_size)
    probabilities = torch.softmax(scaling * queries @ exact_keys.mT, dim=-1)
    probabilities = torch.nn.functional.pad(probabilities, (0, pages * page_size - exact_keys.shape[0]))
    by_page = probabilities.reshape(queries.shape[0], pages, page_size)
    return by_page, by_page.sum(dim=-1), first


def page_gram(page_mass, page_rows):
    """sum_p m_p (rho_p - mu)(rho_p - mu)^T for page representatives [heads, pages, features]."""
    mean = torch.einsum("hp,hpd->hd", page_mass, page_rows)
    weighted = (page_rows - mean.unsqueeze(1)) * page_mass.sqrt().unsqueeze(-1)
    grams = weighted.mT @ weighted
    return 0.5 * (grams + grams.mT)


def residual_fisher_gram(queries, exact_keys, residual_rows, *, scaling, page_size, excluded_prefix_pages):
    """Exact-teacher Page-Fisher Grams of residual-key features, and the teacher energy.

    Page masses and within-page weights come from the exact Key; only the page
    representative uses the residual feature. `residual_rows` is
    [tokens, features], or [heads, tokens, features] for head-specific features.
    The energy is the Page-Fisher energy of the exact residual score, so a
    zero-rank router has normalized loss one. It is zero when the feature width
    differs from the query width; the caller then normalizes with the energy of
    the exact score.
    """
    by_page, page_mass, first = page_teacher(
        queries, exact_keys, scaling=scaling, page_size=page_size, excluded_prefix_pages=excluded_prefix_pages
    )
    residual_rows = residual_rows[..., first:, :]
    heads, pages = page_mass.shape
    tokens, width = residual_rows.shape[-2:]
    if tokens == 0:
        return queries.new_zeros(heads, width, width), 0.0
    residual_rows = torch.nn.functional.pad(residual_rows, (0, 0, 0, pages * page_size - tokens))
    if residual_rows.ndim == 3:
        numerator = torch.einsum("hps,hpsd->hpd", by_page, residual_rows.reshape(heads, pages, page_size, width))
    else:
        numerator = torch.einsum("hps,psd->hpd", by_page, residual_rows.reshape(pages, page_size, width))
    page_rows = numerator / page_mass.clamp_min(torch.finfo(page_mass.dtype).tiny).unsqueeze(-1)
    grams = page_gram(page_mass, page_rows)
    if width != queries.shape[-1]:
        return grams, 0.0
    scores = scaling * torch.einsum("hd,hpd->hp", queries, page_rows)
    centered = scores - (page_mass * scores).sum(dim=-1, keepdim=True)
    return grams, float(0.5 * (page_mass * centered.square()).sum())


def fisher_loss(stats, encoder, query):
    """0.5 s^2 sum_{h,n} (q M_h) G_{h,n} (q M_h)^T with M_h = U_h E_g^T - I."""
    error = query @ encoder[stats.head_to_kv_group].mT
    error = error - torch.eye(stats.dim, device=error.device, dtype=error.dtype)
    projected = torch.einsum("hnk,hki->hni", stats.queries, error)
    quadratic = torch.einsum("hni,hnij,hnj->", projected, stats.grams, projected)
    return float(0.5 * stats.scaling**2 * quadratic)


def _jacobi(diagonal, relative_damping):
    """Absolute damping from the Hessian diagonal, and the matching preconditioner."""
    tiny = torch.finfo(diagonal.dtype).tiny
    damping = relative_damping * max(float(diagonal.abs().mean()), tiny)
    shifted = (diagonal.clamp_min(0) + damping).clamp_min(tiny)
    return damping, lambda value: value / shifted


def solve_query_maps(stats, encoder, *, relative_damping, cg_iterations, cg_tolerance):
    """Per-head query maps U_h [d, r] for fixed encoders."""
    query = encoder.new_empty(stats.heads, stats.dim, encoder.shape[-1])
    records = []
    for head, group in enumerate(stats.head_to_kv_group.tolist()):
        q, grams, e = stats.queries[head], stats.grams[head], encoder[group]
        latent_grams = torch.einsum("ir,nij,js->nrs", e, grams, e)  # E^T G_n E
        rhs = q.mT @ torch.einsum("ni,nij,jr->nr", q, grams, e)

        def operator(value):
            return q.mT @ torch.einsum("nr,nrs->ns", q @ value, latent_grams)

        diagonal = torch.einsum("ni,nr->ir", q.square(), latent_grams.diagonal(dim1=-2, dim2=-1))
        damping, preconditioner = _jacobi(diagonal, relative_damping)
        query[head], record = conjugate_gradient(
            lambda value: operator(value) + damping * value,
            rhs,
            iterations=cg_iterations,
            tolerance=cg_tolerance,
            preconditioner=preconditioner,
        )
        records.append(record)
    return query, records


def solve_encoders(stats, query, *, relative_damping, cg_iterations, cg_tolerance):
    """Per-group encoders E_g [d, r] for fixed query maps."""
    encoder = query.new_empty(stats.groups, stats.dim, query.shape[-1])
    records = []
    for group in range(stats.groups):
        heads = torch.nonzero(stats.head_to_kv_group == group).flatten()
        q, grams = stats.queries[heads], stats.grams[heads]
        codes = torch.einsum("hnk,hkr->hnr", q, query[heads])  # q U_h
        rhs = torch.einsum("hnij,hnj,hnr->ir", grams, q, codes)

        def operator(value):
            projected = torch.einsum("ir,hnr->hni", value, codes)
            return torch.einsum("hnij,hnj,hnr->ir", grams, projected, codes)

        diagonal = torch.einsum("hni,hnr->ir", grams.diagonal(dim1=-2, dim2=-1), codes.square())
        damping, preconditioner = _jacobi(diagonal, relative_damping)
        encoder[group], record = conjugate_gradient(
            lambda value: operator(value) + damping * value,
            rhs,
            iterations=cg_iterations,
            tolerance=cg_tolerance,
            preconditioner=preconditioner,
        )
        records.append(record)
    return encoder, records


def initial_factors(stats, rank):
    """Top Fisher eigenvectors of each group as its encoder; heads start from their group's encoder."""
    encoder = []
    for group in range(stats.groups):
        heads = torch.nonzero(stats.head_to_kv_group == group).flatten()
        gram = stats.grams[heads].sum(dim=(0, 1))
        _, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        encoder.append(vectors[:, -rank:])
    encoder = torch.stack(encoder).float()
    return encoder, encoder[stats.head_to_kv_group]


def fit_router(stats, rank, *, sweeps, relative_damping, cg_iterations, cg_tolerance):
    """Alternating solves from the spectral initialization; returns (encoder, query, info).

    Rank 0 is base-only routing: empty factors whose loss is the teacher energy.
    """
    if rank == 0:
        encoder = stats.queries.new_zeros(stats.groups, stats.dim, 0)
        return encoder, encoder[stats.head_to_kv_group], {
            "sweeps": [], "final_query_maximum_iterations": 0, "final_query_maximum_relative_residual": 0.0,
        }
    options = dict(relative_damping=relative_damping, cg_iterations=cg_iterations, cg_tolerance=cg_tolerance)
    encoder, query = initial_factors(stats, rank)
    history = []
    for sweep in range(1, sweeps + 1):
        before = fisher_loss(stats, encoder, query)
        query, _ = solve_query_maps(stats, encoder, **options)
        after_queries = fisher_loss(stats, encoder, query)
        encoder, _ = solve_encoders(stats, query, **options)
        after = fisher_loss(stats, encoder, query)
        history.append({
            "sweep": sweep,
            "loss_before": before,
            "loss_after_queries": after_queries,
            "loss_after_encoders": after,
        })
        log.info("Page-Fisher sweep %d/%d: loss %.6g -> %.6g", sweep, sweeps, before, after)
    query, records = solve_query_maps(stats, encoder, **options)
    info = {
        "sweeps": history,
        "final_query_maximum_iterations": max(record["iterations"] for record in records),
        "final_query_maximum_relative_residual": max(record["relative_residual"] for record in records),
    }
    return encoder, query, info
