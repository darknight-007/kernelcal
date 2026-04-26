"""
MaxCal kernel mixing (§4 of the design doc).

Implements the per-region category posterior

.. math::

   \\log p(c \\mid r)  \\;=\\; \\text{const}
       + \\sum_s \\lambda_s \\, \\log Q_s\\big(\\hat y_s(r) \\mid c\\big)

where the sum runs over sources that produced a claim on region ``r``.
Sources that *did not* claim contribute no term, so the mixer naturally
handles per-region missing-data — exactly the §4 "OSM is silent
elsewhere" behaviour.

PR-1 ships **uniform** ``λ_s`` (no Lagrange fit). The signature of
:func:`fit_kernel_mix` matches the eventual PR-3 implementation so
callers do not need to change. The :class:`KernelMixFit` returned in
PR-1 carries the uniform weights plus the bookkeeping fields (residuals
zero, ``converged=True``, ``method="uniform"``) so a downstream check
of ``fit.method`` can branch on the fidelity of the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .q_s import ConfusionMatrix
from .region import KernelClaim
from .taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelMixFit:
    """Output of :func:`fit_kernel_mix`.

    Attributes
    ----------
    sources
        Ordered tuple of source ids the mixer was fitted over. The
        ``lambdas`` array is aligned to this order.
    lambdas
        ``(|sources|,)`` non-negative weights, summing to 1 in the
        uniform-prior PR-1 implementation. PR-3 will not require unit
        sum since the §4 likelihood already absorbs the partition
        constant; this convention is purely for human interpretability.
    taxonomy
        Reference back to the taxonomy the fit lives in.
    residuals
        ``(|constraints|,)`` residuals from the Lagrange fit. Zeros in
        PR-1 (no constraints used).
    converged
        ``True`` if the optimiser declared convergence. Always ``True``
        in PR-1 (no optimisation performed).
    method
        ``"uniform"`` (PR-1), ``"lagrange"`` (PR-3), or ``"manual"``
        (caller-supplied lambdas).
    metadata
        Free-form bag for fit diagnostics (e.g. n_supervised_pairs,
        wall-clock, pre/post log-likelihood on a held-out set).
    """

    sources: tuple
    lambdas: np.ndarray
    taxonomy: Taxonomy
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    converged: bool = True
    method: str = "uniform"
    metadata: Dict[str, object] = field(default_factory=dict)

    def lambda_for(self, source_id: str) -> float:
        try:
            return float(self.lambdas[self.sources.index(source_id)])
        except ValueError as exc:
            raise KeyError(
                f"source {source_id!r} not in this fit; sources: "
                f"{list(self.sources)}"
            ) from exc

    # ---- Posterior over c* given the claims for one region -----------

    def log_posterior(
        self,
        claims_for_region: Sequence[KernelClaim],
        q_s_table: Mapping[str, ConfusionMatrix],
        *,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Return the (un-normalised) log-posterior over ``c*`` given a
        list of claims on a single region.

        The unnormalised log-posterior is

        .. math::

           \\log p(c \\mid \\text{claims})
              = \\sum_{s \\in \\text{claims}} \\lambda_s
                \\, \\log Q_s\\big(\\hat y_s \\mid c\\big)

        with claims weighted by the source's :attr:`lambdas` entry.
        Claims from a source missing from :attr:`sources` are silently
        ignored — that's the right behaviour when the orchestrator
        has run a kernel for which no Q_s prior is available.
        """
        log_p = np.zeros(self.taxonomy.n, dtype=np.float64)
        for claim in claims_for_region:
            if claim.source_id not in self.sources:
                continue
            if claim.source_id not in q_s_table:
                continue
            lam = self.lambda_for(claim.source_id)
            qs = q_s_table[claim.source_id]
            try:
                row_log = qs.log_likelihood_row(claim.native_label, eps=eps)
            except KeyError:
                # Source produced a label that's not in its prior
                # vocabulary — treat as uninformative rather than
                # crashing. This protects the orchestrator from
                # native-label drift between kernel versions.
                continue
            # Score-weight: a low-confidence claim should contribute
            # less than a high-confidence one, even from the same
            # source. Linear-in-score is the simplest defensible
            # weighting; PR-3 may swap in a calibrated soft-vote.
            log_p = log_p + (lam * float(claim.score)) * row_log
        return log_p

    def posterior(
        self,
        claims_for_region: Sequence[KernelClaim],
        q_s_table: Mapping[str, ConfusionMatrix],
        *,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Normalised posterior ``p(c | claims)`` as a probability
        vector over :attr:`taxonomy`.

        Returns a uniform distribution if no claim contributed any
        log-evidence (i.e. every claim was for an unknown source or
        an unknown native label).
        """
        log_p = self.log_posterior(claims_for_region, q_s_table, eps=eps)
        log_p = log_p - log_p.max()
        p = np.exp(log_p)
        s = p.sum()
        if s <= 0:
            return np.full(self.taxonomy.n, 1.0 / self.taxonomy.n)
        return p / s

    def to_dict(self) -> Dict[str, object]:
        return {
            "sources": list(self.sources),
            "lambdas": self.lambdas.tolist(),
            "taxonomy": self.taxonomy.name,
            "residuals": self.residuals.tolist(),
            "converged": self.converged,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def uniform_lambdas(
    sources: Sequence[str],
    taxonomy: Taxonomy,
) -> KernelMixFit:
    """Equal-weight mixer: ``λ_s = 1 / |sources|`` for every ``s``.

    The simplest possible §4 fit — useful as a baseline when no
    OSM-anchored supervision is available, and what the PR-1 orchestrator
    uses by default.
    """
    n = len(sources)
    if n == 0:
        raise ValueError("uniform_lambdas requires at least one source")
    return KernelMixFit(
        sources=tuple(sources),
        lambdas=np.full(n, 1.0 / n, dtype=np.float64),
        taxonomy=taxonomy,
        residuals=np.zeros(0),
        converged=True,
        method="uniform",
        metadata={"n_sources": n},
    )


def manual_lambdas(
    lambdas: Mapping[str, float],
    taxonomy: Taxonomy,
) -> KernelMixFit:
    """Caller-supplied weights. Useful for ablations and for
    over-riding individual sources during development.

    Weights are normalised to sum to 1.
    """
    if not lambdas:
        raise ValueError("manual_lambdas requires at least one entry")
    sources = tuple(lambdas)
    arr = np.array([lambdas[s] for s in sources], dtype=np.float64)
    if (arr < 0).any():
        raise ValueError(
            f"manual lambdas must be non-negative; got {lambdas}"
        )
    s = arr.sum()
    if s <= 0:
        raise ValueError("manual lambdas sum to zero")
    return KernelMixFit(
        sources=sources,
        lambdas=arr / s,
        taxonomy=taxonomy,
        residuals=np.zeros(0),
        converged=True,
        method="manual",
        metadata={},
    )


# ---------------------------------------------------------------------------
# Anchor-fit (PR-3 §4): supervised lambda fit on ground-truthed regions
# ---------------------------------------------------------------------------

def _normalise_supervised_pairs(
    supervised_pairs: Sequence,
) -> List:
    """Accept either flat ``(claim, true_c)`` PR-1 tuples or richer
    ``(claims_for_region, true_c)`` PR-3 tuples; emit the latter
    form uniformly.

    Backward-compatible so callers wired to the PR-1 signature don't
    have to change. A single :class:`KernelClaim` is wrapped in a
    one-element list when encountered.
    """
    out: List = []
    for entry in supervised_pairs:
        if not isinstance(entry, tuple) and not isinstance(entry, list):
            raise TypeError(
                "supervised_pairs entries must be tuples; got "
                f"{type(entry).__name__}"
            )
        if len(entry) != 2:
            raise ValueError(
                "supervised_pairs entries must be (claims, true_c) — "
                f"got tuple of length {len(entry)}"
            )
        claims_or_claim, true_c = entry
        if isinstance(claims_or_claim, KernelClaim):
            out.append(([claims_or_claim], int(true_c)))
        else:
            out.append((list(claims_or_claim), int(true_c)))
    return out


def _per_region_log_evidence(
    regions: Sequence,
    sources: Sequence[str],
    q_s_table: Mapping[str, ConfusionMatrix],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return ``E`` of shape ``(n_regions, n_sources, n_categories)``
    where ``E[r, s, c] = score_{r,s} * log Q_s(ŷ_{r,s} | c)`` and
    sources without a claim on region ``r`` contribute zero.

    This is the §4 score-weighted log-evidence tensor; contracting it
    with ``λ`` over ``s`` gives the unnormalised log-posterior over
    ``c`` per region.
    """
    n_regions = len(regions)
    n_sources = len(sources)
    n_cats = next(iter(q_s_table.values())).n_categories
    src_index = {sid: i for i, sid in enumerate(sources)}
    E = np.zeros((n_regions, n_sources, n_cats), dtype=np.float64)
    for r, region_claims in enumerate(regions):
        for claim in region_claims:
            sid = claim.source_id
            i = src_index.get(sid)
            if i is None or sid not in q_s_table:
                continue
            try:
                row_log = q_s_table[sid].log_likelihood_row(claim.native_label, eps=eps)
            except KeyError:
                continue
            E[r, i, :] += float(claim.score) * row_log
    return E


def _log_softmax_rowwise(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _slsqp_lambda_fit(
    log_evidence: np.ndarray,            # (R, S, C)
    soft_targets: np.ndarray,            # (R, C) probability rows
    *,
    prior_lambdas: Optional[np.ndarray] = None,
    max_iter: int = 200,
    ftol: float = 1e-7,
) -> Tuple[np.ndarray, float, bool, int]:
    """Constrained MLE of ``λ`` against soft per-region category targets.

    Maximises::

        Σ_r Σ_c soft_targets[r, c] * (Σ_s λ_s log_evidence[r, s, c]
                                       - logZ_r(λ))

    over the simplex ``λ_s ≥ 0, Σ_s λ_s = 1``. Concave, so SLSQP
    converges from any feasible start.

    Returns ``(lambdas, neg_log_likelihood, converged, n_iter)``.
    """
    try:
        from scipy.optimize import LinearConstraint, minimize  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "fit_kernel_mix anchor-fit requires scipy.optimize"
        ) from exc

    R, S, C = log_evidence.shape
    if soft_targets.shape != (R, C):
        raise ValueError(
            f"soft_targets shape {soft_targets.shape} != ({R}, {C})"
        )

    def neg_ll_and_grad(lam: np.ndarray) -> Tuple[float, np.ndarray]:
        # log_p[r, c] = Σ_s lam_s * log_evidence[r, s, c]
        log_p = np.einsum("s,rsc->rc", lam, log_evidence)
        log_p_n = _log_softmax_rowwise(log_p)
        ll = float((soft_targets * log_p_n).sum())
        # gradient: ∂/∂λ_s of log_p_n[r,c] = log_evidence[r,s,c]
        #                                    - Σ_c' p_n[r,c'] log_evidence[r,s,c']
        # so gradient of ll wrt λ_s is
        # Σ_r Σ_c soft[r,c] (log_evidence[r,s,c] - <log_evidence[r,s,·]>_{p_n[r,·]}).
        p_n = np.exp(log_p_n)                                      # (R, C)
        # E_pn[r, s] = Σ_c p_n[r, c] log_evidence[r, s, c]
        E_pn = np.einsum("rc,rsc->rs", p_n, log_evidence)          # (R, S)
        E_soft = np.einsum("rc,rsc->rs", soft_targets, log_evidence)  # (R, S)
        grad = (E_soft - E_pn).sum(axis=0)                         # (S,)
        return -ll, -grad

    if prior_lambdas is None:
        x0 = np.full(S, 1.0 / S, dtype=np.float64)
    else:
        x0 = np.asarray(prior_lambdas, dtype=np.float64).copy()
        x0 = np.maximum(x0, 1e-9)
        x0 /= x0.sum()
    bounds = [(0.0, 1.0)] * S
    cons = LinearConstraint(np.ones(S), lb=1.0, ub=1.0)
    res = minimize(
        neg_ll_and_grad, x0=x0, jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda v: np.sum(v) - 1.0,
                      "jac": lambda v: np.ones_like(v)}],
        options={"maxiter": max_iter, "ftol": ftol, "disp": False},
    )
    lam = np.asarray(res.x, dtype=np.float64)
    lam = np.maximum(lam, 0.0)
    s = lam.sum()
    if s > 0:
        lam = lam / s
    return lam, float(res.fun), bool(res.success), int(res.nit)


def fit_kernel_mix(
    sources: Sequence[str],
    q_s_table: Mapping[str, ConfusionMatrix],
    supervised_pairs: Optional[Sequence] = None,
    constraints: Optional[Mapping[str, object]] = None,
    *,
    taxonomy: Optional[Taxonomy] = None,
    fallback: str = "uniform",
    method: str = "auto",
    prior_lambdas: Optional[Mapping[str, float]] = None,
    max_iter: int = 200,
    ftol: float = 1e-7,
) -> KernelMixFit:
    """Fit per-source weights ``λ̂`` against the §4 MaxCal anchor likelihood.

    ``supervised_pairs`` carries OSM-anchored (or human-labelled)
    region-level ground truth as ``(claims_for_region, true_c_idx)``
    tuples. Each region's posterior is the score-weighted log-pool

    .. math::

       p(c \\mid \\text{claims}_r)
         \\propto \\exp\\!\\Big(\\sum_s \\lambda_s \\, \\text{score}_{r,s}
                              \\, \\log Q_s(\\hat y_{r,s} \\mid c)\\Big),

    and the fit maximises the resulting log-likelihood under the
    simplex constraint :math:`\\lambda_s \\ge 0, \\sum_s \\lambda_s = 1`.

    Parameters
    ----------
    sources
        Ordered list of source ids the fit lives over. Sources absent
        from any supervised region simply receive their prior weight.
    q_s_table
        Confusion matrices keyed by source id. Every entry in
        ``sources`` must appear here, with all entries sharing the
        same taxonomy.
    supervised_pairs
        Sequence of ``(claims_for_region, true_c)`` pairs (PR-3 form),
        or the legacy PR-1 ``(claim, true_c)`` flat form. Both work.
    constraints
        Reserved for §4 extras (coverage matching, bounded
        disagreement). Currently logged into ``metadata`` only.
    taxonomy
        Optional override; defaults to the q_s_table's shared taxonomy.
    fallback
        Behaviour when no supervised pairs are supplied. ``"uniform"``
        (default) returns equal weights; ``"prior"`` returns the
        ``prior_lambdas`` re-normalised; any other value raises.
    method
        ``"auto"`` (default) — SLSQP if supervised pairs exist, else
        fallback. ``"uniform"`` to force the PR-1 baseline. ``"slsqp"``
        to require a real fit.
    prior_lambdas
        Optional mapping ``source_id -> weight``; used as the SLSQP
        warm-start (renormalised) and as the result for
        ``fallback="prior"``.
    max_iter, ftol
        SLSQP solver knobs.
    """
    if not sources:
        raise ValueError("fit_kernel_mix requires at least one source")
    missing = [s for s in sources if s not in q_s_table]
    if missing:
        raise KeyError(
            f"q_s_table is missing matrices for sources: {missing}"
        )
    if taxonomy is None:
        taxonomy = next(iter(q_s_table.values())).taxonomy
    for sid, qs in q_s_table.items():
        if qs.taxonomy is not taxonomy and qs.taxonomy.name != taxonomy.name:
            raise ValueError(
                f"q_s_table entry {sid!r} uses taxonomy "
                f"{qs.taxonomy.name!r}; expected {taxonomy.name!r}"
            )

    pairs = _normalise_supervised_pairs(list(supervised_pairs or []))
    n_pairs = len(pairs)

    if method == "uniform" or (method == "auto" and n_pairs == 0):
        if fallback == "uniform":
            base = uniform_lambdas(sources, taxonomy)
        elif fallback == "prior":
            if not prior_lambdas:
                raise ValueError("fallback='prior' requires prior_lambdas")
            arr = np.array([float(prior_lambdas.get(s, 0.0)) for s in sources])
            arr = np.maximum(arr, 0.0)
            tot = arr.sum()
            if tot <= 0:
                raise ValueError("prior_lambdas sums to zero")
            base = KernelMixFit(
                sources=tuple(sources),
                lambdas=arr / tot,
                taxonomy=taxonomy,
                residuals=np.zeros(0),
                converged=True,
                method="prior",
                metadata={"n_sources": len(sources)},
            )
        else:
            raise ValueError(
                f"unknown fallback={fallback!r}; expected 'uniform' or 'prior'"
            )
        return KernelMixFit(
            sources=base.sources,
            lambdas=base.lambdas,
            taxonomy=base.taxonomy,
            residuals=base.residuals,
            converged=base.converged,
            method=base.method,
            metadata={
                "n_sources": len(sources),
                "n_supervised_pairs": n_pairs,
                "constraints_supplied": bool(constraints),
                "note": (
                    "no supervised pairs available — returned baseline."
                ),
            },
        )

    if method not in ("auto", "slsqp"):
        raise NotImplementedError(
            f"fit_kernel_mix method={method!r} not implemented; "
            "use 'uniform', 'slsqp', or 'auto'."
        )

    regions = [claims for (claims, _) in pairs]
    true_cs = np.array([c for (_, c) in pairs], dtype=np.int64)
    log_evidence = _per_region_log_evidence(regions, sources, q_s_table)

    # Hard targets: one-hot per region.
    soft_targets = np.zeros((n_pairs, taxonomy.n), dtype=np.float64)
    soft_targets[np.arange(n_pairs), true_cs] = 1.0

    prior_arr = None
    if prior_lambdas is not None:
        prior_arr = np.array(
            [float(prior_lambdas.get(s, 0.0)) for s in sources], dtype=np.float64,
        )
        if prior_arr.sum() <= 0:
            prior_arr = None

    lam, neg_ll, converged, n_iter = _slsqp_lambda_fit(
        log_evidence, soft_targets,
        prior_lambdas=prior_arr, max_iter=max_iter, ftol=ftol,
    )

    return KernelMixFit(
        sources=tuple(sources),
        lambdas=lam,
        taxonomy=taxonomy,
        residuals=np.zeros(0),
        converged=converged,
        method="slsqp",
        metadata={
            "n_sources": len(sources),
            "n_supervised_pairs": n_pairs,
            "constraints_supplied": bool(constraints),
            "neg_log_likelihood": neg_ll,
            "slsqp_iter": n_iter,
        },
    )


# ---------------------------------------------------------------------------
# EM joint fit of (lambdas, Q_s)  — the full PR-3 §4 fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointFitResult:
    """Output of :func:`fit_kernel_mix_em`."""

    mix: KernelMixFit
    q_s_table: Mapping[str, ConfusionMatrix]
    n_iter: int
    log_likelihood_history: List[float]
    converged: bool
    n_anchored_regions: int
    n_unsupervised_regions: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "mix": self.mix.to_dict(),
            "q_s_table": {k: v.to_dict() for k, v in self.q_s_table.items()},
            "n_iter": int(self.n_iter),
            "log_likelihood_history": [float(v) for v in self.log_likelihood_history],
            "converged": bool(self.converged),
            "n_anchored_regions": int(self.n_anchored_regions),
            "n_unsupervised_regions": int(self.n_unsupervised_regions),
        }


def fit_kernel_mix_em(
    regions: Sequence[Sequence[KernelClaim]],
    q_s_table: Mapping[str, ConfusionMatrix],
    *,
    taxonomy: Optional[Taxonomy] = None,
    sources: Optional[Sequence[str]] = None,
    anchors: Optional[Sequence[Optional[int]]] = None,
    prior_lambdas: Optional[Mapping[str, float]] = None,
    fit_q_s: bool = True,
    alpha_q_s: float = 10.0,
    max_iter: int = 20,
    tol: float = 1e-5,
    lambda_max_iter: int = 100,
    lambda_ftol: float = 1e-7,
    e_step: str = "argmax",
) -> JointFitResult:
    """Joint EM fit of ``(λ, Q_s)`` on a list of regions.

    Latent-variable model: each region ``r`` has a true category
    ``c_r`` drawn from a uniform prior; observations are the per-source
    claims with native labels. Anchored regions clamp ``c_r`` to the
    given true category; unsupervised regions get a soft posterior
    ``q_r(c) ∝ exp(Σ_s λ_s score_{r,s} log Q_s(ŷ_{r,s} | c))``.

    The E-step builds those ``q_r(c)`` distributions; the M-step
    splits into

    1. **λ update** — SLSQP MLE against the soft targets ``q_r(c)``.
    2. **Q_s update** — Bayesian-Dirichlet refit on soft pseudo-counts
       ``N_s[r̃, c] = Σ_{r: ŷ_{r,s}=r̃} q_r(c) * score_{r,s}``.

    Parameters
    ----------
    regions
        Sequence of per-region :class:`KernelClaim` lists. Empty
        regions are tolerated but uninformative.
    q_s_table
        Initial / prior confusion matrices. Updated in place is
        avoided — a fresh table is returned.
    taxonomy
        Optional override (defaults to ``q_s_table``'s).
    sources
        Optional ordered source list (defaults to
        ``sorted(q_s_table.keys())``).
    anchors
        Optional list of region anchors. Same length as ``regions``;
        ``None`` for unsupervised regions, ``int`` for clamped true
        category.
    prior_lambdas
        Optional warm-start mapping for the λ MLE.
    fit_q_s
        Whether to update ``Q_s`` each M-step. Set False to fit only
        the mixing weights against fixed priors.
    alpha_q_s
        Dirichlet pseudo-count strength for the Q_s update. Larger
        values trust the prior more. Default 10.0.
    max_iter, tol
        EM outer loop; convergence when log-likelihood improves by
        less than ``tol`` between iterations.
    lambda_max_iter, lambda_ftol
        Inner SLSQP knobs for the λ MLE.
    e_step
        ``"argmax"`` (default/back-compatible) uses the row-wise
        posterior from the existing MaxCal mixture. ``"bp"`` routes the
        same unary evidence through the PR-4 factor-graph engine; today
        that is parity with ``"argmax"`` unless a caller builds richer
        spatial factors upstream, but it exercises the integration path
        used by SceneGraph collapse.

    Returns
    -------
    JointFitResult
    """
    from .q_s_fit import bayesian_refit_q_s_table

    if taxonomy is None:
        taxonomy = next(iter(q_s_table.values())).taxonomy
    if sources is None:
        sources = sorted(q_s_table.keys())
    sources = list(sources)
    src_index = {sid: i for i, sid in enumerate(sources)}
    R = len(regions)
    if R == 0:
        raise ValueError("fit_kernel_mix_em requires at least one region")

    if anchors is None:
        anchors = [None] * R
    if len(anchors) != R:
        raise ValueError(
            f"len(anchors)={len(anchors)} != len(regions)={R}"
        )

    n_anchored = sum(1 for a in anchors if a is not None)
    n_unsup = R - n_anchored
    if e_step not in {"argmax", "bp"}:
        raise ValueError("e_step must be 'argmax' or 'bp'")

    # λ warm-start
    if prior_lambdas is None:
        lam = np.full(len(sources), 1.0 / len(sources), dtype=np.float64)
    else:
        lam = np.array([float(prior_lambdas.get(s, 0.0)) for s in sources], dtype=np.float64)
        lam = np.maximum(lam, 1e-9)
        lam /= lam.sum()

    cur_q_s_table: Dict[str, ConfusionMatrix] = dict(q_s_table)
    history: List[float] = []
    converged = False
    last_ll = -np.inf
    n_iter_done = 0

    for it in range(1, max_iter + 1):
        # ---- E-step: compute log_p[r, c] under current (λ, Q_s) ----
        E = _per_region_log_evidence(regions, sources, cur_q_s_table)  # (R, S, C)
        log_p = np.einsum("s,rsc->rc", lam, E)
        log_p_n = _log_softmax_rowwise(log_p)
        # Soft posterior; anchors clamp.
        if e_step == "bp":
            from .factor_graph import Factor, FactorGraph, loopy_bp

            fg = FactorGraph()
            for r in range(R):
                fg.add_variable(r, taxonomy.n)
                fg.add_factor(Factor((r,), log_p[r, :], name=f"em-unary:{r}"))
            bp = loopy_bp(fg, max_iter=10, damping=0.0, tol=1e-9)
            q = np.vstack([bp.posteriors[r] for r in range(R)])
        else:
            q = np.exp(log_p_n)                                        # (R, C)
        for r, a in enumerate(anchors):
            if a is None:
                continue
            q[r, :] = 0.0
            q[r, int(a)] = 1.0

        # log-likelihood under the current model
        # (anchored regions: log_p_n[r, a]; unsupervised regions: logsumexp via row)
        ll = 0.0
        for r in range(R):
            a = anchors[r]
            if a is not None:
                ll += float(log_p_n[r, int(a)])
            else:
                # logsumexp(log_p[r,:]) is the row's normaliser
                m = log_p[r, :].max()
                ll += float(m + np.log(np.exp(log_p[r, :] - m).sum()))
        history.append(ll)
        n_iter_done = it

        if it > 1 and abs(ll - last_ll) < tol:
            converged = True
            break
        last_ll = ll

        # ---- M-step (a): λ via SLSQP against soft targets q ----
        try:
            lam, _neg_ll, _ok, _nit = _slsqp_lambda_fit(
                E, q, prior_lambdas=lam,
                max_iter=lambda_max_iter, ftol=lambda_ftol,
            )
        except RuntimeError:
            # No scipy — keep current λ; emit a uniform-ish fallback.
            lam = np.full(len(sources), 1.0 / len(sources))

        # ---- M-step (b): Q_s update via soft pseudo-counts ----
        if fit_q_s:
            counts_by_source: Dict[str, np.ndarray] = {
                sid: np.zeros((qm.n_native, taxonomy.n), dtype=np.float64)
                for sid, qm in cur_q_s_table.items()
            }
            for r, region_claims in enumerate(regions):
                for claim in region_claims:
                    sid = claim.source_id
                    if sid not in cur_q_s_table:
                        continue
                    qm = cur_q_s_table[sid]
                    try:
                        ridx = qm.native_index(claim.native_label)
                    except KeyError:
                        continue
                    w = float(claim.score)
                    counts_by_source[sid][ridx, :] += w * q[r, :]
            qs_fit = bayesian_refit_q_s_table(
                cur_q_s_table, counts_by_source, alpha=alpha_q_s,
            )
            cur_q_s_table = dict(qs_fit.table)

    fit = KernelMixFit(
        sources=tuple(sources),
        lambdas=lam,
        taxonomy=taxonomy,
        residuals=np.zeros(0),
        converged=converged,
        method="em",
        metadata={
            "n_sources": len(sources),
            "n_regions": R,
            "n_anchored_regions": n_anchored,
            "n_unsupervised_regions": n_unsup,
            "alpha_q_s": float(alpha_q_s),
            "fit_q_s": bool(fit_q_s),
            "e_step": e_step,
            "log_likelihood_final": history[-1] if history else float("nan"),
            "n_em_iter": int(n_iter_done),
        },
    )
    return JointFitResult(
        mix=fit,
        q_s_table=cur_q_s_table,
        n_iter=int(n_iter_done),
        log_likelihood_history=history,
        converged=converged,
        n_anchored_regions=n_anchored,
        n_unsupervised_regions=n_unsup,
    )
