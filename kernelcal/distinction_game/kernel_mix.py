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
from typing import Callable, Dict, List, Mapping, Optional, Sequence

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
# Lagrange fit (PR-3 stub — keeps the public signature stable)
# ---------------------------------------------------------------------------

def fit_kernel_mix(
    sources: Sequence[str],
    q_s_table: Mapping[str, ConfusionMatrix],
    supervised_pairs: Optional[Sequence] = None,
    constraints: Optional[Mapping[str, object]] = None,
    *,
    taxonomy: Optional[Taxonomy] = None,
    fallback: str = "uniform",
) -> KernelMixFit:
    """Fit per-source weights ``λ̂`` against MaxCal constraints (§4).

    PR-1 returns the uniform-``λ`` baseline regardless of the input;
    the signature is fixed now so the orchestrator and tests can
    bind to it without rework when PR-3 lands the real fit.

    Parameters
    ----------
    sources
        Ordered list of source ids to fit over.
    q_s_table
        Confusion matrices keyed by source id. Every entry in
        ``sources`` must appear here, with all entries sharing the
        same taxonomy.
    supervised_pairs
        Iterable of ``(KernelClaim, true_category_index)`` pairs used
        for the §4 OSM-anchor agreement constraint. Ignored in PR-1.
    constraints
        Optional dict carrying constraint hyper-parameters
        (``mu_osm``, ``hs_distance_bound``, ...). Ignored in PR-1.
    taxonomy
        Optional override; defaults to whatever the q_s_table says.
    fallback
        ``"uniform"`` (default) — what to return until the real fit
        lands.
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

    if fallback != "uniform":
        raise NotImplementedError(
            f"fit_kernel_mix fallback={fallback!r} not implemented in "
            f"PR-1; use 'uniform' or wait for PR-3"
        )

    fit = uniform_lambdas(sources, taxonomy)
    return KernelMixFit(
        sources=fit.sources,
        lambdas=fit.lambdas,
        taxonomy=fit.taxonomy,
        residuals=fit.residuals,
        converged=fit.converged,
        method="uniform",
        metadata={
            "n_sources": len(sources),
            "n_supervised_pairs": len(list(supervised_pairs)) if supervised_pairs else 0,
            "constraints_supplied": bool(constraints),
            "note": (
                "PR-1 returns uniform lambdas regardless of supervised "
                "pairs or constraints. Replace by PR-3 Lagrange fit."
            ),
        },
    )
