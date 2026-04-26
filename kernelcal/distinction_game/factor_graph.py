"""
Discrete factor-graph backend for semantic SceneGraph collapse.

This module keeps PR-4 deliberately small and dependency-free: finite
categorical variables, log-domain factor tables, and loopy sum-product
belief propagation. Continuous SLAM variables can be coupled later by a
GTSAM/Hydra adapter, but the semantic layer already fits naturally here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .q_s import ConfusionMatrix
from .region import KernelClaim
from .taxonomy import Taxonomy


VarId = Hashable


def _as_log_prob(arr: Sequence[float], *, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(arr, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"probability vector must be 1-D; got shape {p.shape}")
    if (p < 0).any():
        raise ValueError("probability vector contains negative entries")
    s = float(p.sum())
    if s <= 0.0:
        p = np.full(p.shape[0], 1.0 / p.shape[0], dtype=np.float64)
    else:
        p = p / s
    return np.log(np.maximum(p, eps))


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    """Small local logsumexp to avoid a hard scipy dependency."""
    if a.size == 0:
        return np.asarray(-np.inf)
    m = np.max(a, axis=axis, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    out = safe_m + np.log(np.sum(np.exp(a - safe_m), axis=axis, keepdims=True))
    out = np.where(np.isfinite(m), out, -np.inf)
    if axis is None:
        return np.asarray(out.squeeze())
    return np.squeeze(out, axis=axis)


def _normalise_log_message(msg: np.ndarray) -> np.ndarray:
    z = _logsumexp(msg)
    if not np.isfinite(z):
        return np.full(msg.shape, -math.log(msg.size), dtype=np.float64)
    return msg - z


@dataclass(frozen=True)
class Variable:
    """A finite categorical variable in the semantic graph."""

    id: VarId
    n_states: int
    prior: Optional[np.ndarray] = None

    @property
    def log_prior(self) -> np.ndarray:
        if self.prior is None:
            return np.full(self.n_states, -math.log(self.n_states), dtype=np.float64)
        return _as_log_prob(self.prior)


@dataclass(frozen=True)
class Factor:
    """A log-potential table over one or more variables."""

    variables: Tuple[VarId, ...]
    log_table: np.ndarray
    name: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("factor must touch at least one variable")
        table = np.asarray(self.log_table, dtype=np.float64)
        if table.ndim != len(self.variables):
            raise ValueError(
                f"factor over {len(self.variables)} vars needs {len(self.variables)} "
                f"dimensions; got shape {table.shape}"
            )
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "log_table", table.copy())


class UnaryPerceptualFactor(Factor):
    """Unary likelihood from distinction-kernel claims."""

    def __init__(
        self,
        variable: VarId,
        claims: Sequence[KernelClaim],
        *,
        q_s_table: Mapping[str, ConfusionMatrix],
        lambdas: Mapping[str, float],
        taxonomy: Taxonomy,
        eps: float = 1e-12,
        name: Optional[str] = None,
    ) -> None:
        log_p = np.zeros(taxonomy.n, dtype=np.float64)
        for claim in claims:
            qs = q_s_table.get(claim.source_id)
            if qs is None:
                continue
            lam = float(lambdas.get(claim.source_id, 0.0))
            if lam == 0.0:
                continue
            try:
                row_log = qs.log_likelihood_row(claim.native_label, eps=eps)
            except KeyError:
                continue
            log_p = log_p + (lam * float(claim.score)) * row_log
        super().__init__(
            variables=(variable,),
            log_table=log_p,
            name=name or f"unary:{variable}",
            metadata={"n_claims": len(claims)},
        )


def taxonomy_distance_table(taxonomy: Taxonomy) -> np.ndarray:
    """Return a small Potts-like category distance matrix.

    Same category costs 0, same super-class costs 0.5, unrelated costs
    1.0. Unknown is intentionally not grouped unless the taxonomy says so.
    """
    k = taxonomy.n
    out = np.ones((k, k), dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    for i, ci in enumerate(taxonomy.categories):
        gi = taxonomy.super_class_of(ci)
        if gi is None:
            continue
        for j, cj in enumerate(taxonomy.categories):
            if i != j and taxonomy.super_class_of(cj) == gi:
                out[i, j] = 0.5
    return out


class PairwiseSpatialFactor(Factor):
    """Pairwise smoothness on adjacency / CityGraph edges."""

    def __init__(
        self,
        a: VarId,
        b: VarId,
        *,
        taxonomy: Taxonomy,
        weight: float = 1.0,
        beta: float = 1.0,
        distance: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        d = taxonomy_distance_table(taxonomy) if distance is None else np.asarray(distance)
        if d.shape != (taxonomy.n, taxonomy.n):
            raise ValueError(f"distance table must be {(taxonomy.n, taxonomy.n)}; got {d.shape}")
        super().__init__(
            variables=(a, b),
            log_table=-float(beta) * float(weight) * d,
            name=name or f"spatial:{a}:{b}",
            metadata={"weight": float(weight), "beta": float(beta)},
        )


class PairwiseAssociationFactor(Factor):
    """Soft equality between two observations believed to be one entity."""

    def __init__(
        self,
        a: VarId,
        b: VarId,
        *,
        n_states: int,
        strength: float = 1.0,
        eps: float = 1e-3,
        name: Optional[str] = None,
    ) -> None:
        if not 0.0 <= eps < 1.0:
            raise ValueError("eps must be in [0, 1)")
        table = np.full((n_states, n_states), -np.inf if eps == 0.0 else math.log(eps), dtype=np.float64)
        same = 0.0 if eps == 0.0 else math.log(max(1.0 - eps, 1e-12))
        np.fill_diagonal(table, same)
        table *= float(strength)
        super().__init__(
            variables=(a, b),
            log_table=table,
            name=name or f"assoc:{a}:{b}",
            metadata={"strength": float(strength), "eps": float(eps)},
        )


class PairwiseTemporalFactor(Factor):
    """Persistence prior across repeated observations in time."""

    def __init__(
        self,
        a: VarId,
        b: VarId,
        *,
        n_states: int,
        alpha: float = 0.95,
        name: Optional[str] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        off = (1.0 - alpha) / max(1, n_states - 1)
        table = np.full((n_states, n_states), math.log(max(off, 1e-12)), dtype=np.float64)
        np.fill_diagonal(table, math.log(alpha))
        super().__init__(
            variables=(a, b),
            log_table=table,
            name=name or f"temporal:{a}:{b}",
            metadata={"alpha": float(alpha)},
        )


@dataclass
class FactorGraph:
    variables: Dict[VarId, Variable] = field(default_factory=dict)
    factors: List[Factor] = field(default_factory=list)

    def add_variable(self, var_id: VarId, n_states: int, prior: Optional[Sequence[float]] = None) -> Variable:
        if var_id in self.variables:
            var = self.variables[var_id]
            if var.n_states != n_states:
                raise ValueError(f"variable {var_id!r} already has {var.n_states} states")
            return var
        var = Variable(var_id, int(n_states), None if prior is None else np.asarray(prior, dtype=np.float64))
        self.variables[var_id] = var
        return var

    def add_factor(self, factor: Factor) -> Factor:
        for axis, var_id in enumerate(factor.variables):
            if var_id not in self.variables:
                raise KeyError(f"factor references unknown variable {var_id!r}")
            expected = self.variables[var_id].n_states
            if factor.log_table.shape[axis] != expected:
                raise ValueError(
                    f"factor {factor.name!r} axis {axis} has size "
                    f"{factor.log_table.shape[axis]}, expected {expected}"
                )
        self.factors.append(factor)
        return factor

    def factor_indices_by_variable(self) -> Dict[VarId, List[int]]:
        out: Dict[VarId, List[int]] = {vid: [] for vid in self.variables}
        for i, factor in enumerate(self.factors):
            for vid in factor.variables:
                out[vid].append(i)
        return out


@dataclass(frozen=True)
class BPResult:
    posteriors: Dict[VarId, np.ndarray]
    n_iter: int
    converged: bool
    max_delta: float
    map_energy: float
    history: List[float]


def loopy_bp(
    graph: FactorGraph,
    *,
    max_iter: int = 30,
    damping: float = 0.5,
    tol: float = 1e-4,
) -> BPResult:
    """Run log-domain loopy sum-product belief propagation."""
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")
    var_to_factors = graph.factor_indices_by_variable()
    f_to_v: Dict[Tuple[int, VarId], np.ndarray] = {}
    v_to_f: Dict[Tuple[VarId, int], np.ndarray] = {}
    for fi, factor in enumerate(graph.factors):
        for vid in factor.variables:
            n = graph.variables[vid].n_states
            uniform = np.full(n, -math.log(n), dtype=np.float64)
            f_to_v[(fi, vid)] = uniform.copy()
            v_to_f[(vid, fi)] = uniform.copy()

    history: List[float] = []
    converged = False
    max_delta = float("inf")
    for it in range(1, max_iter + 1):
        # Variable -> factor: prior plus all incoming factor messages except target.
        for vid, fis in var_to_factors.items():
            base = graph.variables[vid].log_prior
            total = base.copy()
            for fi in fis:
                total = total + f_to_v[(fi, vid)]
            for fi in fis:
                msg = total - f_to_v[(fi, vid)]
                v_to_f[(vid, fi)] = _normalise_log_message(msg)

        max_delta = 0.0
        new_f_to_v: Dict[Tuple[int, VarId], np.ndarray] = {}
        for fi, factor in enumerate(graph.factors):
            table = factor.log_table.copy()
            for axis, vid in enumerate(factor.variables):
                shape = [1] * table.ndim
                shape[axis] = graph.variables[vid].n_states
                table = table + v_to_f[(vid, fi)].reshape(shape)
            for axis, vid in enumerate(factor.variables):
                incoming_shape = [1] * table.ndim
                incoming_shape[axis] = graph.variables[vid].n_states
                without_target = table - v_to_f[(vid, fi)].reshape(incoming_shape)
                sum_axes = tuple(i for i in range(table.ndim) if i != axis)
                msg = _logsumexp(without_target, axis=sum_axes)
                msg = _normalise_log_message(msg)
                old = f_to_v[(fi, vid)]
                damped = damping * old + (1.0 - damping) * msg
                damped = _normalise_log_message(damped)
                delta = float(np.max(np.abs(damped - old)))
                max_delta = max(max_delta, delta)
                new_f_to_v[(fi, vid)] = damped
        f_to_v = new_f_to_v
        history.append(max_delta)
        if max_delta < tol:
            converged = True
            break

    posteriors: Dict[VarId, np.ndarray] = {}
    for vid, var in graph.variables.items():
        belief = var.log_prior.copy()
        for fi in var_to_factors[vid]:
            belief = belief + f_to_v[(fi, vid)]
        belief = _normalise_log_message(belief)
        posteriors[vid] = np.exp(belief)

    map_assignment = {vid: int(np.argmax(p)) for vid, p in posteriors.items()}
    energy = 0.0
    for factor in graph.factors:
        idx = tuple(map_assignment[vid] for vid in factor.variables)
        energy += float(factor.log_table[idx])

    return BPResult(
        posteriors=posteriors,
        n_iter=len(history),
        converged=converged,
        max_delta=max_delta,
        map_energy=energy,
        history=history,
    )


__all__ = [
    "BPResult",
    "Factor",
    "FactorGraph",
    "PairwiseAssociationFactor",
    "PairwiseSpatialFactor",
    "PairwiseTemporalFactor",
    "UnaryPerceptualFactor",
    "VarId",
    "Variable",
    "loopy_bp",
    "taxonomy_distance_table",
]
