"""Measure specifications for :func:`orcut.curvature`.

One-step measures m_u on N[u] = {u} u N(u) (the hop distance is the cost):

* ``"lazy"``     uniform on N[u] (idleness 1/(deg(u)+1));
* ``"nonlazy"``  uniform on N(u);
* ``"lly"``      Lin--Lu--Yau curvature (exact, through idleness 1/(deg_max+1));
* a number p in [0, 1) or a :class:`fractions.Fraction`: uniform idleness p;
* :class:`Walk`: idleness a_u/b_u at u and the rest split over N(u) in proportion to positive
  weights (for example edge multiplicities, conductances, or similarities).

With integer weights all masses are rational, and the solver works with them exactly: every edge
is scaled by the least common denominator of its two measures, which must stay below 2^50
(otherwise the value is NaN and the diagnostics flag the edge). Real weights need no truncation:
the cost is still the hop distance, so the minimum cut is the same and the weights only enter as
capacities; they are solved in floating point by the native backend.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import gcd
from typing import Optional, Union

import numpy as np

from . import _core as C
from .graph import Graph

__all__ = ["Walk", "Spec", "parse", "parse_spec", "walk_tables", "walk_tables_float", "M_WALKF"]

M_WALKF = 5  # real walk weights: float masses (native backend only)

_MAXD = (1 << 62) - 1


@dataclass(frozen=True)
class Walk:
    """A weighted one-step walk.

    weights : positive numbers, one per undirected edge (aligned with ``graph.edges``) or one per
        directed CSR entry (aligned with ``graph.indices``); ``None`` means all ones; with a
        networkx graph, also the name of an edge attribute. Integer weights give exact integer
        masses; real weights give float masses (native backend).
    idleness : ``"lazy"`` (1/(deg+1)), a number or Fraction (the same for every vertex), an array
        of Fractions / floats (one per vertex), or ``"lly"``: the Lin--Lu--Yau curvature of the
        walk, computed at idleness 1/2 and doubled (kappa_p is linear in p on [1/2, 1) for every
        walk and every metric). With integer weights, floats are converted with
        ``Fraction(x).limit_denominator(max_den)``.
    """

    weights: Optional[np.ndarray] = None
    idleness: Union[str, float, Fraction, np.ndarray] = "lazy"
    max_den: int = 10_000


def _frac(x, max_den):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    return Fraction(float(x)).limit_denominator(max_den)


def walk_tables(g: Graph, walk: Walk):
    """(wts, Dp, nself, coef, gdv): directed weights aligned with ``g.indices`` and, per vertex u,
    the reduced common denominator Dp[u] of m_u, the numerator of m_u(u), the complement
    b_u - a_u of the idleness a_u/b_u, and the divisor that reduces the neighbour numerators:
    m_u(u) = nself[u] / Dp[u] and m_u(x) = (coef[u] * w(u, x) / gdv[u]) / Dp[u]."""
    n = g.num_nodes
    if walk.weights is None:
        wts = np.ones(g.indices.shape[0], np.int64)
    else:
        w = np.asarray(walk.weights)
        if not np.issubdtype(w.dtype, np.integer):
            if not np.all(np.equal(np.mod(w, 1), 0)):
                raise ValueError("Walk.weights must be integers (quantize real weights first)")
            w = w.astype(np.int64)
        w = w.astype(np.int64)
        if w.shape[0] == g.num_edges:
            src = np.repeat(np.arange(n, dtype=np.int64), g.deg)
            ids = g.edge_ids(np.stack([src, g.indices], axis=1))
            wts = w[ids]
        elif w.shape[0] == g.indices.shape[0]:
            wts = w.copy()
        else:
            raise ValueError("Walk.weights needs one value per undirected edge or per directed CSR entry")
    if np.any(wts <= 0):
        raise ValueError("Walk.weights must be positive")
    if isinstance(walk.idleness, str):
        if walk.idleness != "lazy":
            raise ValueError("idleness must be 'lazy', a number, or an array")
        idle = [Fraction(1, int(d) + 1) for d in g.deg]
    elif np.ndim(walk.idleness) == 0:
        f = _frac(walk.idleness, walk.max_den)
        idle = [f] * n
    else:
        arr = list(walk.idleness)
        if len(arr) != n:
            raise ValueError("one idleness per vertex")
        idle = [_frac(x, walk.max_den) for x in arr]
    Dp = np.empty(n, np.int64)
    nself = np.empty(n, np.int64)
    coef = np.empty(n, np.int64)
    gdv = np.empty(n, np.int64)
    for u in range(n):
        a = idle[u]
        if not (0 <= a < 1):
            raise ValueError(f"idleness of vertex {u} must lie in [0, 1)")
        an, ad = a.numerator, a.denominator
        nb = wts[g.indptr[u]:g.indptr[u + 1]]
        Wu = int(nb.sum())
        gw = 0
        for x in nb:
            gw = gcd(gw, int(x))
        D = ad * Wu
        s = an * Wu
        gg = gcd(gcd(s, (ad - an) * gw), D) if D else 1
        if D // gg > _MAXD or (ad - an) * int(nb.max(initial=1)) > _MAXD:
            raise ValueError(f"vertex {u}: denominators do not fit in 64-bit integers")
        Dp[u] = D // gg
        nself[u] = s // gg
        coef[u] = ad - an
        gdv[u] = gg
    return wts.astype(np.int64), Dp, nself, coef, gdv


def parse(measure, g: Graph):
    """(mode, pn, pd, walk tables or None) for a measure specification."""
    if isinstance(measure, Walk):
        return C.M_WALK, 0, 1, walk_tables(g, measure)
    if isinstance(measure, str):
        key = measure.lower().replace("-", "").replace("_", "")
        if key in ("lazy", "or", "cor"):
            return C.M_LAZY, 0, 1, None
        if key in ("nonlazy", "or0", "cor0"):
            return C.M_NONLAZY, 0, 1, None
        if key in ("lly", "linluyau", "clly"):
            return C.M_LLY, 0, 1, None
        raise ValueError(f"unknown measure {measure!r}")
    p = _frac(measure, 10_000)
    if not (0 <= p < 1):
        raise ValueError("idleness must lie in [0, 1)")
    if p == 0:
        return C.M_NONLAZY, 0, 1, None
    return C.M_IDLE, p.numerator, p.denominator, None


@dataclass(frozen=True)
class Spec:
    """A parsed measure: kernel mode, uniform idleness pn/pd, walk tables (integer ``tabs`` or float
    ``ftabs``), and ``lly2`` (curvature = 2 (1 - W1 / d): Lin--Lu--Yau through idleness 1/2)."""

    mode: int
    pn: int = 0
    pd: int = 1
    tabs: Optional[tuple] = None
    ftabs: Optional[tuple] = None
    lly2: bool = False


def _integral(w) -> bool:
    if w is None:
        return True
    a = np.asarray(w)
    if np.issubdtype(a.dtype, np.integer):
        return True
    a = a.astype(np.float64)
    return bool(np.all(np.isfinite(a)) and np.all(np.mod(a, 1) == 0) and (a.size == 0 or np.abs(a).max() < 2.0 ** 62))


def _is_lly(x) -> bool:
    return isinstance(x, str) and x.lower().replace("-", "").replace("_", "") in ("lly", "linluyau", "clly")


def walk_tables_float(g: Graph, walk: Walk):
    """(wts, coef, idle) for real weights: directed weights aligned with ``g.indices`` and, per vertex
    u, m_u(u) = idle[u] and m_u(x) = coef[u] * w(u, x), coef[u] = (1 - idle[u]) / sum_x w(u, x)."""
    n = g.num_nodes
    w = np.asarray(walk.weights, dtype=np.float64)
    src = np.repeat(np.arange(n, dtype=np.int64), g.deg)
    if w.shape[0] == g.num_edges:
        wts = w[g.edge_ids(np.stack([src, g.indices], axis=1))]
    elif w.shape[0] == g.indices.shape[0]:
        wts = w.copy()
    else:
        raise ValueError("Walk.weights needs one value per undirected edge or per directed CSR entry")
    if not (np.all(np.isfinite(wts)) and np.all(wts > 0)):
        raise ValueError("Walk.weights must be positive and finite")
    if isinstance(walk.idleness, str):
        if walk.idleness != "lazy":
            raise ValueError("idleness must be 'lazy', 'lly', a number, or an array")
        idle = 1.0 / (g.deg.astype(np.float64) + 1.0)
    elif np.ndim(walk.idleness) == 0:
        idle = np.full(n, float(walk.idleness))
    else:
        idle = np.array([float(x) for x in walk.idleness], dtype=np.float64)
        if idle.shape[0] != n:
            raise ValueError("one idleness per vertex")
    if np.any(idle < 0) or np.any(idle >= 1):
        raise ValueError("idleness must lie in [0, 1)")
    tot = np.bincount(src, weights=wts, minlength=n)
    coef = np.where(tot > 0, (1.0 - idle) / np.where(tot > 0, tot, 1.0), 0.0)
    return wts, coef, idle


def parse_spec(measure, g: Graph, lengths: bool = False) -> Spec:
    """:class:`Spec` of a measure specification (``lengths``: for the length metric, where
    Lin--Lu--Yau is computed at idleness 1/2)."""
    if isinstance(measure, Walk):
        lly = _is_lly(measure.idleness)
        walk = replace(measure, idleness=Fraction(1, 2)) if lly else measure
        if _integral(walk.weights):
            return Spec(C.M_WALK, tabs=walk_tables(g, walk), lly2=lly)
        return Spec(M_WALKF, ftabs=walk_tables_float(g, walk), lly2=lly)
    if lengths and _is_lly(measure):
        return Spec(C.M_IDLE, 1, 2, lly2=True)
    mode, pn, pd, tabs = parse(measure, g)
    return Spec(mode, pn, pd, tabs)
