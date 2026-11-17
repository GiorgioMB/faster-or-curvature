"""Curvature with edge lengths: kappa(i, j) = 1 - W1(m_i, m_j) / d(i, j), with d the shortest-path
distance of positive edge lengths and m_u any one-step measure of :func:`orcut.curvature`.

The minimum-cut formulation needs integral potentials. With integer lengths the Kantorovich dual
keeps an integral optimum, with potentials in {0, ..., D} for D the largest distance between the
supports, so the transport problem is still one minimum cut, on D - 1 layers (the layered closure
of :mod:`orcut._layered`; tested against the native solver). The native solver runs the same
dual as the primal-dual method: one maximum flow (minimum cut) per level of the potential, found by
a Dijkstra on reduced costs, in exact integer arithmetic. With unit lengths there are at most three
levels. Real lengths have no integral optimum: the solver then runs in floating point, with one
level per distinct value of the potential.

``truncation=k`` keeps k decimal digits of every length and drops the rest: the lengths become the
integers floor(length * 10^k), the metric is integral again and the result is an exact fraction.
Every length decreases by less than 10^-k, so every distance decreases by a factor of at most
1 - 10^-k / (smallest length), and W1 and d(i, j) with it. Truncation buys exactness, not speed:
on the graphs we timed, 5 digits ran 1.1x faster than full precision and 1 digit 1.2-1.3x.
"""
from __future__ import annotations

import numpy as np

from . import _backend as B
from .graph import Graph
from .measures import parse_spec

__all__ = ["edge_values", "truncate_lengths"]


def truncate_lengths(lengths, digits: int) -> np.ndarray:
    """``floor(length * 10**digits)`` as int64: every length with ``digits`` decimal digits kept and
    the rest dropped, in units of 10**-digits. A length that truncates to 0 is an error."""
    if isinstance(digits, (bool, np.bool_)) or not isinstance(digits, (int, np.integer)):
        raise ValueError("truncation must be a number of decimal digits (an int >= 0) or None")
    k = int(digits)
    if not 0 <= k <= 15:
        raise ValueError("truncation must be between 0 and 15 decimal digits")
    a = np.asarray(lengths, dtype=np.float64)
    q = np.floor(a * (10.0 ** k) * (1.0 + 1e-12))  # 1e-12: decimal values stored just below an integer
    if q.size and q.min() < 1:
        raise ValueError(f"truncation={k} maps the length {a.min():.6g} to 0: keep more digits")
    if q.size and q.max() >= 2.0 ** 53:
        raise ValueError(f"truncation={k} gives lengths beyond 2^53: keep fewer digits")
    return q.astype(np.int64)


def edge_values(G, attr: str, default=None) -> np.ndarray:
    """The edge attribute ``attr`` of a networkx graph, aligned with ``Graph.from_networkx(G)[0].edges``
    (self-loops are dropped, as in :class:`orcut.Graph`)."""
    pos = {x: k for k, x in enumerate(G.nodes())}
    vals = {}
    for a, b, d in G.edges(data=True):
        u, v = pos[a], pos[b]
        if u == v:
            continue
        if attr not in d and default is None:
            raise KeyError(f"edge ({a!r}, {b!r}) has no attribute {attr!r}")
        x = d.get(attr, default)
        key = (u, v) if u < v else (v, u)
        if key in vals and vals[key] != x:
            raise ValueError(f"parallel edges ({a!r}, {b!r}) with different {attr!r}")
        vals[key] = x
    g, _ = Graph.from_networkx(G)
    return np.array([vals[(int(u), int(v))] for u, v in g.edges])


def _on_csr(g: Graph, vals) -> np.ndarray:
    vals = np.asarray(vals)
    if vals.ndim != 1 or vals.shape[0] != g.num_edges:
        raise ValueError(f"lengths need one value per edge of graph.edges ({g.num_edges}), got shape {vals.shape}")
    src = np.repeat(np.arange(g.num_nodes, dtype=np.int64), g.deg)
    return vals[g.edge_ids(np.stack([src, g.indices], axis=1))]


def length_curvature(g, measure, ids, lengths, truncation, n_jobs, exact_fraction, diagnostics):
    if not B.available():
        raise ImportError("edge lengths need the native backend (orcut._native): install a wheel of "
                          "faster-or-curvature, or build it from source with a Rust toolchain")
    lens = np.asarray(lengths)
    if not (np.issubdtype(lens.dtype, np.integer) or np.issubdtype(lens.dtype, np.floating)):
        lens = lens.astype(np.float64)
    lc = _on_csr(g, lens)
    if lc.size and not (np.all(np.isfinite(lc)) and lc.min() > 0):
        raise ValueError("edge lengths must be positive and finite")
    spec = parse_spec(measure, g, lengths=True)
    scale = 1
    if truncation is not None:
        lc, integral, scale = truncate_lengths(lc, truncation), True, 10 ** int(truncation)
    elif np.issubdtype(lc.dtype, np.integer) or (np.all(lc == np.floor(lc)) and lc.max(initial=1) < 2.0 ** 53):
        lc, integral = lc.astype(np.int64), True
    else:
        lc, integral = lc.astype(np.float64), False
    val, num, den, dist_i, w1, dist, aug, rows, cols = B.lengths(g, ids, spec, lc, integral, B.threads_for(n_jobs))
    if spec.lly2:
        val = 2.0 * val
    if not (exact_fraction or diagnostics):
        return val
    out = {"curvature": val}
    if exact_fraction:
        if integral:
            out.update(w1_num=num, w1_den=den, dist=dist_i, length_scale=scale)
        else:
            out.update(w1=w1, dist=dist)
    if diagnostics:
        out.update(augmentations=aug, rows=rows, cols=cols)
    return out
