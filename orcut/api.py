"""Python entry points of the solver."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import numpy as np

from .graph import Graph

__all__ = ["as_graph", "threads"]


@contextmanager
def threads(n_jobs: Optional[int]):
    """Temporarily set the number of Numba threads (scikit-learn semantics:
    None = Numba default, -1 = all, k > 0 = k threads, k < -1 = all but |k| - 1)."""
    import numba

    prev = numba.get_num_threads()
    if n_jobs is None:
        k = prev
    else:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        cpus = numba.config.NUMBA_NUM_THREADS
        k = cpus if n_jobs == -1 else (n_jobs if n_jobs > 0 else max(1, cpus + 1 + n_jobs))
        k = max(1, min(k, cpus))
    numba.set_num_threads(k)
    try:
        yield
    finally:
        numba.set_num_threads(prev)


def as_graph(graph) -> Graph:
    """A :class:`Graph` from a Graph, a networkx graph, a PyG ``Data``, or an edge array."""
    if isinstance(graph, Graph):
        return graph
    if hasattr(graph, "edge_index"):
        return Graph.from_edge_index(graph.edge_index, getattr(graph, "num_nodes", None))
    if hasattr(graph, "nodes") and hasattr(graph, "edges") and callable(getattr(graph, "edges")):
        g, _ = Graph.from_networkx(graph)
        return g
    return Graph.from_edges(graph)


def curvature(graph, measure="lazy", edges: Optional[np.ndarray] = None, n_jobs: Optional[int] = None,
              exact_fraction: bool = False, diagnostics: bool = False, flags: Optional[int] = None, *,
              lengths=None, truncation=None, backend: str = "auto"):
    """Exact curvature 1 - W1(m_i, m_j) / d(i, j) of the undirected edges of ``graph`` (all, or the
    edge ids ``edges``), aligned with ``graph.edges``; d = 1 without ``lengths``.

    ``measure``: ``"lazy"`` (default, idleness 1/(deg+1)), ``"nonlazy"``, ``"lly"`` (Lin--Lu--Yau),
    a number or Fraction p in [0, 1) (uniform idleness p), or a :class:`orcut.measures.Walk`
    (weighted walk: integer weights exactly, real weights in floating point, any idleness or
    ``"lly"``). With ``exact_fraction`` the result also contains W1 as an integer fraction
    (numerator, denominator) per edge (for ``"lly"``, curvature = (1 - W1) (deg_max + 1) / deg_max);
    with ``diagnostics`` the per-edge diagnostics of :mod:`orcut._core` (``N_DIAG`` columns).
    ``flags`` switches off engineering steps (ablations; values do not change).

    ``lengths``: positive edge lengths, one per edge of ``graph.edges`` (or, for a networkx graph, an
    edge attribute name); see :mod:`orcut.weighted`. Integer lengths are exact, real lengths run in
    floating point, and ``truncation=k`` keeps k decimal digits of real lengths (the result is then
    exact on the truncated metric). With ``exact_fraction`` the result has
    ``w1_num``, ``w1_den``, ``dist`` (integer lengths; curvature = 1 - w1_num / (w1_den dist), in
    units of 1 / ``length_scale``) or ``w1`` and ``dist`` (real lengths); with ``diagnostics`` the
    network-simplex pivots, rows and columns of the reduced transportation problem per edge.

    ``backend``: ``"auto"`` (native when installed, unless ``diagnostics`` or ``flags`` ask for the
    numba kernel), ``"native"`` or ``"numba"``; the two give the same values bit for bit. Returns an
    array, or a dict when extra outputs are requested.
    """
    from dataclasses import replace

    from . import _backend as B
    from . import _core as C
    from .measures import M_WALKF, Walk, parse_spec
    from .weighted import edge_values, length_curvature

    if backend not in ("auto", "native", "numba"):
        raise ValueError("backend must be 'auto', 'native' or 'numba'")
    g = as_graph(graph)
    if isinstance(measure, Walk) and isinstance(measure.weights, str):
        measure = replace(measure, weights=edge_values(graph, measure.weights))
    if isinstance(lengths, str):
        lengths = edge_values(graph, lengths)
    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    if lengths is not None:
        if backend == "numba":
            raise ValueError("edge lengths are solved by the native backend only")
        if flags is not None:
            raise ValueError("flags are the ablations of the numba kernel of the hop metric")
        return length_curvature(g, measure, ids, lengths, truncation, n_jobs, exact_fraction, diagnostics)
    if truncation is not None:
        raise ValueError("truncation applies to edge lengths (lengths=...)")
    spec = parse_spec(measure, g)
    numba_only = diagnostics or flags is not None
    if backend == "native":
        if not B.available():
            raise ImportError("the native backend (orcut._native) is not installed")
        if numba_only:
            raise ValueError("diagnostics and flags belong to the numba kernel: use backend='numba'")
    use_native = backend == "native" or (backend == "auto" and B.available() and not numba_only)
    if spec.mode == M_WALKF and not use_native:
        raise ValueError("real walk weights need the native backend (or integer weights for the numba kernel)")
    diag = None
    if use_native:
        val, num, den, _ = B.hop(g, ids, spec, B.threads_for(n_jobs))
    else:
        import numba

        with threads(n_jobs):
            val, nd, diag = C.core_ids(g.indptr, g.indices, g.edges, ids, spec.mode, spec.pn, spec.pd,
                                       C.F_ALL if flags is None else flags, 8 * numba.get_num_threads(), spec.tabs)
        num, den = nd[:, 0], nd[:, 1]
    if spec.lly2:
        val = 2.0 * val
    if not (exact_fraction or diagnostics):
        return val
    out = {"curvature": val}
    if exact_fraction:
        out["w1_num"] = num
        out["w1_den"] = den
    if diagnostics:
        out["diagnostics"] = diag
    return out


def ball_curvature(graph, radius: int = 2, edges: Optional[np.ndarray] = None, n_jobs: Optional[int] = None,
                   certify: bool = False):
    """Curvature for m_u uniform on the ball of radius ``radius`` around u (Ollivier's ball
    measures), exact, by the layered closure with D = 2 radius + 1 (reference implementation:
    explicit distances). With ``certify`` the diagnostics report, per edge, whether the plan and the
    potentials of the minimum cut were verified (column 5: 1 = verified)."""
    from . import _layered as LY

    g = as_graph(graph)
    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    import numba

    with threads(n_jobs):
        val, diag = LY.layered_ids(g.indptr, g.indices, g.edges, ids, LY.K_BALL, radius, certify,
                                   8 * numba.get_num_threads())
    return (val, diag) if certify else val


def walk_curvature(graph, steps: int = 2, lazy: bool = True, edges: Optional[np.ndarray] = None,
                   n_jobs: Optional[int] = None, certify: bool = False):
    """Curvature for the ``steps``-step random walk (lazy: uniform on N[v] at every step), by the
    layered closure with D = 2 steps + 1; masses are rounded to 2^-40 (error of W1 at most
    D (|B_i| + |B_j|) 2^-40)."""
    from . import _layered as LY

    g = as_graph(graph)
    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    import numba

    with threads(n_jobs):
        val, diag = LY.layered_ids(g.indptr, g.indices, g.edges, ids, LY.K_WALK_LAZY if lazy else LY.K_WALK,
                                   steps, certify, 8 * numba.get_num_threads())
    return (val, diag) if certify else val


def affected_edges(graph, pairs) -> np.ndarray:
    """Ids of the edges of ``graph`` whose curvature (any measure of :func:`curvature`) can differ
    from its value before the vertex pairs ``pairs`` were inserted or deleted (``graph`` is the
    graph after the changes): the edges with an endpoint in the union of the closed
    neighbourhoods N[a] u N[b] of the changed pairs (a, b). The curvature of (i, j) depends only
    on the measures of i and j and on the distances within N[i] u N[j], which are 1 on edges, 2 on
    pairs with a common neighbour, and 3 otherwise; changing (a, b) alters the measures of a and b
    and the distances of the pairs that contain a or b, so only edges with a or b in N[i] u N[j]
    can change. Recompute these (``curvature(graph, edges=affected_edges(graph, pairs))``) to
    update the curvature of all edges exactly after a batch of changes."""
    g = as_graph(graph)
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    mask = np.zeros(g.num_nodes, bool)
    verts = np.unique(p.ravel())
    verts = verts[(verts >= 0) & (verts < g.num_nodes)]
    mask[verts] = True
    for x in verts:
        mask[g.indices[g.indptr[x]:g.indptr[x + 1]]] = True
    return np.flatnonzero(mask[g.edges[:, 0]] | mask[g.edges[:, 1]]).astype(np.int64)
