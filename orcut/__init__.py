"""orcut: exact Ollivier--Ricci curvature with one minimum cut per edge.

>>> import orcut
>>> g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
>>> orcut.curvature(g)                          # lazy curvature (idleness 1/(deg+1)), one value per edge
>>> orcut.curvature(g, "lly")                   # Lin--Lu--Yau; also "nonlazy", a number p, or a Walk
>>> orcut.certify(g, (0, 1)).verified           # optimal plan and potential, checked exactly
>>> orcut.ball_curvature(g, 2)                  # uniform measures on balls of radius 2
>>> orcut.affected_edges(g2, [(0, 3)])          # edges to recompute after changing the pair (0, 3)
>>> orcut.curvature(g, lengths=w, truncation=3) # edge lengths, 3 decimal digits kept (exact min cut)

Core dependencies are numpy and numba; the native backend (``orcut._native``, Rust) is used when
installed.
"""
from .graph import Graph
from .api import affected_edges, as_graph, ball_curvature, curvature, walk_curvature
from .certify import Certificate, certify
from .measures import Walk
from .weighted import edge_values, truncate_lengths
from ._backend import available as native_available

__all__ = ["Graph", "as_graph", "curvature", "ball_curvature", "walk_curvature", "certify", "Certificate", "Walk",
           "affected_edges", "edge_values", "truncate_lengths", "native_available"]
__version__ = "1.3.1"
