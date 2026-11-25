# faster-or-curvature

[![PyPI version](https://img.shields.io/pypi/v/faster-or-curvature.svg)](https://pypi.org/project/faster-or-curvature/)
[![Downloads](https://img.shields.io/pepy/dt/faster-or-curvature)](https://pepy.tech/project/faster-or-curvature)
[![License](https://img.shields.io/pypi/l/faster-or-curvature.svg)](https://pypi.org/project/faster-or-curvature/)

Exact Ollivier–Ricci curvature of every edge of a graph, with one minimum cut per edge, in exact
integer arithmetic and without a distance matrix; also for weighted walks and edge lengths. A native
backend (Rust) computes it on all cores with no JIT warm-up.

```bash
pip install faster-or-curvature        # imports as orcut
```

```python
import orcut
g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph / PyG Data
orcut.curvature(g)                      # lazy (idleness 1/(deg+1)), aligned with g.edges (u < v)
orcut.curvature(g, "lly")               # Lin–Lu–Yau; also "nonlazy", or a number p (idleness p)
orcut.curvature(g, orcut.Walk(weights=w, idleness=0.5))   # a weighted walk (any positive weights)
orcut.curvature(g, exact_fraction=True) # also W1 as an exact integer fraction per edge
c = orcut.certify(g, (0, 1))            # an optimal plan and an optimal 1-Lipschitz potential
c.verified, c.verify(g)                 # checked exactly when built, and re-checkable from scratch
orcut.ball_curvature(g, 2)              # Ollivier's ball measures, radius 2 (layered closure)
orcut.walk_curvature(g, 2)              # two-step lazy random walk
orcut.affected_edges(g2, changed_pairs) # after inserting/deleting pairs: the only edges to recompute
orcut.curvature(g, lengths=l)           # edge lengths: 1 - W1 / d(i, j), d the weighted distance
orcut.curvature(g, lengths=l, truncation=3)  # real lengths, 3 decimals kept: exact fractions
orcut.curvature(G, lengths="weight")    # with a networkx graph, an edge attribute name
```

Any two vertices of the closed neighbourhoods of an edge are at distance at most three, so the
Kantorovich dual of its transport problem has an integral optimum with potentials in {0, 1, 2, 3}:
in unary, a maximum-weight closure, solved by one minimum cut whose arcs come from adjacency and
common neighbours only. *Walk weights* only change the capacities of that cut: integer weights are
exact, real weights run in floating point. *Integer edge lengths* keep an integral optimum, so the
transport problem is still one minimum cut, on D − 1 layers; with real lengths, each edge's
transport problem is solved by a network simplex in floating point (`truncation=k` keeps k decimal
digits and is exact on the truncated metric).

Against GraphRicciCurvature 0.6.1 (exact transport, `method="OTD"`, one process) on real weighted
networks, with the same values to 1e-13 on every edge (seconds on one core; lazy walk with idleness
1/2, or its default heat kernel exp(-w^2) with lengths divided by their median):

| network | edges | measure | faster-or-curvature | GraphRicciCurvature |
| --- | ---: | --- | ---: | ---: |
| Les Misérables (length 1/co-occurrences) | 254 | uniform / heat | 0.0043 / 0.0039 | 0.069 / 0.059 |
| OpenFlights airports, 6 nearest neighbours (km) | 11636 | uniform / heat | 0.16 / 0.17 | 3.7 / 3.9 |
| OpenFlights routes (great-circle km) | 18833 | uniform | 5.4 | 14.0 |

The native backend (`orcut._native`) is a port of the numba kernel of `orcut._core` and returns the
same values bit for bit; `backend="auto"` uses it when installed, except for `diagnostics` and the
ablation `flags` of the numba kernel, which also computes certificates and ball and multi-step
measures. PyPI has wheels for Linux, macOS and Windows (Python ≥ 3.10); building
from source needs a Rust toolchain: `pip install -e ".[dev]"`, then `pytest -q`.

## Versions

1.3.0 adds the native backend, real walk weights and edge lengths; 1.3.1 changes no result. 1.2.2
is the version described in *The Min-Cut Formulation of Ollivier–Ricci Curvature*: it contains the
solver of orcut 1.4.4, as do 1.2.0 and 1.2.1, which differ from it in documentation only. Details:
[CHANGELOG.md](CHANGELOG.md).

## Layout

| Path | Contents |
| --- | --- |
| `orcut/_core.py` | the one-step kernel: bitsets, relays, witness relays, twin merging, warm-started Dinic |
| `orcut/_witness.py` | the linear-size witness network (one relay per common neighbour) |
| `orcut/_layered.py` | W1 on bounded integer metrics by a layered closure: balls, multi-step walks |
| `orcut/certify.py` | certificates (an optimal plan and potential), checked exactly |
| `orcut/weighted.py`, `_backend.py` | edge lengths; the calls into the native backend |
| `orcut/_flow.py` | integer constants, arenas, bitsets, CSR networks and Dinic's maximum flow, shared by the kernels |
| `orcut/_oracle.py` | the oracle: explicit distances, three-level closure, its own max flow |
| `orcut/api.py`, `graph.py`, `measures.py` | entry points, CSR graphs, measures |
| `src/` | the native backend: `hop.rs` (port of `_core`), `lengths.rs`, `flow.rs`, `nbhd.rs`, `lib.rs` |
| `tests/` | kernels against each other (bit for bit), the oracle and linear programming |

## Citation

If you use faster-or-curvature, please cite the paper that describes it (version 1.2.2 is the version
described there):

```bibtex
@misc{micaletto2026mincut,
  author        = {Giorgio Micaletto},
  title         = {The Min-Cut Formulation of {Ollivier--Ricci} Curvature},
  year          = {2026},
  eprint        = {2611.32528},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  doi           = {10.48550/arXiv.2611.32528},
  url           = {https://arxiv.org/abs/2611.32528},
}
```

GitHub's "Cite this repository" points to the same paper, from `CITATION.cff`.
