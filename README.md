# faster-or-curvature

[![PyPI version](https://img.shields.io/pypi/v/faster-or-curvature.svg)](https://pypi.org/project/faster-or-curvature/)
[![Downloads](https://img.shields.io/pepy/dt/faster-or-curvature)](https://pepy.tech/project/faster-or-curvature)
[![License](https://img.shields.io/pypi/l/faster-or-curvature.svg)](https://pypi.org/project/faster-or-curvature/)

Exact Ollivier–Ricci curvature of every edge of a graph, with one minimum cut per edge, in exact
integer arithmetic and without a distance matrix; also for weighted walks and for edge lengths.
A native backend (Rust) computes it with no JIT warm-up, on all cores.

```bash
pip install faster-or-curvature        # imports as orcut
```

Any two vertices of the closed neighbourhoods of an edge are at distance at most
three, so the Kantorovich dual of the edgewise transport problem has an integral
optimum with potentials in {0, 1, 2, 3}. Written in unary, it is a maximum-weight
closure, solved by one s–t minimum cut. Its arcs come from adjacency and common
neighbours only (pairs at distance three give none), and with one relay node per
common neighbour the network is linear in the adjacency lists of the vertices that
carry net mass. The same holds for any one-step measures (lazy, non-lazy, any idleness,
weighted walks), and, with 2r layers, for measures supported on balls of radius r.

```python
import orcut
g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph / PyG Data
orcut.curvature(g)                      # lazy (idleness 1/(deg+1)), aligned with g.edges (u < v)
orcut.curvature(g, "lly")               # Lin–Lu–Yau; also "nonlazy", a number p (idleness p),
orcut.curvature(g, orcut.Walk(weights=w, idleness=0.5))   # or a weighted walk (any positive weights)
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

## Weighted graphs

*Walk weights* (conductances: m_u(x) proportional to w(u, x)) leave the metric alone, so the cut
is the same and the weights only enter as capacities: integer weights are exact, real weights run
in floating point (native backend), and neither needs truncation. `Walk(weights=w,
idleness="lly")` is the Lin–Lu–Yau curvature of the walk, computed as twice the curvature at
idleness 1/2: kappa_p is linear in p on [1/2, 1) for every walk and every metric (replacing an
optimal potential f by min(f, f(j) - eps + d(., j)) changes the objective by (2p - 1) eps >= 0),
which the tests check exactly against the 1/(deg_max + 1) route of unweighted graphs.

*Edge lengths* change the metric. With integer lengths the Kantorovich dual keeps an integral
optimum, with potentials in {0, ..., D} for D the largest distance between the supports, so the
transport problem is still one minimum cut, on D - 1 layers (the layered closure of
`orcut._layered`, which the tests match exactly). Real lengths have no integral optimum. The
native solver computes the distances between the vertices of positive and negative net mass by
Dijkstras that stop at their targets (shared by every edge of a run with a common endpoint, or,
when the graph is a small world, one single-source search per vertex shared by all edges), merges
the rows and columns whose costs differ by a constant (paying the constant exactly), and solves
the reduced transportation problem by successive shortest paths: exactly (i64, i128) with integer
lengths, in floating point otherwise (checked against linear programming).

`truncation=k` keeps k decimal digits of every real length and drops the rest: the lengths become
the integers floor(length * 10^k), the result is an exact fraction of the truncated metric, and
every distance shrinks by a factor of at most 1 - 10^-k / (smallest length). It buys exactness
rather than speed: 1.2–1.3x at k = 5 and up to 1.6x at k = 1 on the graphs below. Seconds on one
core (lazy measure; lengths uniform in [0.5, 2], or Euclidean for the geometric graph):

| graph | edges | hop metric | real lengths | truncation=5 | truncation=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barabási–Albert, n = 2000, m = 4 | 7984 | 0.10 | 2.4 | 1.9 | 1.5 |
| Barabási–Albert, n = 4000, m = 4 | 15984 | 0.18 | 7.8 | 6.3 | 5.0 |
| powerlaw cluster, n = 4000 | 15978 | 0.18 | 7.8 | 6.0 | 4.8 |
| random geometric, n = 3000 | 17129 | 0.14 | 0.72 | 0.62 | – |

## Native backend

`orcut._native` is a Rust (PyO3, rayon) port of the general one-step kernel of `orcut._core`: the
same bitsets, hubs, twin merging, fast path, witness relays and warm start, and the linear-size
witness network for the edges it dispatches. W1 is unique, so it returns the same exact fractions
and the same float values as the numba kernel, bit for bit, and it also picks the same network for
every edge (tested on every graph and measure of the suite). There is no JIT: the first call in a
fresh environment takes about 0.1 s instead of about a minute of numba compilation. Per thread, on
one core (lazy measure, seconds):

| graph | edges | numba | native | speedup |
| --- | ---: | ---: | ---: | ---: |
| Barabási–Albert, n = 20000, m = 5 | 99975 | 1.81 | 1.33 | 1.36x |
| powerlaw cluster, n = 20000 | 99944 | 2.93 | 2.32 | 1.26x |
| Watts–Strogatz, n = 20000, k = 10 | 100000 | 0.80 | 0.48 | 1.68x |
| random geometric, n = 10000 | 137691 | 5.15 | 4.32 | 1.19x |
| G(n, p), n = 5000, mean degree 20 | 49857 | 1.05 | 0.87 | 1.21x |
| grid 150 x 150 | 44700 | 0.17 | 0.04 | 4.41x |

`orcut.curvature(..., backend="auto")` uses it whenever it is installed, except for `diagnostics`
and ablation `flags`, which belong to the numba kernel; `backend="native"` or `"numba"` forces
one. Numba stays for the frozen 1.0.0 kernel, certificates, ball and multi-step measures,
diagnostics and ablations.

## Installation

Python 3.10 or later; numpy and numba come with it. PyPI has wheels with the native backend for
Linux (x86_64, aarch64), macOS (universal2) and Windows (x64), one per platform for every Python
from 3.10 (abi3). Building from source (the sdist, or a clone) needs a Rust toolchain
([rustup](https://rustup.rs)): from a clone, `pip install -e ".[dev]"` builds the extension and adds
what the tests need (pytest, networkx, scipy), and `pytest -q` runs them.

Without a package index (for example on a cluster without PyPI access), copy the wheel for the
cluster's platform over from a machine that has PyPI:

```bash
pip download faster-or-curvature --no-deps --only-binary=:all: \
    --platform manylinux2014_x86_64 --python-version 3.12 -d wheels/   # where PyPI is reachable
conda install -c conda-forge "numpy>=2.2" "numba>=0.62"                  # on the cluster
pip install --no-index --no-deps wheels/faster_or_curvature-*.whl
```

## Versions

* **1.3.0**: the native backend (Rust, bit for bit the numba kernel, no JIT), real walk weights,
  Lin–Lu–Yau for weighted walks, and edge lengths (exact with integer lengths, `truncation` for
  real ones). Values on the hop metric unchanged (tested).
* **1.2.0** is the version described in *The Min-Cut Formulation of Ollivier–Ricci Curvature*
  (Giorgio Micaletto, arXiv, forthcoming); `pip install faster-or-curvature==1.2.0` reproduces it.
  It adds witness relays (one zero-weight relay per common neighbour) for the pairs at distance 2
  that the bitset network would list one by one, chosen per edge when the network gets smaller, and a
  bound on the bitset work (four words per adjacency entry) beyond which an edge is solved on the
  linear-size witness network (`orcut._witness`, also usable on its own). Values
  unchanged (tested). 1.2.1 and 1.2.2 change documentation only.
* **1.1.0**: `orcut.curvature` (any one-step measure, exact fractions, diagnostics, ablation
  flags), certificates (`orcut.certify`), the layered closure for supports of radius r
  (`ball_curvature`, `walk_curvature`), and exact incremental updates (`affected_edges`). The kernel of 1.0.0 is kept unchanged in
  `orcut/_kernels.py`, and `orcut.curvature` returns the same values bit for bit (tested).
* **1.0.0** is the solver used for every exact curvature in Micaletto and Nigrelli,
  *From Proxy to Certificate: Two-Sided Ollivier–Ricci Bounds from Balanced Forman Curvature*:
  its compiled kernels (`orcut/_kernels.py`) are those of `curvature_transfer`.

## Layout

| Path | Contents |
| --- | --- |
| `orcut/graph.py` | CSR graph, canonical undirected edges, networkx / PyG / edge-list constructors |
| `orcut/_kernels.py` | the 1.0.0 kernel (frozen): lazy, non-lazy, LLY, and the TLW certificate |
| `orcut/_core.py` | the general one-step kernel: per-vertex masses, bitsets, relays, witness relays, ablation flags, diagnostics |
| `orcut/_witness.py` | the linear-size witness network on its own (one relay per common neighbour; no bitsets, no distances) |
| `orcut/_layered.py` | W1 on bounded integer metrics by a layered closure; ball and multi-step measures; certificates |
| `orcut/measures.py` | measure specifications (`"lazy"`, `"nonlazy"`, `"lly"`, idleness p, `Walk` with integer or real weights) |
| `orcut/weighted.py` | edge lengths: `truncate_lengths`, `edge_values` (networkx attributes), the solver's front end |
| `orcut/_backend.py` | the native backend: availability, threads, calls into `orcut._native` |
| `src/` | the native backend in Rust: `hop.rs` (port of `_core`), `lengths.rs` (edge lengths), `flow.rs` (Dinic), `nbhd.rs` (neighbourhoods, masses), `lib.rs` (bindings, threads) |
| `orcut/certify.py` | per-edge certificates (plan and potential) and their exact verification |
| `orcut/_reference.py` | independent oracle: full local distance matrix, three-level closure, recursive Dinic |
| `orcut/api.py` | `curvature` (measures, lengths, truncation, backend), `ball_curvature`, `walk_curvature`, `affected_edges`, `exact` (1.0.0 API), thread control |
| `tests/` | kernels against each other (bit for bit), against the oracle, and against linear programming |

## Citation

Version 1.2.0 is the version described in *The Min-Cut Formulation of Ollivier–Ricci Curvature*
(Giorgio Micaletto), which will appear on arXiv. Until then, please cite the software:

```bibtex
@misc{faster-or-curvature,
  author       = {Giorgio Micaletto},
  title        = {faster-or-curvature: Exact Ollivier--Ricci curvature in exact integer arithmetic},
  year         = {2026},
  version      = {1.2.0},
  howpublished = {\url{https://github.com/GiorgioMB/faster-or-curvature}},
  note         = {Version 1.2.0, described in ``The Min-Cut Formulation of Ollivier--Ricci Curvature''},
}
```

GitHub's "Cite this repository" button gives the same entry, from `CITATION.cff`.
<!--
When the paper is on arXiv, replace the entry above with the preprint (and fill in the
preferred-citation block at the end of CITATION.cff):

@misc{micaletto2026mincut,
  author        = {Giorgio Micaletto},
  title         = {The Min-Cut Formulation of {Ollivier--Ricci} Curvature},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
}

and, once the conference version is out, with @inproceedings{..., booktitle = {...}, pages = {...}}.
-->
