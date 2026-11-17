# Changelog

## 1.3.0

- Native backend: `orcut._native`, a Rust (PyO3, rayon) port of the general one-step kernel of
  `orcut._core` (bitsets, hubs, twin merging, fast path, witness relays, warm start) with the
  linear-size witness network for the edges it dispatches. The same exact fractions and the same
  float values as the numba kernel, bit for bit, and the same network for every edge (tested). No
  JIT: the first call takes about 0.1 s instead of about a minute of compilation; 1.2--1.7x faster
  per thread on our benchmark graphs (4.4x on a grid). `orcut.curvature(backend="auto" | "native" |
  "numba")`; diagnostics and ablation flags stay on the numba kernel, as do certificates, ball and
  multi-step measures and the frozen 1.0.0 kernel.
- Real walk weights: `Walk(weights=...)` with floats (or, for a networkx graph, an edge attribute
  name) runs the same minimum cut with float capacities. `Walk(idleness="lly")`: the Lin--Lu--Yau
  curvature of a weighted walk, as twice the curvature at idleness 1/2 (kappa_p is linear on
  [1/2, 1) for every walk and every metric; tested exactly against the 1/(deg_max + 1) route on
  unweighted graphs).
- Edge lengths: `orcut.curvature(g, measure, lengths=...)` is 1 - W1(m_i, m_j) / d(i, j) for the
  shortest-path metric of positive lengths and any one-step measure. Integer lengths are exact
  (`exact_fraction=True` gives W1 and d(i, j) as integers), equal to the layered min cut on D - 1
  layers (tested); real lengths run in floating point (checked against linear programming).
  `truncation=k` keeps k decimal digits of real lengths (exact on the truncated metric).
  `orcut.truncate_lengths`, `orcut.edge_values`, `orcut.native_available`.
- Packaging: built with maturin; wheels (abi3, Python >= 3.10) for Linux x86_64 and aarch64,
  macOS universal2 and Windows x64. Building from source needs a Rust toolchain.

## 1.2.2

- Documentation only; no function changes. 1.2.0 is the version described in *The Min-Cut
  Formulation of Ollivier--Ricci Curvature* (Giorgio Micaletto, arXiv, forthcoming): the README says
  so in its versions and citation sections, the new `CITATION.cff` gives the citation (GitHub's
  "Cite this repository"), and the tag `v1.2.0` is annotated with the same note.

## 1.2.1

- Documentation only; no function changes. The paper of Micaletto and Nigrelli that used the 1.0.0
  solver is now titled *From Proxy to Certificate: Two-Sided Ollivier--Ricci Bounds from Balanced
  Forman Curvature* (formerly *What Balanced Forman Curvature Certifies About Ollivier--Ricci
  Curvature*); the README and the docstring of `orcut/_kernels.py` give the new title. The README
  also has PyPI badges and a citation section.

## 1.2.0

- The version described in *The Min-Cut Formulation of Ollivier--Ricci Curvature* (Giorgio
  Micaletto, arXiv, forthcoming).
- First release on PyPI, as `faster-or-curvature` (it imports as `orcut`), with minimum versions
  of numpy and numba instead of exact pins.
- Witness relays (`F_WITNESS`, on by default: `F_ALL = 31`). A pair of P x N at distance 2 has a
  common neighbour w, and every pair in (N(w) & P) x (N(w) & N) is within distance 2, so the
  implications of these pairs can go through one zero-weight relay per common neighbour. When the
  pairs left at distance 2 after the relays H1--H4 would need many explicit arcs, the kernel routes
  them through per-witness relays instead, if this is at least four times smaller (an upper bound
  is checked first, so the test costs little when it fails). Values are unchanged (tested bit for
  bit against the previous kernels); diagnostics report the new path `PATH_CUTW = 4`.
- Bounded bitset work (part of `F_WITNESS`; `core_ids(..., dispatch=4)`): when the bitsets of an
  edge would cost more than four times its local input (the sum of the degrees of the rows and
  columns), before they are built or during the coverage while most rows are not yet within
  distance 2 of all of U_j, the edge is solved on the witness network instead (`PATH_WIT = 5`).
  On large sparse neighbourhoods this is 2--13x faster; on the real graphs we tried it never
  triggers and costs under 3%.
- `orcut._witness.witness_ids`: the linear-size witness network on its own (one relay per vertex
  adjacent to both P and N; twin merging; direct arcs for witnesses that reach one row class or one
  column; warm start). It has at most 3 m_ij + 2 nodes and 5 m_ij arcs, m_ij the sum of the degrees
  of P and N, and is built in O(m_ij) from the adjacency lists, with no bitsets and no distances.
  Exact fractions identical to `orcut.curvature` (tested for six measures and every flag).

## 1.1.0

- `orcut.curvature(graph, measure)`: the general one-step kernel (`orcut/_core.py`) for lazy,
  non-lazy, LLY, uniform idleness p, and weighted walks (`orcut.Walk`: positive integer weights,
  rational idleness), in exact integer arithmetic (common denominators up to 2^50; edges above it
  are NaN and flagged). Optional exact W1 fractions and per-edge diagnostics (path, nodes, arcs, row
  groups after twin merging, warm-start flow). Every engineering step can be switched off
  (`flags`) for ablations; values do not change (tested).
- The TLW certificate is no longer computed by the new kernel (it stays in the frozen kernel).
- `orcut.certify(graph, edge, measure)`: an optimal transport plan and an optimal potential, read
  off the minimum cut and a maximum flow on the tight pairs, verified exactly; `Certificate.verify`
  re-checks them from scratch.
- `orcut.ball_curvature`, `orcut.walk_curvature`: the layered closure (D = 2r + 1) for measures
  supported on balls of radius r (reference implementation, explicit distances).
- `orcut.affected_edges(graph, pairs)`: the edges whose curvature can change after inserting or
  deleting vertex pairs (an endpoint in the union of N[a] u N[b]), for exact incremental updates
  in rewiring and edge-removal algorithms.
- Certificates of one-step measures take the P x N distances from adjacency and 2-paths (numba)
  instead of a BFS per vertex; `Certificate.verify` explores distances only up to the largest
  potential difference (exactly for the pairs of the plan). An edge whose two measures coincide is
  reported as certified by the layered closure.
- Tests: the new kernel against the frozen one (bit for bit, 16 flag combinations) and against
  linear programming; layered closure against LP; certificates; kappa_p = (1 - p) kappa_LLY for
  p >= 1/(deg_max + 1); locality of `affected_edges` under random insertions and deletions.

## 1.0.0

- The exact solver of `curvature_transfer` v11, extracted without changes (`orcut/_kernels.py`,
  `orcut/_reference.py`), with its tests.
