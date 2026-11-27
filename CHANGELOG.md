# Changelog

## 1.3.2

- No change in results. The kernel kept frozen since the first release (`orcut/_kernels.py`) is
  removed, with its interface (`orcut.exact`, `exact_ids`, `exact_or_ids`, `N_EXACT`, the screening
  and TLW drivers): for `orcut.exact(g)["c_OR"]`, `["c_OR0"]` and `["c_LLY"]`, use
  `orcut.curvature(g)`, `orcut.curvature(g, "nonlazy")` and `orcut.curvature(g, "lly")`, which
  return the same values, bit for bit (`edges` and `n_jobs` work as before). The primitives the
  kernels share (constants, arenas, bitsets, CSR networks, Dinic, row deduplication) move unchanged
  to `orcut/_flow.py`.
- The oracle is `orcut._oracle` (`exact_ids_oracle`), formerly `orcut._reference`; the tests check
  `orcut._core` against it bit for bit, under every ablation flag.
- One citation, the paper (arXiv:2611.32528): the BibTeX entry of the README and the
  `preferred-citation` of `CITATION.cff`, which GitHub's "Cite this repository" uses; a `Paper`
  link among the project URLs.

## 1.3.1

- No change in results. The citation names 1.2.2, the version of *The Min-Cut Formulation of
  Ollivier–Ricci Curvature* (1.2.0 and 1.2.1 have the same code); a shorter README and changelog;
  the native backend without its private developer hooks (`_prof`, `_prof_get`, `_transport`); the
  length solver described as the network simplex it is.

## 1.3.0

- Native backend `orcut._native` (Rust, PyO3, rayon): a port of the one-step kernel of
  `orcut._core`, with the same exact fractions, float values and network for every edge (tested),
  no JIT, and `orcut.curvature(backend="auto" | "native" | "numba")`.
- Real walk weights (`Walk(weights=...)`, floats or a networkx attribute name): the same minimum cut
  with float capacities. `Walk(idleness="lly")`: Lin–Lu–Yau as twice the curvature at idleness 1/2.
- Edge lengths (`lengths=...`): 1 - W1 / d(i, j) for any one-step measure; exact with integer lengths
  (equal to the layered min cut, tested), floating point with real ones (checked against linear
  programming), each edge solved by a network simplex after merging rows and columns whose costs
  differ by a constant. `truncation=k` keeps k decimal digits (exact on the truncated metric).
  `orcut.truncate_lengths`, `orcut.edge_values`, `orcut.native_available`. Against
  GraphRicciCurvature 0.6.1: 2.6x faster on OpenFlights routes and 15–23x on smaller or spatial
  networks, with the same values to 1e-13.
- Built with maturin: abi3 wheels for Linux, macOS and Windows; a Rust toolchain builds the sdist.

## 1.2.2, 1.2.1

- Documentation only: citation (`CITATION.cff`), badges, the new title of the paper of Micaletto and
  Nigrelli in the README.

## 1.2.0

- The solver of *The Min-Cut Formulation of Ollivier–Ricci Curvature*; first release on PyPI, as
  `faster-or-curvature`.
- Witness relays (`F_WITNESS`): one zero-weight relay per common neighbour for the pairs at distance
  2 left after the relays H1–H4, when the network gets at least four times smaller
  (`PATH_CUTW`), and a bound on the bitset work beyond which an edge is solved on the linear-size
  witness network (`PATH_WIT`; `orcut._witness.witness_ids` on its own). Values unchanged (tested).

## 1.1.0

- `orcut.curvature` for any one-step measure (lazy, non-lazy, LLY, idleness p, `orcut.Walk`), exact
  fractions, diagnostics and ablation flags; `orcut.certify` and `Certificate.verify`;
  `ball_curvature` and `walk_curvature` (layered closure); `affected_edges` for exact incremental
  updates.
