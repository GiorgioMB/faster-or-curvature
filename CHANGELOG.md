# Changelog

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
  Nigrelli in the README and in the docstring of `orcut/_kernels.py`.

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
  updates. The new kernel matches the frozen 1.0.0 kernel bit for bit (tested).

## 1.0.0

- The exact solver of `curvature_transfer` v11 (the code of Micaletto and Nigrelli, *From Proxy to
  Certificate*), extracted without changes (`orcut/_kernels.py`, `orcut/_reference.py`).
