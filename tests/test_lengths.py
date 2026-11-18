"""Edge lengths (orcut.weighted): exact on integer lengths (the hop kernel at unit lengths, and the
layered min cut on D - 1 layers), linear programming on real lengths, idleness linearity, and the
truncation of real lengths."""
import warnings
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import orcut
from orcut import _backend as B
from orcut import _layered as LY
from orcut import measures as Ms

pytestmark = pytest.mark.skipif(not B.available(), reason="native backend not built")


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, orcut.Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


def _adj(g, lens):
    u, v = g.edges[:, 0], g.edges[:, 1]
    lens = np.asarray(lens, dtype=float)
    return csr_matrix((np.concatenate([lens, lens]), (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(g.num_nodes, g.num_nodes))


def _masses(G, i, kind):
    nb = sorted(G[i])
    d = len(nb)
    if kind == "lazy":
        m = {u: Fraction(1, d + 1) for u in nb + [i]}
    elif kind == "nonlazy":
        m = {u: Fraction(1, d) for u in nb}
    else:
        p = Fraction(kind)
        m = {u: (1 - p) / d for u in nb}
        m[i] = p
    return {u: x for u, x in m.items() if x}


def _net(G, i, j, kind):
    mi, mj = _masses(G, i, kind), _masses(G, j, kind)
    net = {u: mi.get(u, 0) - mj.get(u, 0) for u in set(mi) | set(mj)}
    return [u for u in net if net[u] > 0], [u for u in net if net[u] < 0], net


def _lp(A, G, i, j, kind):
    P, N, net = _net(G, i, j, kind)
    if not P:
        return 0.0
    D = dijkstra(A, directed=False, indices=P)[:, N]
    m, n = len(P), len(N)
    Aeq = np.zeros((m + n, m * n))
    beq = np.zeros(m + n)
    for a in range(m):
        Aeq[a, a * n:(a + 1) * n] = 1
        beq[a] = float(net[P[a]])
    for k in range(n):
        Aeq[m + k, k::n] = 1
        beq[m + k] = float(-net[N[k]])
    return linprog(D.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").fun


GRAPHS = [nx.karate_club_graph(), nx.petersen_graph(), nx.grid_2d_graph(5, 6), nx.barabasi_albert_graph(200, 3, seed=1),
          nx.powerlaw_cluster_graph(150, 3, 0.6, seed=2), nx.random_geometric_graph(120, 0.2, seed=3)]


@pytest.mark.parametrize("G", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_unit_lengths_reproduce_the_hop_kernel(G):
    _, g = _g(G)
    ones = np.ones(g.num_edges, np.int64)
    w = np.random.default_rng(0).integers(1, 6, g.num_edges)
    for meas in ("lazy", "nonlazy", Fraction(2, 5), Ms.Walk(weights=w)):
        a = orcut.curvature(g, meas, exact_fraction=True)
        b = orcut.curvature(g, meas, lengths=ones, exact_fraction=True, diagnostics=True)
        assert np.array_equal(a["w1_num"], b["w1_num"]) and np.array_equal(a["w1_den"], b["w1_den"])
        assert np.array_equal(a["curvature"], b["curvature"]) and np.all(b["dist"] == 1)
        assert np.all(b["pivots"] >= 0) and np.all(b["rows"] >= 0)
    assert np.allclose(orcut.curvature(g, "lly"), orcut.curvature(g, "lly", lengths=ones), atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_integer_lengths_match_the_layered_min_cut(seed):
    """The min-cut formulation with D - 1 layers (orcut._layered, explicit distances) and the
    primal-dual solver give the same exact W1 on integer lengths."""
    rng = np.random.default_rng(seed)
    G, g = _g(nx.powerlaw_cluster_graph(60, 3, 0.4, seed=seed))
    lens = rng.integers(1, 5, g.num_edges)
    A = _adj(g, lens)
    bi = np.empty(1 << 12, np.int64)
    for kind in ("lazy", "nonlazy", Fraction(1, 3)):
        out = orcut.curvature(g, kind, lengths=lens, exact_fraction=True)
        for e, (i, j) in enumerate(g.edges):
            i, j = int(i), int(j)
            P, N, net = _net(G, i, j, kind)
            assert out["dist"][e] == round(dijkstra(A, directed=False, indices=[i])[0, j])
            if not P:
                assert out["w1_num"][e] == 0
                continue
            den = int(out["w1_den"][e])
            wP = np.array([int(net[u] * den) for u in P], np.int64)
            wN = np.array([int(-net[u] * den) for u in N], np.int64)
            assert all((net[u] * den).denominator == 1 for u in P + N)
            Dm = np.rint(dijkstra(A, directed=False, indices=P)[:, N]).astype(np.int64)
            W = LY._layered_cut(len(P), len(N), Dm.ravel().copy(), int(Dm.max()), wP, wN, False, bi, 0)[0]
            assert W == out["w1_num"][e], (kind, e)


@pytest.mark.parametrize("seed", range(3))
def test_real_lengths_match_lp(seed):
    rng = np.random.default_rng(seed)
    G, g = _g(nx.powerlaw_cluster_graph(45, 3, 0.5, seed=seed))
    lens = rng.uniform(0.2, 2.0, g.num_edges)
    A = _adj(g, lens)
    dist = {}
    for kind in ("lazy", "nonlazy", Fraction(1, 2)):
        v = orcut.curvature(g, kind, lengths=lens)
        for e, (i, j) in enumerate(g.edges):
            i, j = int(i), int(j)
            d = dist.setdefault((i, j), dijkstra(A, directed=False, indices=[i])[0, j])
            assert v[e] == pytest.approx(1 - _lp(A, G, i, j, kind) / d, abs=1e-9), (kind, e)
    lly = orcut.curvature(g, "lly", lengths=lens)
    half = orcut.curvature(g, Fraction(1, 2), lengths=lens)
    assert np.allclose(lly, 2 * half, atol=1e-12)


def test_real_walk_weights_with_real_lengths_match_lp():
    rng = np.random.default_rng(5)
    G, g = _g(nx.random_geometric_graph(50, 0.3, seed=5))
    lens = rng.uniform(0.5, 1.5, g.num_edges)
    w = np.exp(-lens ** 2)  # the measure of GraphRicciCurvature (exp_power 2), idleness 1/2
    A = _adj(g, lens)
    wd = {}
    for e, (a, b) in enumerate(g.edges):
        wd[(int(a), int(b))] = wd[(int(b), int(a))] = float(w[e])
    v = orcut.curvature(g, Ms.Walk(weights=w, idleness=0.5), lengths=lens)
    for e, (i, j) in enumerate(g.edges):
        i, j = int(i), int(j)

        def m(u):
            tot = sum(wd[(u, x)] for x in G[u])
            out = {x: 0.5 * wd[(u, x)] / tot for x in G[u]}
            out[u] = 0.5
            return out

        mi, mj = m(i), m(j)
        net = {u: mi.get(u, 0) - mj.get(u, 0) for u in set(mi) | set(mj)}
        P = [u for u in net if net[u] > 1e-15]
        N = [u for u in net if net[u] < -1e-15]
        D = dijkstra(A, directed=False, indices=P)[:, N]
        nP, nN = len(P), len(N)
        Aeq = np.zeros((nP + nN, nP * nN))
        for a in range(nP):
            Aeq[a, a * nN:(a + 1) * nN] = 1
        for k in range(nN):
            Aeq[nP + k, k::nN] = 1
        beq = np.array([net[u] for u in P] + [-net[u] for u in N])
        lp = linprog(D.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").fun
        d = dijkstra(A, directed=False, indices=[i])[0, j]
        assert v[e] == pytest.approx(1 - lp / d, abs=1e-8)


@pytest.mark.parametrize("seed", [0, 1])
def test_lengths_are_idleness_linear_on_one_half_to_one(seed):
    rng = np.random.default_rng(seed)
    _, g = _g(nx.powerlaw_cluster_graph(70, 3, 0.5, seed=seed))
    lens = rng.integers(1, 6, g.num_edges)
    vals = []
    for p in (Fraction(1, 2), Fraction(3, 5), Fraction(7, 8)):
        out = orcut.curvature(g, p, lengths=lens, exact_fraction=True)
        vals.append([(1 - Fraction(int(n), int(d) * int(t))) / (1 - p)
                     for n, d, t in zip(out["w1_num"], out["w1_den"], out["dist"])])
    assert vals[0] == vals[1] == vals[2]


def test_truncation_keeps_k_digits_and_is_exact():
    rng = np.random.default_rng(1)
    _, g = _g(nx.powerlaw_cluster_graph(120, 3, 0.5, seed=1))
    lens = rng.uniform(0.5, 3.0, g.num_edges)
    a = orcut.curvature(g, lengths=lens, truncation=2, exact_fraction=True)
    b = orcut.curvature(g, lengths=orcut.truncate_lengths(lens, 2), exact_fraction=True)
    assert np.array_equal(a["curvature"], b["curvature"]) and np.array_equal(a["w1_num"], b["w1_num"])
    assert a["length_scale"] == 100 and b["length_scale"] == 1
    full = orcut.curvature(g, lengths=lens)
    eps = 1e-2 / lens.min()  # every distance shrinks by a factor of at most 1 - eps
    assert np.max(np.abs(full - a["curvature"])) <= 3 * eps / (1 - eps)
    assert np.max(np.abs(full - orcut.curvature(g, lengths=lens, truncation=6))) < 1e-4


def test_truncate_lengths_is_decimal_truncation():
    assert orcut.truncate_lengths([0.29, 1.005, 2.5, 7.0], 2).tolist() == [29, 100, 250, 700]
    assert orcut.truncate_lengths([0.123456789], 5).tolist() == [12345]
    with pytest.raises(ValueError):
        orcut.truncate_lengths([0.004], 2)
    with pytest.raises(ValueError):
        orcut.truncate_lengths([1.5], True)


def test_real_lengths_need_no_truncation_and_raise_no_warning():
    rng = np.random.default_rng(2)
    _, g = _g(nx.barabasi_albert_graph(300, 3, seed=2))
    lens = rng.uniform(0.5, 2.0, g.num_edges)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = orcut.curvature(g, lengths=lens)
        orcut.curvature(g, lengths=np.ceil(lens * 10))  # integral floats are integer lengths
    assert np.all(np.isfinite(v))


def test_local_and_global_modes_agree_exactly():
    """Small worlds use shared single-source distances, other graphs targeted Dijkstras: the exact
    fractions do not depend on the mode (edge subsets in shuffled order force the local mode)."""
    rng = np.random.default_rng(4)
    _, g = _g(nx.barabasi_albert_graph(400, 3, seed=4))
    lens = rng.integers(1, 9, g.num_edges)
    a = orcut.curvature(g, lengths=lens, exact_fraction=True)
    ids = rng.permutation(g.num_edges)
    b = orcut.curvature(g, lengths=lens, edges=ids, exact_fraction=True)
    assert np.array_equal(a["w1_num"][ids], b["w1_num"]) and np.array_equal(a["dist"][ids], b["dist"])
    c = orcut.curvature(g, lengths=lens, n_jobs=-1, exact_fraction=True)
    assert np.array_equal(a["w1_num"], c["w1_num"])


def test_networkx_attribute_names_and_errors():
    G = nx.karate_club_graph()  # integer 'weight' on every edge
    g, _ = orcut.Graph.from_networkx(G)
    w = orcut.edge_values(G, "weight")
    assert np.array_equal(orcut.curvature(G, lengths="weight"), orcut.curvature(g, lengths=w))
    assert np.array_equal(orcut.curvature(G, Ms.Walk(weights="weight")), orcut.curvature(g, Ms.Walk(weights=w)))
    with pytest.raises(ValueError):
        orcut.curvature(g, lengths=w, backend="numba")
    with pytest.raises(ValueError):
        orcut.curvature(g, lengths=-w)
    with pytest.raises(ValueError):
        orcut.curvature(g, truncation=3)
