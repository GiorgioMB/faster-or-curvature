"""The native backend (orcut._native) against the numba kernel of orcut._core: the same exact
fractions and the same float values, bit for bit, for every measure and on the instances that
force each network (fast path, closure, witness relays, witness network); real walk weights
against linear programming; Lin--Lu--Yau through idleness 1/2."""
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import orcut
from orcut import _backend as B
from orcut import measures as Ms
from conftest import _w1, random_graphs, small_graphs

pytestmark = pytest.mark.skipif(not B.available(), reason="native backend not built")


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, orcut.Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


GRAPHS = small_graphs() + random_graphs(2) + [nx.barabasi_albert_graph(300, 4, seed=7),
                                              nx.powerlaw_cluster_graph(250, 3, 0.7, seed=8),
                                              nx.random_geometric_graph(300, 0.15, seed=9)]


def _measures(g, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 7, g.num_edges)
    idl = np.array([Fraction(int(rng.integers(0, 3)), int(rng.integers(3, 8))) for _ in range(g.num_nodes)], dtype=object)
    return ["lazy", "nonlazy", "lly", Fraction(1, 3), 0.25, Ms.Walk(weights=w, idleness=idl), Ms.Walk(weights=w)]


def _same(g, meas, ids=None):
    a = orcut.curvature(g, meas, edges=ids, exact_fraction=True, backend="numba")
    b = orcut.curvature(g, meas, edges=ids, exact_fraction=True, backend="native", n_jobs=1)
    assert np.array_equal(a["curvature"], b["curvature"], equal_nan=True), meas
    assert np.array_equal(a["w1_num"], b["w1_num"]) and np.array_equal(a["w1_den"], b["w1_den"]), meas


@pytest.mark.parametrize("G", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_native_matches_numba_bit_for_bit(G):
    _, g = _g(G)
    for meas in _measures(g):
        _same(g, meas)


def _instances():
    from test_witness import connector_edge, middle_edge
    return [connector_edge(1500, 1500, 4, 0.01, 0), middle_edge(700, 640, 50, 40, 1, 0),
            middle_edge(120, 100, 10, 2, 40, 0), middle_edge(200, 150, 12, 3, 50, 1)]


def test_native_matches_numba_on_the_stress_instances_and_uses_every_network():
    from orcut import _core as C
    paths = set()

    def run(g, ids):
        p = B.hop(g, ids, Ms.parse_spec("lazy", g), 1)[3]
        d = orcut.curvature(g, edges=ids, diagnostics=True)["diagnostics"][:, C.D_PATH]
        assert np.array_equal(p, d)  # the same network is chosen for every edge
        return set(p.tolist())

    for g in _instances():
        eid = np.array([g.edge_id(0, 1)], np.int64)
        for meas in _measures(g)[:5]:
            _same(g, meas, eid)
        paths |= run(g, eid)
    for G in GRAPHS[-3:]:
        _, g = _g(G)
        paths |= run(g, np.arange(g.num_edges, dtype=np.int64))
    assert {1, 2, 3, 4, 5} <= paths, paths  # no adjacency, fast path, closure, witness relays, witness network


def test_threads_and_edge_order_do_not_change_values():
    _, g = _g(nx.powerlaw_cluster_graph(400, 4, 0.5, seed=1))
    ids = np.arange(g.num_edges, dtype=np.int64)
    spec = Ms.parse_spec("lazy", g)
    a = B.hop(g, ids, spec, 1)
    b = B.hop(g, ids, spec, 4)
    c = B.hop(g, ids[::-1].copy(), spec, 3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert np.array_equal(a[1], c[1][::-1])


@pytest.mark.parametrize("seed", [0, 1])
def test_real_walk_weights_match_lp(seed):
    rng = np.random.default_rng(seed)
    G, g = _g(nx.powerlaw_cluster_graph(40, 3, 0.5, seed=seed))
    w = rng.uniform(0.1, 3.0, g.num_edges)
    wd = {}
    for e, (a, b) in enumerate(g.edges):
        wd[(int(a), int(b))] = wd[(int(b), int(a))] = float(w[e])
    for idle in (0.0, 0.3, "lazy"):
        v = orcut.curvature(g, Ms.Walk(weights=w, idleness=idle))

        def meas(u):
            p = 1.0 / (G.degree(u) + 1) if idle == "lazy" else idle
            tot = sum(wd[(u, x)] for x in G[u])
            m = {x: (1 - p) * wd[(u, x)] / tot for x in G[u]}
            if p:
                m[u] = p
            return m

        for e, (i, j) in enumerate(g.edges):
            assert v[e] == pytest.approx(1 - _w1(G, meas(int(i)), meas(int(j))), abs=1e-9)


def test_integral_float_weights_take_the_exact_path():
    _, g = _g(nx.barabasi_albert_graph(150, 3, seed=2))
    w = np.random.default_rng(3).integers(1, 9, g.num_edges)
    a = orcut.curvature(g, Ms.Walk(weights=w), exact_fraction=True)
    b = orcut.curvature(g, Ms.Walk(weights=w.astype(float)), exact_fraction=True)
    assert np.array_equal(a["curvature"], b["curvature"]) and np.array_equal(a["w1_num"], b["w1_num"])
    c = orcut.curvature(g, Ms.Walk(weights=w + 1e-9))  # real weights: float masses, same cut
    assert np.allclose(a["curvature"], c, atol=1e-7)


@pytest.mark.parametrize("G", GRAPHS[:10] + GRAPHS[-3:], ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_lly_is_twice_the_curvature_at_idleness_one_half(G):
    """kappa_p = (1 - p) kappa_LLY for p in [1/2, 1) on every graph: the threshold 1/(deg_max + 1)
    of the unweighted kernel and the threshold 1/2 used for weighted walks and lengths agree."""
    _, g = _g(G)
    a = orcut.curvature(g, "lly", exact_fraction=True)
    b = orcut.curvature(g, Fraction(1, 2), exact_fraction=True)
    dmax = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]])
    for e in range(g.num_edges):
        k_lly = (1 - Fraction(int(a["w1_num"][e]), int(a["w1_den"][e]))) * Fraction(int(dmax[e]) + 1, int(dmax[e]))
        assert k_lly == 2 * (1 - Fraction(int(b["w1_num"][e]), int(b["w1_den"][e])))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_walk_lly_is_idleness_linear(seed):
    _, g = _g(nx.powerlaw_cluster_graph(60, 3, 0.5, seed=seed))
    w = np.random.default_rng(seed).integers(1, 9, g.num_edges)
    vals = []
    for p in (Fraction(1, 2), Fraction(2, 3), Fraction(4, 5)):
        out = orcut.curvature(g, Ms.Walk(weights=w, idleness=p), exact_fraction=True)
        vals.append([(1 - Fraction(int(n), int(d))) / (1 - p) for n, d in zip(out["w1_num"], out["w1_den"])])
    assert vals[0] == vals[1] == vals[2]
    lly = orcut.curvature(g, Ms.Walk(weights=w, idleness="lly"))
    assert np.allclose(lly, [float(x) for x in vals[0]], atol=1e-12)


def test_version_matches_the_crate():
    from orcut import _native
    assert orcut.__version__ == _native.__version__
