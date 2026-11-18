//! Edge lengths: W1 for the shortest-path metric of positive edge lengths.
//!
//! For an edge (u, v) the vertices of positive net mass (P) lie in N[i] and those of negative net
//! mass (N) in N[j], {i, j} = {u, v}, so every pair of P x N has one vertex in N[u]. Distances come
//! from Dijkstras that stop as soon as all their targets are settled: for a run of edges that share
//! their first endpoint u (consecutive in ``graph.edges``), one Dijkstra from each vertex of N[u]
//! serves the whole run; otherwise each edge runs them from the smaller of P and N. Rows whose costs
//! differ by a constant are merged (the constant is paid exactly), then columns, and the reduced
//! transportation problem is solved by a network simplex (row-minimum starting basis, block pricing
//! by rows, strongly feasible trees). With integer lengths and masses everything is exact (i64
//! costs, i128 totals) and equal rows merge exactly; with real lengths the same steps run in
//! floating point.
use crate::nbhd::{Mass, Meas, Nbhd};
use crate::Graph;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Developer profiling (``orcut._native._prof``): nanoseconds in distance searches and in solves.
pub static PROF: AtomicBool = AtomicBool::new(false);
pub static T_DIST: AtomicU64 = AtomicU64::new(0);
pub static T_SOLVE: AtomicU64 = AtomicU64::new(0);

#[inline]
fn prof_start() -> Option<Instant> {
    if PROF.load(Ordering::Relaxed) {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline]
fn prof_add(c: &AtomicU64, t: Option<Instant>) {
    if let Some(t) = t {
        c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub trait Dist:
    Copy + Default + PartialOrd + Add<Output = Self> + Sub<Output = Self> + AddAssign + SubAssign + Send + Sync + 'static
{
    const ZERO: Self;
    const MAXV: Self;
    const EXACT: bool;
    fn to_f64(self) -> f64;
    fn to_i64(self) -> i64;
    fn bits(self) -> u64;
    /// Cost of the artificial arcs of the network simplex: more than any path of real arcs.
    fn art(maxc: Self, n: usize) -> Self;
    /// Reduced costs above -tol do not enter (0 for integers; float rounding near ``art``).
    fn ntol(art: Self) -> Self;
}
impl Dist for i64 {
    const ZERO: i64 = 0;
    const MAXV: i64 = i64::MAX / 4;
    const EXACT: bool = true;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn to_i64(self) -> i64 {
        self
    }
    fn bits(self) -> u64 {
        self as u64
    }
    fn art(maxc: i64, n: usize) -> i64 {
        (maxc + 1).saturating_mul(n as i64 + 1)
    }
    fn ntol(_art: i64) -> i64 {
        0
    }
}
impl Dist for f64 {
    const ZERO: f64 = 0.0;
    const MAXV: f64 = f64::INFINITY;
    const EXACT: bool = false;
    fn to_f64(self) -> f64 {
        self
    }
    fn to_i64(self) -> i64 {
        0
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn art(maxc: f64, n: usize) -> f64 {
        (maxc + 1.0) * (n as f64 + 1.0)
    }
    fn ntol(art: f64) -> f64 {
        art * 1e-14
    }
}

/// Mass times length, accumulated exactly (i128) when both are integers.
pub trait Prod<D: Dist>: Mass {
    type Acc: Copy + Default + AddAssign + SubAssign + Send;
    fn mul(self, d: D) -> Self::Acc;
    fn acc_f64(a: Self::Acc) -> f64;
    fn acc_i64(a: Self::Acc) -> Option<i64>;
}
impl Prod<i64> for i64 {
    type Acc = i128;
    fn mul(self, d: i64) -> i128 {
        self as i128 * d as i128
    }
    fn acc_f64(a: i128) -> f64 {
        a as f64
    }
    fn acc_i64(a: i128) -> Option<i64> {
        i64::try_from(a).ok()
    }
}
impl Prod<f64> for i64 {
    type Acc = f64;
    fn mul(self, d: f64) -> f64 {
        self as f64 * d
    }
    fn acc_f64(a: f64) -> f64 {
        a
    }
    fn acc_i64(_a: f64) -> Option<i64> {
        None
    }
}
impl Prod<i64> for f64 {
    type Acc = f64;
    fn mul(self, d: i64) -> f64 {
        self * d as f64
    }
    fn acc_f64(a: f64) -> f64 {
        a
    }
    fn acc_i64(_a: f64) -> Option<i64> {
        None
    }
}
impl Prod<f64> for f64 {
    type Acc = f64;
    fn mul(self, d: f64) -> f64 {
        self * d
    }
    fn acc_f64(a: f64) -> f64 {
        a
    }
    fn acc_i64(_a: f64) -> Option<i64> {
        None
    }
}

#[inline]
fn hpush<D: PartialOrd + Copy>(h: &mut Vec<(D, u32)>, x: (D, u32)) {
    h.push(x);
    let mut k = h.len() - 1;
    while k > 0 {
        let p = (k - 1) / 2;
        if h[k].0 < h[p].0 {
            h.swap(k, p);
            k = p;
        } else {
            break;
        }
    }
}

#[inline]
fn hpop<D: PartialOrd + Copy>(h: &mut Vec<(D, u32)>) -> Option<(D, u32)> {
    let n = h.len();
    if n == 0 {
        return None;
    }
    h.swap(0, n - 1);
    let top = h.pop();
    let n = h.len();
    let mut k = 0usize;
    loop {
        let l = 2 * k + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let c = if r < n && h[r].0 < h[l].0 { r } else { l };
        if h[c].0 < h[k].0 {
            h.swap(c, k);
            k = c;
        } else {
            break;
        }
    }
    top
}

#[inline]
fn bump(ep: &mut u32, arr: &mut [u32]) -> u32 {
    if *ep == u32::MAX {
        arr.fill(0);
        *ep = 0;
    }
    *ep += 1;
    *ep
}

/// Group equal rows of a (nrows, kw) matrix of distances (bitwise equality).
fn dedupe_d<D: Dist>(keys: &[D], nrows: usize, kw: usize, table: &mut Vec<i32>, gid: &mut Vec<u32>, leader: &mut Vec<u32>) -> usize {
    let mut tsize = 4usize;
    while tsize < 2 * nrows {
        tsize <<= 1;
    }
    let mask = (tsize - 1) as u64;
    table.clear();
    table.resize(tsize, -1);
    gid.clear();
    gid.resize(nrows, 0);
    leader.clear();
    let mut ng = 0usize;
    for r in 0..nrows {
        let base = r * kw;
        let mut h: u64 = 1469598103934665603;
        for x in 0..kw {
            h = (h ^ keys[base + x].bits()).wrapping_mul(1099511628211);
        }
        h ^= h >> 29;
        let mut p = (h & mask) as usize;
        loop {
            let g = table[p];
            if g < 0 {
                table[p] = ng as i32;
                leader.push(r as u32);
                gid[r] = ng as u32;
                ng += 1;
                break;
            }
            let lb = leader[g as usize] as usize * kw;
            if (0..kw).all(|x| keys[lb + x].bits() == keys[base + x].bits()) {
                gid[r] = g as u32;
                break;
            }
            p = (p + 1) & (tsize - 1);
        }
    }
    ng
}

#[derive(Clone, Copy, Default)]
pub struct LenOut {
    pub val: f64,
    pub num: i64,
    pub den: i64,
    pub dist_i: i64,
    pub w1: f64,
    pub dist: f64,
    pub aug: i32,
    pub rows: i32,
    pub cols: i32,
}

const MAT_LIMIT: usize = 1 << 23;
const MAX_PIVOTS: i32 = 1 << 26;

#[derive(Default)]
pub struct LenWs<F: Prod<D>, D: Dist> {
    nb: Nbhd,
    rm: Vec<F>,
    cm: Vec<F>,
    rowv: Vec<u32>,
    colv: Vec<u32>,
    dst: Vec<u32>,
    fin: Vec<u32>,
    dval: Vec<D>,
    dep: u32,
    heap: Vec<(D, u32)>,
    tst: Vec<u32>,
    tloc: Vec<u32>,
    tep: u32,
    sst: Vec<u32>,
    sloc: Vec<u32>,
    sep: u32,
    slist: Vec<u32>,
    tlist: Vec<u32>,
    mat: Vec<D>,
    px: Vec<u32>,
    pm: Vec<F>,
    ny: Vec<u32>,
    nm: Vec<F>,
    cmat: Vec<D>,
    rkey: Vec<D>,
    ckey: Vec<D>,
    tc: Vec<D>,
    table: Vec<i32>,
    gid: Vec<u32>,
    leader: Vec<u32>,
    gid2: Vec<u32>,
    leader2: Vec<u32>,
    sa: Vec<F>,
    sb: Vec<F>,
    nflow: Vec<F>,
    nin: Vec<bool>,
    nadj: Vec<Vec<(u32, u32)>>,
    npar: Vec<u32>,
    npred: Vec<u32>,
    nfwd: Vec<bool>,
    ndep: Vec<u32>,
    npot: Vec<D>,
    nq: Vec<u32>,
    nra: Vec<F>,
    nrd: Vec<F>,
    nseen: Vec<bool>,
    pub n_dij: u64,
    pub n_settled: u64,
}

impl<F: Prod<D>, D: Dist> LenWs<F, D> {
    pub fn new(n: usize) -> Self {
        let mut w = Self::default();
        w.dst = vec![0; n];
        w.fin = vec![0; n];
        w.dval = vec![D::ZERO; n];
        w.tst = vec![0; n];
        w.tloc = vec![0; n];
        w.sst = vec![0; n];
        w.sloc = vec![0; n];
        w
    }

    fn fail() -> LenOut {
        LenOut { val: f64::NAN, num: -1, den: 0, dist_i: 0, w1: f64::NAN, dist: f64::NAN, aug: -1, rows: 0, cols: 0 }
    }

    fn next_dep(&mut self) -> u32 {
        if self.dep == u32::MAX {
            self.dst.fill(0);
            self.fin.fill(0);
            self.dep = 0;
        }
        self.dep += 1;
        self.dep
    }

    /// Dijkstra from s until the nt targets (tst == tep) are settled; row ``row`` of ``mat``.
    fn dij_targets(&mut self, g: &Graph, lens: &[D], s: usize, nt: usize, row: usize) {
        let t0 = prof_start();
        let dep = self.next_dep();
        let tep = self.tep;
        self.heap.clear();
        self.dst[s] = dep;
        self.dval[s] = D::ZERO;
        hpush(&mut self.heap, (D::ZERO, s as u32));
        let mut found = 0usize;
        self.n_dij += 1;
        while let Some((d, w)) = hpop(&mut self.heap) {
            let w = w as usize;
            if self.fin[w] == dep || d > self.dval[w] {
                continue;
            }
            self.fin[w] = dep;
            self.n_settled += 1;
            if self.tst[w] == tep {
                self.mat[row * nt + self.tloc[w] as usize] = d;
                found += 1;
                if found == nt {
                    break;
                }
            }
            for e in g.lo(w)..g.hi(w) {
                let v = g.at(e);
                if self.fin[v] == dep {
                    continue;
                }
                let nd = d + lens[e];
                if self.dst[v] != dep || nd < self.dval[v] {
                    self.dst[v] = dep;
                    self.dval[v] = nd;
                    hpush(&mut self.heap, (nd, v as u32));
                }
            }
        }
        prof_add(&T_DIST, t0);
    }

    /// d(i, j), at most ``bound`` (the length of the edge).
    fn pair_dist(&mut self, g: &Graph, lens: &[D], i: usize, j: usize, bound: D) -> D {
        let t0 = prof_start();
        let dep = self.next_dep();
        self.heap.clear();
        self.dst[i] = dep;
        self.dval[i] = D::ZERO;
        hpush(&mut self.heap, (D::ZERO, i as u32));
        while let Some((d, w)) = hpop(&mut self.heap) {
            let w = w as usize;
            if self.fin[w] == dep || d > self.dval[w] {
                continue;
            }
            self.fin[w] = dep;
            if w == j {
                prof_add(&T_DIST, t0);
                return d;
            }
            for e in g.lo(w)..g.hi(w) {
                let v = g.at(e);
                let nd = d + lens[e];
                if !(nd < bound) || self.fin[v] == dep {
                    continue;
                }
                if self.dst[v] != dep || nd < self.dval[v] {
                    self.dst[v] = dep;
                    self.dval[v] = nd;
                    hpush(&mut self.heap, (nd, v as u32));
                }
            }
        }
        prof_add(&T_DIST, t0);
        bound
    }

    fn prepare(&mut self, g: &Graph, m: &Meas, a: usize, b: usize) -> Option<i64> {
        self.nb.build(g, a, b);
        let l = F::masses(m, &self.nb, &mut self.rm, &mut self.cm)?;
        self.nb.rows_into(&mut self.rowv);
        self.nb.cols_into(&mut self.colv);
        self.px.clear();
        self.pm.clear();
        self.ny.clear();
        self.nm.clear();
        for r in 0..self.rowv.len() {
            if self.rm[r].mpos() {
                self.px.push(self.rowv[r]);
                self.pm.push(self.rm[r]);
            }
        }
        for k in 0..self.colv.len() {
            if self.cm[k].mpos() {
                self.ny.push(self.colv[k]);
                self.nm.push(self.cm[k]);
            }
        }
        Some(l)
    }

    /// One edge on its own: Dijkstras from the smaller of P and N.
    fn edge_single(&mut self, g: &Graph, lens: &[D], m: &Meas, a: usize, b: usize) -> LenOut {
        let l = match self.prepare(g, m, a, b) {
            Some(l) => l,
            None => return Self::fail(),
        };
        let (i, j) = (self.nb.i, self.nb.j);
        let dij = self.pair_dist(g, lens, i, j, lens[self.nb.pij]);
        let (r, c) = (self.px.len(), self.ny.len());
        self.cmat.clear();
        if r > 0 && c > 0 {
            let from_p = r <= c;
            let (ns, nt) = if from_p { (r, c) } else { (c, r) };
            let tep = bump(&mut self.tep, &mut self.tst);
            for t in 0..nt {
                let v = (if from_p { self.ny[t] } else { self.px[t] }) as usize;
                self.tst[v] = tep;
                self.tloc[v] = t as u32;
            }
            self.mat.clear();
            self.mat.resize(ns * nt, D::MAXV);
            for s in 0..ns {
                let sv = (if from_p { self.px[s] } else { self.ny[s] }) as usize;
                self.dij_targets(g, lens, sv, nt, s);
            }
            self.cmat.resize(r * c, D::ZERO);
            for x in 0..r {
                for y in 0..c {
                    self.cmat[x * c + y] = if from_p { self.mat[x * nt + y] } else { self.mat[y * nt + x] };
                }
            }
        }
        self.solve(l, dij)
    }

    /// One edge of a run that shares the endpoint u (distances from N[u] already in ``mat``).
    fn edge_in_block(&mut self, g: &Graph, m: &Meas, u: usize, v: usize, nt: usize) -> LenOut {
        let l = match self.prepare(g, m, u, v) {
            Some(l) => l,
            None => return Self::fail(),
        };
        let iu = self.nb.i == u;
        let dij = self.mat[self.sloc[u] as usize * nt + self.tloc[v] as usize];
        let (r, c) = (self.px.len(), self.ny.len());
        self.cmat.clear();
        self.cmat.resize(r * c, D::ZERO);
        for x in 0..r {
            let xv = self.px[x] as usize;
            for y in 0..c {
                let yv = self.ny[y] as usize;
                self.cmat[x * c + y] = if iu {
                    self.mat[self.sloc[xv] as usize * nt + self.tloc[yv] as usize]
                } else {
                    self.mat[self.sloc[yv] as usize * nt + self.tloc[xv] as usize]
                };
            }
        }
        self.solve(l, dij)
    }

    /// One edge, with the distances from every vertex of N[u] in the rows ``grow`` of ``mat``
    /// (single-source distances shared by all the edges of a chunk).
    pub fn edge_global(&mut self, g: &Graph, m: &Meas, u: usize, v: usize, mat: &[D], grow: &[u32]) -> LenOut {
        let n = g.n();
        let l = match self.prepare(g, m, u, v) {
            Some(l) => l,
            None => return Self::fail(),
        };
        let iu = self.nb.i == u;
        let dij = mat[grow[u] as usize * n + v];
        let (r, c) = (self.px.len(), self.ny.len());
        self.cmat.clear();
        self.cmat.resize(r * c, D::ZERO);
        for x in 0..r {
            let xv = self.px[x] as usize;
            for y in 0..c {
                let yv = self.ny[y] as usize;
                self.cmat[x * c + y] = if iu { mat[grow[xv] as usize * n + yv] } else { mat[grow[yv] as usize * n + xv] };
            }
        }
        self.solve(l, dij)
    }

    /// The transport step alone on a dense r x c cost matrix (row merging, column merging, network
    /// simplex): (W1, pivots). For benchmarks (``orcut._native._transport``).
    pub fn transport_dense(&mut self, cost: &[D], a: &[F], b: &[F], r: usize, c: usize) -> (f64, i32) {
        self.px.clear();
        self.pm.clear();
        self.ny.clear();
        self.nm.clear();
        for x in 0..r {
            self.px.push(x as u32);
            self.pm.push(a[x]);
        }
        for y in 0..c {
            self.ny.push(y as u32);
            self.nm.push(b[y]);
        }
        self.cmat.clear();
        self.cmat.extend_from_slice(&cost[..r * c]);
        let out = self.solve(1, D::MAXV);
        (out.w1, out.aug)
    }

    /// The edges ``ids`` (all with first endpoint u = ev[2 ids[0]]), into ``out``.
    pub fn block(&mut self, g: &Graph, lens: &[D], m: &Meas, ev: &[i64], ids: &[i64], out: &mut [LenOut]) {
        let k = ids.len();
        let ends = |x: usize| (ev[2 * ids[x] as usize] as usize, ev[2 * ids[x] as usize + 1] as usize);
        let u = ends(0).0;
        let du = g.deg(u) as usize;
        let mut est = 0usize;
        for x in 0..k {
            let (a, b) = ends(x);
            let o = if a == u { b } else { a };
            est += du.min(g.deg(o) as usize) + 1;
        }
        let mut use_block = k > 1 && du + 1 < est;
        let mut nt = 0usize;
        if use_block {
            let sep = bump(&mut self.sep, &mut self.sst);
            self.slist.clear();
            self.slist.push(u as u32);
            for e in g.lo(u)..g.hi(u) {
                self.slist.push(g.indices[e] as u32);
            }
            for s in 0..self.slist.len() {
                let v = self.slist[s] as usize;
                self.sst[v] = sep;
                self.sloc[v] = s as u32;
            }
            let tep = bump(&mut self.tep, &mut self.tst);
            self.tlist.clear();
            for s in 0..self.slist.len() {
                let w = self.slist[s] as usize;
                if self.tst[w] != tep {
                    self.tst[w] = tep;
                    self.tloc[w] = self.tlist.len() as u32;
                    self.tlist.push(w as u32);
                }
            }
            for x in 0..k {
                let (a, b) = ends(x);
                let o = if a == u { b } else { a };
                for e in g.lo(o)..g.hi(o) {
                    let w = g.at(e);
                    if self.tst[w] != tep {
                        self.tst[w] = tep;
                        self.tloc[w] = self.tlist.len() as u32;
                        self.tlist.push(w as u32);
                    }
                }
            }
            nt = self.tlist.len();
            if self.slist.len() * nt > MAT_LIMIT {
                use_block = false;
            }
        }
        if use_block {
            let ns = self.slist.len();
            self.mat.clear();
            self.mat.resize(ns * nt, D::MAXV);
            for s in 0..ns {
                let sv = self.slist[s] as usize;
                self.dij_targets(g, lens, sv, nt, s);
            }
            for x in 0..k {
                let (a, b) = ends(x);
                let o = if a == u { b } else { a };
                out[x] = self.edge_in_block(g, m, u, o, nt);
            }
        } else {
            for x in 0..k {
                let (a, b) = ends(x);
                out[x] = self.edge_single(g, lens, m, a, b);
            }
        }
    }

    /// Merge rows and columns that differ by a constant, then solve the transportation problem.
    fn solve(&mut self, l: i64, dij: D) -> LenOut {
        let t0 = prof_start();
        let (r, c) = (self.px.len(), self.ny.len());
        let mut cost = <F as Prod<D>>::Acc::default();
        let (mut aug, mut nr2, mut nc2) = (0i32, 0usize, 0usize);
        if r > 0 && c > 0 {
            self.rkey.clear();
            self.rkey.resize(r * c, D::ZERO);
            for x in 0..r {
                let mut mn = self.cmat[x * c];
                for y in 1..c {
                    let v = self.cmat[x * c + y];
                    if v < mn {
                        mn = v;
                    }
                }
                cost += self.pm[x].mul(mn);
                for y in 0..c {
                    self.rkey[x * c + y] = self.cmat[x * c + y] - mn;
                }
            }
            let ng = dedupe_d(&self.rkey, r, c, &mut self.table, &mut self.gid, &mut self.leader);
            self.sa.clear();
            self.sa.resize(ng, F::ZERO);
            for x in 0..r {
                let gg = self.gid[x] as usize;
                let mv = self.pm[x];
                self.sa[gg] += mv;
            }
            self.ckey.clear();
            self.ckey.resize(c * ng, D::ZERO);
            for y in 0..c {
                let mut mn = D::MAXV;
                for gg in 0..ng {
                    let v = self.rkey[self.leader[gg] as usize * c + y];
                    if v < mn {
                        mn = v;
                    }
                }
                cost += self.nm[y].mul(mn);
                for gg in 0..ng {
                    self.ckey[y * ng + gg] = self.rkey[self.leader[gg] as usize * c + y] - mn;
                }
            }
            let nh = dedupe_d(&self.ckey, c, ng, &mut self.table, &mut self.gid2, &mut self.leader2);
            self.sb.clear();
            self.sb.resize(nh, F::ZERO);
            for y in 0..c {
                let h = self.gid2[y] as usize;
                let mv = self.nm[y];
                self.sb[h] += mv;
            }
            self.tc.clear();
            self.tc.resize(ng * nh, D::ZERO);
            for h in 0..nh {
                let y = self.leader2[h] as usize;
                for gg in 0..ng {
                    self.tc[gg * nh + h] = self.ckey[y * ng + gg];
                }
            }
            match self.transport(ng, nh) {
                Some((t, a)) => {
                    cost += t;
                    aug = a;
                }
                None => return Self::fail(),
            }
            nr2 = ng;
            nc2 = nh;
        }
        let w1 = F::acc_f64(cost) / (l as f64);
        let df = dij.to_f64();
        let (val, num, den, dist_i) = if F::EXACT && D::EXACT {
            let di = dij.to_i64();
            match F::acc_i64(cost) {
                Some(num) => {
                    let dd = (l as i128) * (di as i128);
                    (1.0 - (num as f64) / (dd as f64), num, l, di)
                }
                None => (1.0 - w1 / df, -1, 0, di),
            }
        } else {
            (1.0 - w1 / df, -1, 0, if D::EXACT { dij.to_i64() } else { 0 })
        };
        prof_add(&T_SOLVE, t0);
        LenOut { val, num, den, dist_i, w1, dist: df, aug, rows: nr2 as i32, cols: nc2 as i32 }
    }

    /// The tree arc e between u and its neighbour v: (v, cost, v is the tail). Rows are 0..r, columns
    /// r..r+c, the root r+c; real arcs go row -> column (e = x c + y), artificial ones row -> root
    /// (cost 0) and root -> column (cost ``art``).
    #[inline(always)]
    fn ns_side(&self, e: usize, u: usize, v: usize, r: usize, c: usize, art: D) -> (D, bool) {
        let root = r + c;
        if u != root && v != root {
            (self.tc[e], v < r)
        } else {
            let w = if u == root { v } else { u };
            if w < r {
                (D::ZERO, v < r)
            } else {
                (art, v == root)
            }
        }
    }

    /// Hang the tree below ``es`` (whose parent fields are set) by a BFS: parents, depths,
    /// orientations, and potentials with zero reduced cost on tree arcs.
    fn ns_hang(&mut self, es: usize, r: usize, c: usize, art: D) {
        self.nq.clear();
        self.nq.push(es as u32);
        let mut h = 0usize;
        while h < self.nq.len() {
            let u = self.nq[h] as usize;
            h += 1;
            for q in 0..self.nadj[u].len() {
                let (e, v) = self.nadj[u][q];
                let (e, v) = (e as usize, v as usize);
                if v as u32 == self.npar[u] {
                    continue;
                }
                let (cst, v_tail) = self.ns_side(e, u, v, r, c, art);
                self.npar[v] = u as u32;
                self.npred[v] = e as u32;
                self.ndep[v] = self.ndep[u] + 1;
                self.nfwd[v] = v_tail;
                self.npot[v] = if v_tail { self.npot[u] - cst } else { self.npot[u] + cst };
                self.nq.push(v as u32);
            }
        }
    }

    /// Network simplex on the transportation problem ``tc`` (r x c; rows supply ``sa``, columns
    /// demand ``sb``): artificial starting basis at a root (as LEMON), block-search pricing, and the
    /// leaving arc of strongly feasible trees (the last blocking arc of the cycle from its apex), which
    /// rules out cycling on these degenerate problems. Exact with integer masses and costs.
    fn transport(&mut self, r: usize, c: usize) -> Option<(<F as Prod<D>>::Acc, i32)> {
        let mut cost = <F as Prod<D>>::Acc::default();
        if c == 1 {
            for x in 0..r {
                cost += self.sa[x].mul(self.tc[x]);
            }
            return Some((cost, 0));
        }
        if r == 1 {
            for y in 0..c {
                cost += self.sb[y].mul(self.tc[y]);
            }
            return Some((cost, 0));
        }
        let n = r + c + 1;
        let root = r + c;
        let nreal = r * c;
        let na = nreal + r + c;
        let mut maxc = D::ZERO;
        for e in 0..nreal {
            if self.tc[e] > maxc {
                maxc = self.tc[e];
            }
        }
        let art = D::art(maxc, n);
        let tol = D::ntol(art);
        self.nflow.clear();
        self.nflow.resize(na, F::ZERO);
        self.nin.clear();
        self.nin.resize(na, false);
        if self.nadj.len() < n {
            self.nadj.resize_with(n, Vec::new);
        }
        for v in 0..n {
            self.nadj[v].clear();
        }
        // starting basis: the row-minimum rule (each row to its cheapest columns with demand left)
        // gives a forest whose arcs all carry flow (two rows never share two columns, so no cycle);
        // each of its trees hangs from the root by a zero-flow arc root -> column, so every zero-flow
        // tree arc points away from the root: the basis is strongly feasible. Leftovers of float
        // rounding go on artificial arcs with positive flow.
        self.nra.clear();
        self.nra.extend_from_slice(&self.sa[..r]);
        self.nrd.clear();
        self.nrd.extend_from_slice(&self.sb[..c]);
        for x in 0..r {
            while self.nra[x].mpos() {
                let mut by = usize::MAX;
                let mut bc = D::MAXV;
                for y in 0..c {
                    if self.nrd[y].mpos() && self.tc[x * c + y] < bc {
                        bc = self.tc[x * c + y];
                        by = y;
                    }
                }
                if by == usize::MAX {
                    break;
                }
                let f = if self.nra[x] < self.nrd[by] { self.nra[x] } else { self.nrd[by] };
                let e = x * c + by;
                self.nin[e] = true;
                self.nflow[e] = f;
                self.nadj[x].push((e as u32, (r + by) as u32));
                self.nadj[r + by].push((e as u32, x as u32));
                self.nra[x] -= f;
                self.nrd[by] -= f;
            }
        }
        self.nseen.clear();
        self.nseen.resize(n, false);
        for v in (r..r + c).chain(0..r) {
            if self.nseen[v] {
                continue;
            }
            let e = nreal + v;
            self.nin[e] = true;
            self.nflow[e] = if v < r { self.nra[v] } else { self.nrd[v - r] };
            self.nadj[v].push((e as u32, root as u32));
            self.nadj[root].push((e as u32, v as u32));
            self.nq.clear();
            self.nq.push(v as u32);
            self.nseen[v] = true;
            let mut h = 0usize;
            while h < self.nq.len() {
                let u = self.nq[h] as usize;
                h += 1;
                for q in 0..self.nadj[u].len() {
                    let w = self.nadj[u][q].1 as usize;
                    if w != root && !self.nseen[w] {
                        self.nseen[w] = true;
                        self.nq.push(w as u32);
                    }
                }
            }
        }
        self.npar.resize(n, 0);
        self.npred.resize(n, 0);
        self.nfwd.resize(n, false);
        self.ndep.resize(n, 0);
        self.npot.resize(n, D::ZERO);
        self.npar[root] = u32::MAX;
        self.ndep[root] = 0;
        self.npot[root] = D::ZERO;
        self.ns_hang(root, r, c, art);
        let block = ((nreal as f64).sqrt().ceil() as usize).max(10);
        let mut nx = 0usize;
        let mut piv = 0i32;
        loop {
            // block search, by whole rows: the most negative reduced cost of the first rows that cover a
            // block and contain one. Tree arcs have zero reduced cost and never qualify.
            let mut best = usize::MAX;
            let mut bmin = D::ZERO - tol;
            let mut scanned = 0usize;
            let mut x = nx;
            {
                let pc = &self.npot[r..r + c];
                for _ in 0..r {
                    let px = self.npot[x];
                    let row = &self.tc[x * c..x * c + c];
                    let mut m = D::MAXV;
                    for y in 0..c {
                        let v = row[y] + px - pc[y];
                        if v < m {
                            m = v;
                        }
                    }
                    if m < bmin {
                        bmin = m;
                        for y in 0..c {
                            if row[y] + px - pc[y] == m {
                                best = x * c + y;
                                break;
                            }
                        }
                    }
                    scanned += c;
                    x += 1;
                    if x == r {
                        x = 0;
                    }
                    if scanned >= block && best != usize::MAX {
                        break;
                    }
                }
            }
            if best == usize::MAX {
                break;
            }
            nx = x;
            piv += 1;
            if piv > MAX_PIVOTS {
                return None;
            }
            let k = best / c;
            let l = r + best % c;
            let (mut a, mut b) = (k, l);
            while self.ndep[a] > self.ndep[b] {
                a = self.npar[a] as usize;
            }
            while self.ndep[b] > self.ndep[a] {
                b = self.npar[b] as usize;
            }
            while a != b {
                a = self.npar[a] as usize;
                b = self.npar[b] as usize;
            }
            let join = a;
            // leaving arc: the last blocking arc from the apex along the orientation of k -> l
            let mut delta = F::INF;
            let mut uout = usize::MAX;
            let mut on_k = true;
            let mut u = k;
            while u != join {
                if self.nfwd[u] {
                    let f = self.nflow[self.npred[u] as usize];
                    if f < delta {
                        delta = f;
                        uout = u;
                    }
                }
                u = self.npar[u] as usize;
            }
            let mut u = l;
            while u != join {
                if !self.nfwd[u] {
                    let f = self.nflow[self.npred[u] as usize];
                    if f <= delta {
                        delta = f;
                        uout = u;
                        on_k = false;
                    }
                }
                u = self.npar[u] as usize;
            }
            if uout == usize::MAX {
                return None;
            }
            if delta > F::ZERO {
                let mut u = k;
                while u != join {
                    let ea = self.npred[u] as usize;
                    if self.nfwd[u] {
                        self.nflow[ea] -= delta;
                    } else {
                        self.nflow[ea] += delta;
                    }
                    u = self.npar[u] as usize;
                }
                let mut u = l;
                while u != join {
                    let ea = self.npred[u] as usize;
                    if self.nfwd[u] {
                        self.nflow[ea] += delta;
                    } else {
                        self.nflow[ea] -= delta;
                    }
                    u = self.npar[u] as usize;
                }
            }
            self.nflow[best] = delta;
            let eout = self.npred[uout] as usize;
            self.nin[eout] = false;
            self.nin[best] = true;
            let pout = self.npar[uout] as usize;
            for w in [uout, pout] {
                if let Some(p) = self.nadj[w].iter().position(|&(z, _)| z as usize == eout) {
                    self.nadj[w].swap_remove(p);
                }
            }
            self.nadj[k].push((best as u32, l as u32));
            self.nadj[l].push((best as u32, k as u32));
            // the subtree below the leaving arc contains k (leaving arc on k's side) or l; it hangs
            // from the other endpoint of the entering arc now, and only its nodes change
            let (es, eo) = if on_k { (k, l) } else { (l, k) };
            self.npar[es] = eo as u32;
            self.npred[es] = best as u32;
            self.ndep[es] = self.ndep[eo] + 1;
            let cb = self.tc[best];
            if es == l {
                self.nfwd[es] = false;
                self.npot[es] = self.npot[eo] + cb;
            } else {
                self.nfwd[es] = true;
                self.npot[es] = self.npot[eo] - cb;
            }
            self.ns_hang(es, r, c, art);
        }
        for e in 0..nreal {
            if self.nin[e] {
                let f = self.nflow[e];
                if f > F::ZERO {
                    cost += f.mul(self.tc[e]);
                }
            }
        }
        Some((cost, piv))
    }
}

/// Scratch of a full single-source Dijkstra.
#[derive(Default)]
pub struct DijWs<D: Dist> {
    fin: Vec<u32>,
    ep: u32,
    heap: Vec<(D, u32)>,
}

impl<D: Dist> DijWs<D> {
    pub fn new(n: usize) -> Self {
        DijWs { fin: vec![0; n], ep: 0, heap: Vec::new() }
    }
}

/// Distances from s to every vertex of its component, into ``row`` (initialized to MAXV).
pub fn full_dijkstra<D: Dist>(g: &Graph, lens: &[D], s: usize, row: &mut [D], ws: &mut DijWs<D>) {
    let t0 = prof_start();
    let ep = bump(&mut ws.ep, &mut ws.fin);
    ws.heap.clear();
    row[s] = D::ZERO;
    hpush(&mut ws.heap, (D::ZERO, s as u32));
    while let Some((d, w)) = hpop(&mut ws.heap) {
        let w = w as usize;
        if ws.fin[w] == ep || d > row[w] {
            continue;
        }
        ws.fin[w] = ep;
        for e in g.lo(w)..g.hi(w) {
            let v = g.at(e);
            if ws.fin[v] == ep {
                continue;
            }
            let nd = d + lens[e];
            if nd < row[v] {
                row[v] = nd;
                hpush(&mut ws.heap, (nd, v as u32));
            }
        }
    }
    prof_add(&T_DIST, t0);
}

