//! Edge lengths: W1 for the shortest-path metric of positive edge lengths.
//!
//! For an edge (u, v) the vertices of positive net mass (P) lie in N[i] and those of negative net
//! mass (N) in N[j], {i, j} = {u, v}, so every pair of P x N has one vertex in N[u]. Distances come
//! from Dijkstras that stop as soon as all their targets are settled: for a run of edges that share
//! their first endpoint u (consecutive in ``graph.edges``), one Dijkstra from each vertex of N[u]
//! serves the whole run; otherwise each edge runs them from the smaller of P and N. Rows whose costs
//! differ by a constant are merged (the constant is paid exactly), then columns, and the reduced
//! transportation problem is solved by successive shortest paths with potentials. With integer
//! lengths and masses everything is exact (i64 costs, i128 totals) and equal rows merge exactly;
//! with real lengths the same steps run in floating point.
use crate::nbhd::{Mass, Meas, Nbhd};
use crate::Graph;
use std::ops::{Add, AddAssign, Sub, SubAssign};

pub trait Dist:
    Copy + Default + PartialOrd + Add<Output = Self> + Sub<Output = Self> + AddAssign + SubAssign + Send + Sync + 'static
{
    const ZERO: Self;
    const MAXV: Self;
    const EXACT: bool;
    fn to_f64(self) -> f64;
    fn to_i64(self) -> i64;
    fn bits(self) -> u64;
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
const MAX_AUG: i32 = 1 << 26;

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
    pi: Vec<F>,
    phi: Vec<D>,
    ra: Vec<F>,
    rd: Vec<F>,
    tdist: Vec<D>,
    tdone: Vec<bool>,
    tpred: Vec<u32>,
    theap: Vec<(D, u32)>,
    tsettled: Vec<u32>,
    insup: Vec<bool>,
    supp: Vec<Vec<u32>>,
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
    }

    /// d(i, j), at most ``bound`` (the length of the edge).
    fn pair_dist(&mut self, g: &Graph, lens: &[D], i: usize, j: usize, bound: D) -> D {
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
        LenOut { val, num, den, dist_i, w1, dist: df, aug, rows: nr2 as i32, cols: nc2 as i32 }
    }

    /// Successive shortest paths on the bipartite residual network of ``tc`` (r x c): one row at a
    /// time as the source, Dijkstra on reduced costs up to the nearest column with demand left, with
    /// the reverse arcs of a column read from its support (the rows that send it flow).
    fn transport(&mut self, r: usize, c: usize) -> Option<(<F as Prod<D>>::Acc, i32)> {
        let nv = r + c;
        self.pi.clear();
        self.pi.resize(r * c, F::ZERO);
        self.insup.clear();
        self.insup.resize(r * c, false);
        if self.supp.len() < c {
            self.supp.resize_with(c, Vec::new);
        }
        for y in 0..c {
            self.supp[y].clear();
        }
        self.phi.clear();
        self.phi.resize(nv, D::ZERO);
        self.ra.clear();
        self.ra.extend_from_slice(&self.sa);
        self.rd.clear();
        self.rd.extend_from_slice(&self.sb);
        self.tdist.resize(nv, D::ZERO);
        self.tdone.resize(nv, false);
        self.tpred.resize(nv, u32::MAX);
        let mut cost = <F as Prod<D>>::Acc::default();
        let mut aug = 0i32;
        for s0 in 0..r {
            while self.ra[s0].mpos() {
                if aug >= MAX_AUG {
                    return None;
                }
                for w in 0..nv {
                    self.tdist[w] = D::MAXV;
                    self.tdone[w] = false;
                    self.tpred[w] = u32::MAX;
                }
                self.theap.clear();
                self.tsettled.clear();
                self.tdist[s0] = D::ZERO;
                hpush(&mut self.theap, (D::ZERO, s0 as u32));
                let mut target = usize::MAX;
                while let Some((d, w)) = hpop(&mut self.theap) {
                    let w = w as usize;
                    if self.tdone[w] || d > self.tdist[w] {
                        continue;
                    }
                    self.tdone[w] = true;
                    self.tsettled.push(w as u32);
                    if w >= r {
                        let y = w - r;
                        if self.rd[y].mpos() {
                            target = w;
                            break;
                        }
                        let py = self.phi[w];
                        for q in 0..self.supp[y].len() {
                            let x = self.supp[y][q] as usize;
                            if self.tdone[x] || !self.pi[x * c + y].pos() {
                                continue;
                            }
                            let mut rc = py - self.phi[x] - self.tc[x * c + y];
                            if rc < D::ZERO {
                                rc = D::ZERO;
                            }
                            let nd = d + rc;
                            if nd < self.tdist[x] {
                                self.tdist[x] = nd;
                                self.tpred[x] = w as u32;
                                hpush(&mut self.theap, (nd, x as u32));
                            }
                        }
                    } else {
                        let px = self.phi[w];
                        for y in 0..c {
                            let wy = r + y;
                            if self.tdone[wy] {
                                continue;
                            }
                            let mut rc = self.tc[w * c + y] + px - self.phi[wy];
                            if rc < D::ZERO {
                                rc = D::ZERO;
                            }
                            let nd = d + rc;
                            if nd < self.tdist[wy] {
                                self.tdist[wy] = nd;
                                self.tpred[wy] = w as u32;
                                hpush(&mut self.theap, (nd, wy as u32));
                            }
                        }
                    }
                }
                if target == usize::MAX {
                    return None;
                }
                let delta = self.tdist[target];
                for idx in 0..self.tsettled.len() {
                    let w = self.tsettled[idx] as usize;
                    let dw = self.tdist[w];
                    self.phi[w] += dw - delta;
                }
                let mut amt = self.rd[target - r];
                if self.ra[s0] < amt {
                    amt = self.ra[s0];
                }
                let mut w = target;
                while self.tpred[w] != u32::MAX {
                    let p = self.tpred[w] as usize;
                    if w < r {
                        let f = self.pi[w * c + (p - r)];
                        if f < amt {
                            amt = f;
                        }
                    }
                    w = p;
                }
                let mut w = target;
                while self.tpred[w] != u32::MAX {
                    let p = self.tpred[w] as usize;
                    if w >= r {
                        let (x, y) = (p, w - r);
                        let k = x * c + y;
                        self.pi[k] += amt;
                        if !self.insup[k] {
                            self.insup[k] = true;
                            self.supp[y].push(x as u32);
                        }
                        cost += amt.mul(self.tc[k]);
                    } else {
                        let k = w * c + (p - r);
                        self.pi[k] -= amt;
                        cost -= amt.mul(self.tc[k]);
                    }
                    w = p;
                }
                self.ra[s0] -= amt;
                self.rd[target - r] -= amt;
                aug += 1;
            }
        }
        Some((cost, aug))
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
}
