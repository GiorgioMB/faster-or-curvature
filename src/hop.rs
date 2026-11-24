//! One-step measures on the hop metric: the closure network of `orcut._core` (bitsets over the
//! columns, hubs H1--H4, twin merging, the one-layer fast path, witness relays, the warm start),
//! and a linear-size witness network (one relay per common neighbour, as `orcut._witness`) for
//! the edges whose bitsets would cost more than four times their local input. W1 is unique, so
//! the exact fractions (and the float values computed from them) equal those of the numba kernel.
use crate::flow::{Cap, Net};
use crate::nbhd::{Mass, Meas, Nbhd};
use crate::{EdgeOut, Graph};

pub const PATH_ZERO: i8 = 0;
pub const PATH_NOADJ: i8 = 1;
pub const PATH_FAST: i8 = 2;
pub const PATH_CUT: i8 = 3;
pub const PATH_CUTW: i8 = 4;
pub const PATH_WIT: i8 = 5;
pub const PATH_OVERFLOW: i8 = -1;

const BUDGET: usize = 1 << 22;
const DISPATCH_K: i64 = 4;
const WIT_MIN: i64 = 256;
const WARM_MIN_ARCS: usize = 300;
const H1: usize = 2;
const H2: usize = 3;
const H3: usize = 4;
const H4: usize = 5;

#[inline(always)]
fn pc(x: u64) -> i64 {
    x.count_ones() as i64
}

#[inline(always)]
fn low_mask(n: usize, x: usize) -> u64 {
    let lo = x * 64;
    if lo + 64 <= n {
        !0u64
    } else if lo < n {
        (1u64 << (n - lo)) - 1
    } else {
        0
    }
}

/// Group equal rows of a (nrows, kw) key matrix (as `orcut._flow._dedupe`).
fn dedupe(keys: &[u64], nrows: usize, kw: usize, table: &mut Vec<i32>, gid: &mut Vec<u32>, leader: &mut Vec<u32>) -> usize {
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
            h = (h ^ keys[base + x]).wrapping_mul(1099511628211);
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
            if keys[lb..lb + kw] == keys[base..base + kw] {
                gid[r] = g as u32;
                break;
            }
            p = (p + 1) & (tsize - 1);
        }
    }
    ng
}

/// The same for variable-length keys codes[cstart[r]..cstart[r + 1]].
fn dedupe_var(codes: &[u64], cstart: &[usize], nrows: usize, table: &mut Vec<i32>, gid: &mut Vec<u32>, leader: &mut Vec<u32>) -> usize {
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
        let key = &codes[cstart[r]..cstart[r + 1]];
        let mut h: u64 = 1469598103934665603 ^ (key.len() as u64);
        for &c in key {
            h = (h ^ c).wrapping_mul(1099511628211);
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
            let l = leader[g as usize] as usize;
            if &codes[cstart[l]..cstart[l + 1]] == key {
                gid[r] = g as u32;
                break;
            }
            p = (p + 1) & (tsize - 1);
        }
    }
    ng
}

#[derive(Default)]
struct WitWs<C: Cap> {
    cst: Vec<u32>,
    cpos: Vec<u32>,
    wstp: Vec<u32>,
    wslot: Vec<u32>,
    cdem: Vec<C>,
    wcnt: Vec<u32>,
    wstart: Vec<u32>,
    wfill: Vec<u32>,
    wcols: Vec<u32>,
    rcnt: Vec<u32>,
    rnode: Vec<i32>,
    codes: Vec<u64>,
    cstart: Vec<usize>,
    rmass: Vec<C>,
    gid: Vec<u32>,
    leader: Vec<u32>,
    table: Vec<i32>,
    gm: Vec<C>,
    net: Net<C>,
}

impl<C: Cap> WitWs<C> {
    fn ensure(&mut self, n: usize) {
        if self.cst.len() < n {
            self.cst = vec![0; n];
            self.cpos = vec![0; n];
            self.wstp = vec![0; n];
            self.wslot = vec![0; n];
        }
    }
    fn reset(&mut self) {
        self.cst.fill(0);
        self.wstp.fill(0);
    }
}

#[derive(Default)]
pub struct HopWs<C: Mass> {
    n: usize,
    slot: Vec<u32>,
    sstamp: Vec<u32>,
    scnt: Vec<u32>,
    epoch: u32,
    nb: Nbhd,
    rowv: Vec<u32>,
    colv: Vec<u32>,
    rm: Vec<C>,
    cm: Vec<C>,
    ujmask: Vec<u64>,
    adj: Vec<u64>,
    cov: Vec<u64>,
    rb: Vec<u64>,
    nm: Vec<u64>,
    colfull: Vec<u64>,
    pcm: Vec<u64>,
    keys: Vec<u64>,
    wc: Vec<i64>,
    fullr: Vec<bool>,
    rid: Vec<u32>,
    gid: Vec<u32>,
    leader: Vec<u32>,
    table: Vec<i32>,
    gm: Vec<C>,
    gkind: Vec<u8>,
    cidx: Vec<i32>,
    cnode: Vec<i32>,
    srow: Vec<i64>,
    scol: Vec<i64>,
    snode: Vec<i32>,
    wst: Vec<usize>,
    wl: Vec<u32>,
    tsp: Vec<u32>,
    tg1: Vec<u32>,
    tcol: Vec<u32>,
    net: Net<C>,
    net1: Net<C>,
    wit: WitWs<C>,
}

impl<C: Mass> HopWs<C> {
    pub fn new(n: usize) -> Self {
        let mut w = Self::default();
        w.n = n;
        w.slot = vec![0; n];
        w.sstamp = vec![0; n];
        w.scnt = vec![0; n];
        w
    }

    fn tick(&mut self) -> u32 {
        if self.epoch == u32::MAX {
            self.sstamp.fill(0);
            self.wit.reset();
            self.epoch = 0;
        }
        self.epoch += 1;
        self.epoch
    }

    pub fn edge(&mut self, g: &Graph, m: &Meas, i0: usize, j0: usize) -> EdgeOut {
        self.nb.build(g, i0, j0);
        let l = match C::masses(m, &self.nb, &mut self.rm, &mut self.cm) {
            Some(l) => l,
            None => return EdgeOut { val: f64::NAN, num: -1, den: 0, path: PATH_OVERFLOW },
        };
        let (nui, nuj, nt) = (self.nb.ui.len(), self.nb.uj.len(), self.nb.tt.len());
        let nr = nui + 2 + nt;
        let nc = nuj + 2 + nt;
        let w = (nc + 63) >> 6;
        self.nb.rows_into(&mut self.rowv);
        self.nb.cols_into(&mut self.colv);
        let ep = self.tick();
        // middle vertices: every vertex adjacent to a column, with the number of such columns
        let mut ns = 0usize;
        let mut msum: i64 = 0;
        for kk in 0..nc {
            let bb = self.colv[kk] as usize;
            msum += g.deg(bb);
            for e in g.lo(bb)..g.hi(bb) {
                let v = g.at(e);
                if self.sstamp[v] != ep {
                    self.sstamp[v] = ep;
                    self.slot[v] = ns as u32;
                    self.scnt[v] = 0;
                    ns += 1;
                }
                self.scnt[v] += 1;
            }
        }
        for r in 0..nr {
            msum += g.deg(self.rowv[r] as usize);
        }
        let lim = DISPATCH_K * msum;
        let (wd, path) = if ((ns + nr) * w) as i64 > lim
            || ns * w > BUDGET
            || !self.coverage(g, nui, nuj, nr, nc, w, ns, ep, lim)
        {
            self.witness(g, nr, nc, ep)
        } else {
            self.measure(g, nui, nuj, nt, w, ns, ep)
        };
        let (val, num, den) = C::finish(wd, l, m, &self.nb);
        EdgeOut { val, num, den, path }
    }

    /// Bitsets: columns adjacent to each middle vertex (rb) and to each row (adj), and the U_j
    /// columns within distance 2 of each U_i row (cov). False if the work exceeds ``lim``.
    #[allow(clippy::too_many_arguments)]
    fn coverage(&mut self, g: &Graph, nui: usize, nuj: usize, nr: usize, nc: usize, w: usize, ns: usize, ep: u32, lim: i64) -> bool {
        self.ujmask.clear();
        for x in 0..w {
            self.ujmask.push(low_mask(nuj, x));
        }
        self.wc.clear();
        self.wc.resize(nui, 0);
        self.adj.clear();
        self.adj.resize(nr * w, 0);
        self.cov.clear();
        self.cov.resize(nui * w, 0);
        self.rb.clear();
        self.rb.resize(ns * w, 0);
        for kk in 0..nc {
            let wq = kk >> 6;
            let bit = 1u64 << (kk & 63);
            let bb = self.colv[kk] as usize;
            for e in g.lo(bb)..g.hi(bb) {
                let s = self.slot[g.at(e)] as usize;
                self.rb[s * w + wq] |= bit;
            }
        }
        for r in 0..nr {
            let av = self.rowv[r] as usize;
            if self.sstamp[av] == ep {
                let sb = self.slot[av] as usize * w;
                for x in 0..w {
                    self.adj[r * w + x] = self.rb[sb + x];
                }
            }
        }
        if nuj > 0 {
            let (mut ops, mut nfull, mut nseen) = (0i64, 0i64, 0i64);
            for r in 0..nui {
                let cb = r * w;
                nseen += 1;
                let mut done = true;
                for x in 0..w {
                    let c = self.adj[cb + x];
                    self.cov[cb + x] = c;
                    if c & self.ujmask[x] != self.ujmask[x] {
                        done = false;
                    }
                }
                if done {
                    nfull += 1;
                    continue;
                }
                let xv = self.rowv[r] as usize;
                for e in g.lo(xv)..g.hi(xv) {
                    let v = g.at(e);
                    if self.sstamp[v] != ep {
                        continue;
                    }
                    self.wc[r] += 1;
                    let sb = self.slot[v] as usize * w;
                    let mut done = true;
                    for x in 0..w {
                        let c = self.cov[cb + x] | self.rb[sb + x];
                        self.cov[cb + x] = c;
                        if c & self.ujmask[x] != self.ujmask[x] {
                            done = false;
                        }
                    }
                    if done {
                        nfull += 1;
                        break;
                    }
                    ops += w as i64;
                    if ops > lim && 2 * nfull < nseen {
                        return false;
                    }
                }
            }
        }
        self.fullr.clear();
        self.fullr.resize(nui, false);
        self.colfull.clear();
        for x in 0..w {
            let v = if nui > 0 { self.ujmask[x] } else { 0 };
            self.colfull.push(v);
        }
        for r in 0..nui {
            let mut f = true;
            for x in 0..w {
                let c = self.cov[r * w + x] & self.ujmask[x];
                self.cov[r * w + x] = c;
                self.colfull[x] &= c;
                if c != self.ujmask[x] {
                    f = false;
                }
            }
            self.fullr[r] = f;
        }
        true
    }

    /// The fast path or the two-layer closure network (`orcut._core._measure_arr` with F_ALL).
    #[allow(clippy::too_many_arguments)]
    fn measure(&mut self, g: &Graph, nui: usize, nuj: usize, nt: usize, w: usize, ns: usize, ep: u32) -> (C, i8) {
        let nr = nui + 2 + nt;
        let nc = nuj + 2 + nt;
        let mut a = C::ZERO;
        for r in 0..nr {
            if self.rm[r].mpos() {
                a += self.rm[r];
            }
        }
        if !a.pos() {
            return (C::ZERO, PATH_ZERO);
        }
        self.nm.clear();
        self.nm.resize(w, 0);
        let (mut nnc, mut npc) = (0usize, 0usize);
        for k in 0..nc {
            if self.cm[k].mpos() {
                self.nm[k >> 6] |= 1u64 << (k & 63);
                if k >= nuj {
                    nnc += 1;
                }
            }
        }
        for r in nui..nr {
            if self.rm[r].mpos() {
                npc += 1;
            }
        }
        let nfull = self.fullr.iter().filter(|&&f| f).count();
        if nfull == nui {
            // ---- fast path: every pair of P x N is within distance 2, W1 = 2A - maxflow(G1)
            self.keys.clear();
            self.keys.resize(nr * w, 0);
            self.rid.clear();
            let mut ncand = 0usize;
            for r in 0..nr {
                if !self.rm[r].mpos() {
                    continue;
                }
                let mut nz = false;
                for x in 0..w {
                    let z = self.adj[r * w + x] & self.nm[x];
                    self.keys[ncand * w + x] = z;
                    nz |= z != 0;
                }
                if nz {
                    self.rid.push(r as u32);
                    ncand += 1;
                }
            }
            if ncand == 0 {
                return (a + a, PATH_NOADJ);
            }
            let ng = dedupe(&self.keys, ncand, w, &mut self.table, &mut self.gid, &mut self.leader);
            self.gm.clear();
            self.gm.resize(ng, C::ZERO);
            for c in 0..ncand {
                let gg = self.gid[c] as usize;
                let mv = self.rm[self.rid[c] as usize];
                self.gm[gg] += mv;
            }
            self.cnode.clear();
            self.cnode.resize(nc, -1);
            self.net.clear();
            let mut nv = 2 + ng;
            for gg in 0..ng {
                let gmv = self.gm[gg];
                self.net.arc(0, 2 + gg, gmv);
                let lb = self.leader[gg] as usize * w;
                for x in 0..w {
                    let mut z = self.keys[lb + x];
                    while z != 0 {
                        let k = x * 64 + z.trailing_zeros() as usize;
                        if self.cnode[k] < 0 {
                            self.cnode[k] = nv as i32;
                            nv += 1;
                        }
                        let cn = self.cnode[k] as usize;
                        self.net.arc(2 + gg, cn, C::INF);
                        z &= z - 1;
                    }
                }
            }
            for k in 0..nc {
                if self.cnode[k] >= 0 {
                    let (cn, d) = (self.cnode[k] as usize, self.cm[k]);
                    self.net.arc(cn, 1, d);
                }
            }
            self.net.build(nv);
            let f = self.net.dinic(nv, 0, 1);
            return (a + a - f, PATH_FAST);
        }

        // ---- general path: token network (p, q per row class; r, t per column)
        let kw = 2 * w + 1;
        self.keys.clear();
        self.keys.resize(nr * kw, 0);
        self.rid.clear();
        let mut ncand = 0usize;
        for r in 0..nr {
            if !self.rm[r].mpos() {
                continue;
            }
            let b0 = ncand * kw;
            for x in 0..w {
                self.keys[b0 + x] = self.adj[r * w + x] & self.nm[x];
            }
            if r < nui {
                for x in 0..w {
                    self.keys[b0 + w + x] = self.cov[r * w + x];
                }
                self.keys[b0 + 2 * w] = 0;
            } else {
                self.keys[b0 + 2 * w] = 1;
            }
            self.rid.push(r as u32);
            ncand += 1;
        }
        let ng = dedupe(&self.keys, ncand, kw, &mut self.table, &mut self.gid, &mut self.leader);
        self.gm.clear();
        self.gm.resize(ng, C::ZERO);
        self.gkind.clear();
        self.gkind.resize(ng, 0);
        for gg in 0..ng {
            let ld = self.leader[gg] as usize;
            self.gkind[gg] = if self.keys[ld * kw + 2 * w] == 1 {
                2
            } else if self.fullr[self.rid[ld] as usize] {
                1
            } else {
                0
            };
        }
        for c in 0..ncand {
            let gg = self.gid[c] as usize;
            let mv = self.rm[self.rid[c] as usize];
            self.gm[gg] += mv;
        }
        self.cidx.clear();
        self.cidx.resize(nc, -1);
        let mut nn = 0usize;
        for k in 0..nc {
            if self.cm[k].mpos() {
                self.cidx[k] = nn as i32;
                nn += 1;
            }
        }
        let (mut n_rowfull, mut n_rowpart) = (0usize, 0usize);
        for gg in 0..ng {
            match self.gkind[gg] {
                1 => n_rowfull += 1,
                0 => n_rowpart += 1,
                _ => {}
            }
        }
        let mut n_colfull = 0i64;
        for x in 0..w {
            n_colfull += pc(self.colfull[x]);
        }
        let use_h4 = n_colfull > 0 && n_rowpart > 0;
        // witness relays for the remaining pairs at distance 2, when they need fewer arcs
        let mut e2r = 0i64;
        if n_rowpart > 0 {
            for gg in 0..ng {
                if self.gkind[gg] != 0 {
                    continue;
                }
                let lb = self.leader[gg] as usize * kw;
                for x in 0..w {
                    e2r += pc(self.keys[lb + w + x] & !self.keys[lb + x] & !self.colfull[x]);
                }
            }
        }
        let mut tryw = e2r >= WIT_MIN;
        if tryw {
            let mut inc = 0i64;
            for gg in 0..ng {
                if self.gkind[gg] == 0 {
                    inc += self.wc[self.rid[self.leader[gg] as usize] as usize];
                }
            }
            tryw = 4 * inc < e2r;
        }
        let mut witness = false;
        let mut nh = 0usize;
        if tryw {
            self.srow.clear();
            self.srow.resize(ns, 0);
            self.scol.clear();
            self.scol.resize(ns, -1);
            self.snode.clear();
            self.snode.resize(ns, -1);
            self.wst.clear();
            self.wst.resize(nui + 1, 0);
            for gg in 0..ng {
                if self.gkind[gg] != 0 {
                    continue;
                }
                let r = self.rid[self.leader[gg] as usize] as usize;
                let xv = self.rowv[r] as usize;
                let mut cnt = 0usize;
                for e in g.lo(xv)..g.hi(xv) {
                    if self.sstamp[g.at(e)] == ep {
                        cnt += 1;
                    }
                }
                self.wst[r + 1] = cnt;
            }
            for r in 0..nui {
                self.wst[r + 1] += self.wst[r];
            }
            self.wl.clear();
            self.wl.resize(self.wst[nui], 0);
            for gg in 0..ng {
                if self.gkind[gg] != 0 {
                    continue;
                }
                let r = self.rid[self.leader[gg] as usize] as usize;
                let xv = self.rowv[r] as usize;
                let mut z = self.wst[r];
                for e in g.lo(xv)..g.hi(xv) {
                    let v = g.at(e);
                    if self.sstamp[v] == ep {
                        self.wl[z] = v as u32;
                        z += 1;
                        self.srow[self.slot[v] as usize] += 1;
                    }
                }
            }
            let mut ub = 0i64;
            for r in 0..nui {
                for z in self.wst[r]..self.wst[r + 1] {
                    let v = self.wl[z] as usize;
                    let sl = self.slot[v] as usize;
                    if self.scol[sl] == -1 {
                        self.scol[sl] = -2;
                        let sc = self.scnt[v] as i64;
                        if self.srow[sl] == 1 || sc == 1 {
                            ub += self.srow[sl] * sc;
                        } else {
                            ub += self.srow[sl] + sc + 1;
                        }
                    }
                }
            }
            if 4 * ub < e2r {
                witness = true;
                self.pcm.clear();
                for x in 0..w {
                    let v = low_mask(nuj, x) & self.nm[x] & !self.colfull[x];
                    self.pcm.push(v);
                }
                for sl in 0..ns {
                    if self.scol[sl] == -2 {
                        let mut cnt = 0i64;
                        for x in 0..w {
                            cnt += pc(self.rb[sl * w + x] & self.pcm[x]);
                        }
                        self.scol[sl] = cnt;
                    } else {
                        self.scol[sl] = 0;
                    }
                    if self.srow[sl] > 1 && self.scol[sl] > 1 {
                        self.snode[sl] = nh as i32;
                        nh += 1;
                    } else {
                        self.snode[sl] = -1;
                    }
                }
            }
        }
        // nodes: s = 0, t = 1, hubs H1..H4, rows (p, q), columns (r, t), witness relays
        let cbase = 6 + 2 * ng;
        let hbase = cbase + 2 * nn;
        let nv = hbase + nh;
        self.net.clear();
        self.tsp.clear();
        self.tg1.clear();
        self.tcol.clear();
        for gg in 0..ng {
            let pn = 6 + 2 * gg;
            let qn = pn + 1;
            let gmv = self.gm[gg];
            let k0 = self.net.arc(0, pn, gmv);
            self.tsp.push(k0);
            self.net.arc(0, qn, gmv);
            self.net.arc(qn, pn, C::INF);
            let lb = self.leader[gg] as usize * kw;
            let kind = self.gkind[gg];
            for x in 0..w {
                let mut z = self.keys[lb + x];
                while z != 0 {
                    let c = self.cidx[x * 64 + z.trailing_zeros() as usize] as usize;
                    let k1 = self.net.arc(pn, cbase + 2 * c, C::INF);
                    self.tg1.push(k1);
                    self.net.arc(qn, cbase + 2 * c + 1, C::INF);
                    z &= z - 1;
                }
                if kind == 0 && !witness {
                    let mut z = self.keys[lb + w + x] & !self.keys[lb + x] & !self.colfull[x];
                    while z != 0 {
                        let c = self.cidx[x * 64 + z.trailing_zeros() as usize] as usize;
                        self.net.arc(qn, cbase + 2 * c, C::INF);
                        z &= z - 1;
                    }
                }
            }
            if witness && kind == 0 {
                let r = self.rid[self.leader[gg] as usize] as usize;
                for z0 in self.wst[r]..self.wst[r + 1] {
                    let sl = self.slot[self.wl[z0] as usize] as usize;
                    if self.scol[sl] <= 0 {
                        continue;
                    }
                    if self.snode[sl] >= 0 {
                        let hn = hbase + self.snode[sl] as usize;
                        self.net.arc(qn, hn, C::INF);
                    } else {
                        for x in 0..w {
                            let mut z = self.rb[sl * w + x] & self.pcm[x];
                            while z != 0 {
                                let c = self.cidx[x * 64 + z.trailing_zeros() as usize] as usize;
                                self.net.arc(qn, cbase + 2 * c, C::INF);
                                z &= z - 1;
                            }
                        }
                    }
                }
            }
            if nnc > 0 {
                self.net.arc(qn, H1, C::INF);
            }
            if kind == 2 && npc > 0 {
                self.net.arc(qn, H2, C::INF);
            }
            if kind == 1 {
                self.net.arc(qn, H3, C::INF);
            }
            if kind == 0 && use_h4 {
                self.net.arc(qn, H4, C::INF);
            }
        }
        if witness {
            for sl in 0..ns {
                if self.snode[sl] < 0 {
                    continue;
                }
                let hn = hbase + self.snode[sl] as usize;
                for x in 0..w {
                    let mut z = self.rb[sl * w + x] & self.pcm[x];
                    while z != 0 {
                        let c = self.cidx[x * 64 + z.trailing_zeros() as usize] as usize;
                        self.net.arc(hn, cbase + 2 * c, C::INF);
                        z &= z - 1;
                    }
                }
            }
        }
        for k in 0..nc {
            let c = self.cidx[k];
            if c < 0 {
                continue;
            }
            let rn = cbase + 2 * c as usize;
            let tn = rn + 1;
            let mk = self.cm[k];
            let k2 = self.net.arc(rn, 1, mk);
            self.tcol.push(k2);
            self.net.arc(tn, 1, mk);
            if nnc > 0 && k >= nuj {
                self.net.arc(H1, rn, C::INF);
            }
            if npc > 0 {
                self.net.arc(H2, rn, C::INF);
            }
            if n_rowfull > 0 && k < nuj {
                self.net.arc(H3, rn, C::INF);
            }
            if use_h4 && k < nuj && (self.colfull[k >> 6] >> (k & 63)) & 1 != 0 {
                self.net.arc(H4, rn, C::INF);
            }
        }
        // warm start: a maximum flow of the adjacency layer, loaded on both layers
        let warm = self.net.n_arcs() >= WARM_MIN_ARCS;
        let ng1 = self.tg1.len();
        let mut f1 = C::ZERO;
        if warm {
            self.net1.clear();
            for gg in 0..ng {
                let gmv = self.gm[gg];
                self.net1.arc(0, 2 + gg, gmv);
            }
            for gg in 0..ng {
                let lb = self.leader[gg] as usize * kw;
                for x in 0..w {
                    let mut z = self.keys[lb + x];
                    while z != 0 {
                        let c = self.cidx[x * 64 + z.trailing_zeros() as usize] as usize;
                        self.net1.arc(2 + gg, 2 + ng + c, C::INF);
                        z &= z - 1;
                    }
                }
            }
            for k in 0..nc {
                let c = self.cidx[k];
                if c >= 0 {
                    let d = self.cm[k];
                    self.net1.arc(2 + ng + c as usize, 1, d);
                }
            }
            let nv1 = 2 + ng + nn;
            self.net1.build(nv1);
            f1 = self.net1.dinic(nv1, 0, 1);
        }
        self.net.build(nv);
        if warm {
            for gg in 0..ng {
                let f = self.net1.flow_of(gg);
                if f > C::ZERO {
                    let k = self.tsp[gg] as usize;
                    self.net.push_arc(k, f);
                    self.net.push_arc(k + 1, f);
                }
            }
            for x in 0..ng1 {
                let f = self.net1.flow_of(ng + x);
                if f > C::ZERO {
                    let k = self.tg1[x] as usize;
                    self.net.push_arc(k, f);
                    self.net.push_arc(k + 1, f);
                }
            }
            for c in 0..nn {
                let f = self.net1.flow_of(ng + ng1 + c);
                if f > C::ZERO {
                    let k = self.tcol[c] as usize;
                    self.net.push_arc(k, f);
                    self.net.push_arc(k + 1, f);
                }
            }
        }
        let f2 = self.net.dinic(nv, 0, 1);
        (a + a + a - (f1 + f1 + f2), if witness { PATH_CUTW } else { PATH_CUT })
    }

    /// Linear-size witness network: adjacency arcs, and one relay per vertex adjacent to both
    /// sides (direct arcs when it reaches one row class or one column).
    fn witness(&mut self, g: &Graph, nr: usize, nc: usize, ep: u32) -> (C, i8) {
        let n = self.n;
        let Self { ref mut wit, ref rowv, ref colv, ref rm, ref cm, .. } = *self;
        let mut a = C::ZERO;
        for r in 0..nr {
            if rm[r].mpos() {
                a += rm[r];
            }
        }
        if !a.pos() {
            return (C::ZERO, PATH_ZERO);
        }
        wit.ensure(n);
        wit.cdem.clear();
        for k in 0..nc {
            if !cm[k].mpos() {
                continue;
            }
            let y = colv[k] as usize;
            wit.cst[y] = ep;
            wit.cpos[y] = wit.cdem.len() as u32;
            wit.cdem.push(cm[k]);
        }
        let ncol = wit.cdem.len();
        wit.wcnt.clear();
        for k in 0..nc {
            if !cm[k].mpos() {
                continue;
            }
            let y = colv[k] as usize;
            for e in g.lo(y)..g.hi(y) {
                let v = g.at(e);
                if wit.wstp[v] != ep {
                    wit.wstp[v] = ep;
                    wit.wslot[v] = wit.wcnt.len() as u32;
                    wit.wcnt.push(0);
                }
                let s = wit.wslot[v] as usize;
                wit.wcnt[s] += 1;
            }
        }
        let nws = wit.wcnt.len();
        wit.wstart.clear();
        wit.wstart.resize(nws + 1, 0);
        for s in 0..nws {
            wit.wstart[s + 1] = wit.wstart[s] + wit.wcnt[s];
        }
        wit.wfill.clear();
        wit.wfill.extend_from_slice(&wit.wstart[..nws]);
        wit.wcols.clear();
        wit.wcols.resize(wit.wstart[nws] as usize, 0);
        for k in 0..nc {
            if !cm[k].mpos() {
                continue;
            }
            let y = colv[k] as usize;
            let c = wit.cpos[y];
            for e in g.lo(y)..g.hi(y) {
                let s = wit.wslot[g.at(e)] as usize;
                let p = wit.wfill[s] as usize;
                wit.wcols[p] = c;
                wit.wfill[s] += 1;
            }
        }
        wit.codes.clear();
        wit.cstart.clear();
        wit.cstart.push(0);
        wit.rmass.clear();
        for r in 0..nr {
            if !rm[r].mpos() {
                continue;
            }
            let x = rowv[r] as usize;
            for e in g.lo(x)..g.hi(x) {
                let v = g.at(e);
                if wit.cst[v] == ep {
                    wit.codes.push(2 * v as u64);
                }
                if wit.wstp[v] == ep {
                    wit.codes.push(2 * v as u64 + 1);
                }
            }
            let len = wit.codes.len();
            wit.cstart.push(len);
            wit.rmass.push(rm[r]);
        }
        let nrow = wit.rmass.len();
        let ng = dedupe_var(&wit.codes, &wit.cstart, nrow, &mut wit.table, &mut wit.gid, &mut wit.leader);
        wit.gm.clear();
        wit.gm.resize(ng, C::ZERO);
        for r in 0..nrow {
            let gg = wit.gid[r] as usize;
            let mv = wit.rmass[r];
            wit.gm[gg] += mv;
        }
        wit.rcnt.clear();
        wit.rcnt.resize(nws, 0);
        for gg in 0..ng {
            let ld = wit.leader[gg] as usize;
            for z in wit.cstart[ld]..wit.cstart[ld + 1] {
                let code = wit.codes[z];
                if code & 1 == 1 {
                    let s = wit.wslot[(code >> 1) as usize] as usize;
                    wit.rcnt[s] += 1;
                }
            }
        }
        wit.rnode.clear();
        wit.rnode.resize(nws, -1);
        let mut nh = 0usize;
        for s in 0..nws {
            if wit.rcnt[s] > 1 && wit.wcnt[s] > 1 {
                wit.rnode[s] = nh as i32;
                nh += 1;
            }
        }
        let cb = 2 + 2 * ng;
        let hb = cb + 2 * ncol;
        let nv = hb + nh;
        wit.net.clear();
        for gg in 0..ng {
            let (pn, qn) = (2 + 2 * gg, 3 + 2 * gg);
            let gmv = wit.gm[gg];
            wit.net.arc(0, pn, gmv);
            wit.net.arc(0, qn, gmv);
            wit.net.arc(qn, pn, C::INF);
            let ld = wit.leader[gg] as usize;
            for z in wit.cstart[ld]..wit.cstart[ld + 1] {
                let code = wit.codes[z];
                let v = (code >> 1) as usize;
                if code & 1 == 0 {
                    let c = wit.cpos[v] as usize;
                    wit.net.arc(pn, cb + 2 * c, C::INF);
                    wit.net.arc(qn, cb + 2 * c + 1, C::INF);
                } else {
                    let s = wit.wslot[v] as usize;
                    if wit.rnode[s] >= 0 {
                        let h = hb + wit.rnode[s] as usize;
                        wit.net.arc(qn, h, C::INF);
                    } else {
                        for t in wit.wstart[s] as usize..wit.wstart[s + 1] as usize {
                            let c = wit.wcols[t] as usize;
                            wit.net.arc(qn, cb + 2 * c, C::INF);
                        }
                    }
                }
            }
        }
        for s in 0..nws {
            if wit.rnode[s] < 0 {
                continue;
            }
            let h = hb + wit.rnode[s] as usize;
            for t in wit.wstart[s] as usize..wit.wstart[s + 1] as usize {
                let c = wit.wcols[t] as usize;
                wit.net.arc(h, cb + 2 * c, C::INF);
            }
        }
        for c in 0..ncol {
            let d = wit.cdem[c];
            wit.net.arc(cb + 2 * c, 1, d);
            wit.net.arc(cb + 2 * c + 1, 1, d);
        }
        wit.net.build(nv);
        let f = wit.net.dinic(nv, 0, 1);
        (a + a + a - f, PATH_WIT)
    }
}
