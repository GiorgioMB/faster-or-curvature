//! The two closed neighbourhoods of an edge and the net masses of a one-step measure on them, with
//! the orientation, the common denominators and the overflow rules of `orcut._core._edge_core`.
use crate::flow::Cap;
use crate::Graph;

pub const LMAX: i64 = 1 << 50;
pub const MASS_TOL: f64 = 1e-15;

pub enum Meas<'a> {
    Lazy,
    NonLazy,
    Idle { a: i64, b: i64 },
    Lly,
    WalkI { wts: &'a [i64], dp: &'a [i64], nself: &'a [i64], coef: &'a [i64], gdv: &'a [i64] },
    WalkF { wts: &'a [f64], coef: &'a [f64], idle: &'a [f64] },
}

#[inline]
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Rows: U_i, then i, j and the triangle vertices T; columns: U_j, then i, j, T. ``p*`` are CSR
/// positions (for the weights of a walk or the lengths of the edges).
#[derive(Default)]
pub struct Nbhd {
    pub i: usize,
    pub j: usize,
    pub di: i64,
    pub dj: i64,
    pub ui: Vec<u32>,
    pub pui: Vec<usize>,
    pub uj: Vec<u32>,
    pub puj: Vec<usize>,
    pub tt: Vec<u32>,
    pub pti: Vec<usize>,
    pub ptj: Vec<usize>,
    pub pij: usize,
    pub pji: usize,
}

impl Nbhd {
    pub fn build(&mut self, g: &Graph, i0: usize, j0: usize) {
        let (i, j) = if g.deg(i0) >= g.deg(j0) { (i0, j0) } else { (j0, i0) };
        self.i = i;
        self.j = j;
        self.di = g.deg(i);
        self.dj = g.deg(j);
        self.ui.clear();
        self.pui.clear();
        self.uj.clear();
        self.puj.clear();
        self.tt.clear();
        self.pti.clear();
        self.ptj.clear();
        let (mut p, pe, mut q, qe) = (g.lo(i), g.hi(i), g.lo(j), g.hi(j));
        while p < pe || q < qe {
            if q >= qe || (p < pe && g.indices[p] < g.indices[q]) {
                let x = g.at(p);
                if x != j {
                    self.ui.push(x as u32);
                    self.pui.push(p);
                } else {
                    self.pij = p;
                }
                p += 1;
            } else if p >= pe || g.indices[q] < g.indices[p] {
                let y = g.at(q);
                if y != i {
                    self.uj.push(y as u32);
                    self.puj.push(q);
                } else {
                    self.pji = q;
                }
                q += 1;
            } else {
                self.tt.push(g.at(p) as u32);
                self.pti.push(p);
                self.ptj.push(q);
                p += 1;
                q += 1;
            }
        }
    }

    pub fn rows_into(&self, v: &mut Vec<u32>) {
        v.clear();
        v.extend_from_slice(&self.ui);
        v.push(self.i as u32);
        v.push(self.j as u32);
        v.extend_from_slice(&self.tt);
    }

    pub fn cols_into(&self, v: &mut Vec<u32>) {
        v.clear();
        v.extend_from_slice(&self.uj);
        v.push(self.i as u32);
        v.push(self.j as u32);
        v.extend_from_slice(&self.tt);
    }
}

pub trait Mass: Cap {
    const EXACT: bool;
    /// Net masses m_i - m_j on the rows (rm) and demands m_j - m_i on the columns (cm); returns the
    /// common denominator, or None if it does not fit (exact masses only).
    fn masses(m: &Meas, nb: &Nbhd, rm: &mut Vec<Self>, cm: &mut Vec<Self>) -> Option<i64>;
    fn mpos(self) -> bool;
    /// (curvature, W1 numerator, denominator) from the scaled W1.
    fn finish(w: Self, l: i64, m: &Meas, nb: &Nbhd) -> (f64, i64, i64);
}

impl Mass for i64 {
    const EXACT: bool = true;

    fn masses(m: &Meas, nb: &Nbhd, rm: &mut Vec<i64>, cm: &mut Vec<i64>) -> Option<i64> {
        let (nui, nuj, nt) = (nb.ui.len(), nb.uj.len(), nb.tt.len());
        let ncc = 2 + nt;
        rm.clear();
        rm.resize(nui + ncc, 0);
        cm.clear();
        cm.resize(nuj + ncc, 0);
        let (di, dj) = (nb.di, nb.dj);
        let dmax = di.max(dj);
        let l: i64;
        match *m {
            Meas::WalkI { wts, dp, nself, coef, gdv } => {
                let (i, j) = (nb.i, nb.j);
                let (dpi, dpj) = (dp[i], dp[j]);
                let g = gcd(dpi, dpj);
                if dpi / g > LMAX / dpj {
                    return None;
                }
                l = (dpi / g) * dpj;
                let (si, sj) = (l / dpi, l / dpj);
                let (ci, cj, gi, gj) = (coef[i], coef[j], gdv[i], gdv[j]);
                for r in 0..nui {
                    rm[r] = (ci * wts[nb.pui[r]] / gi) * si;
                }
                for t in 0..nuj {
                    cm[t] = (cj * wts[nb.puj[t]] / gj) * sj;
                }
                rm[nui] = nself[i] * si - (cj * wts[nb.pji] / gj) * sj;
                rm[nui + 1] = (ci * wts[nb.pij] / gi) * si - nself[j] * sj;
                for t in 0..nt {
                    rm[nui + 2 + t] = (ci * wts[nb.pti[t]] / gi) * si - (cj * wts[nb.ptj[t]] / gj) * sj;
                }
            }
            Meas::WalkF { .. } => unreachable!("a float walk needs float masses"),
            _ => {
                let (mi_s, mi_n, mj_s, mj_n);
                match *m {
                    Meas::Lazy => {
                        l = (di + 1) * (dj + 1);
                        mi_s = dj + 1;
                        mi_n = dj + 1;
                        mj_s = di + 1;
                        mj_n = di + 1;
                    }
                    Meas::NonLazy => {
                        l = di * dj;
                        mi_s = 0;
                        mi_n = dj;
                        mj_s = 0;
                        mj_n = di;
                    }
                    _ => {
                        let (a, b) = match *m {
                            Meas::Idle { a, b } => (a, b),
                            _ => (1, dmax + 1),
                        };
                        if b > LMAX / (di * dj).max(1) {
                            return None;
                        }
                        l = b * di * dj;
                        mi_s = a * di * dj;
                        mi_n = (b - a) * dj;
                        mj_s = a * di * dj;
                        mj_n = (b - a) * di;
                    }
                }
                for r in 0..nui {
                    rm[r] = mi_n;
                }
                for t in 0..nuj {
                    cm[t] = mj_n;
                }
                rm[nui] = mi_s - mj_n;
                rm[nui + 1] = mi_n - mj_s;
                for t in 0..nt {
                    rm[nui + 2 + t] = mi_n - mj_n;
                }
            }
        }
        for t in 0..ncc {
            cm[nuj + t] = -rm[nui + t];
        }
        Some(l)
    }

    #[inline(always)]
    fn mpos(self) -> bool {
        self > 0
    }

    fn finish(w: i64, l: i64, m: &Meas, nb: &Nbhd) -> (f64, i64, i64) {
        if let Meas::Lly = m {
            let dmax = nb.di.max(nb.dj);
            let kp = 1.0 - (w as f64) / (l as f64);
            (kp * ((dmax + 1) as f64) / (dmax as f64), w, l)
        } else {
            (1.0 - (w as f64) / (l as f64), w, l)
        }
    }
}

impl Mass for f64 {
    const EXACT: bool = false;

    fn masses(m: &Meas, nb: &Nbhd, rm: &mut Vec<f64>, cm: &mut Vec<f64>) -> Option<i64> {
        let Meas::WalkF { wts, coef, idle } = *m else { unreachable!("an integer measure needs integer masses") };
        let (nui, nuj, nt) = (nb.ui.len(), nb.uj.len(), nb.tt.len());
        let ncc = 2 + nt;
        rm.clear();
        rm.resize(nui + ncc, 0.0);
        cm.clear();
        cm.resize(nuj + ncc, 0.0);
        let (i, j) = (nb.i, nb.j);
        let (ci, cj) = (coef[i], coef[j]);
        for r in 0..nui {
            rm[r] = ci * wts[nb.pui[r]];
        }
        for t in 0..nuj {
            cm[t] = cj * wts[nb.puj[t]];
        }
        rm[nui] = idle[i] - cj * wts[nb.pji];
        rm[nui + 1] = ci * wts[nb.pij] - idle[j];
        for t in 0..nt {
            rm[nui + 2 + t] = ci * wts[nb.pti[t]] - cj * wts[nb.ptj[t]];
        }
        for t in 0..ncc {
            cm[nuj + t] = -rm[nui + t];
        }
        Some(1)
    }

    #[inline(always)]
    fn mpos(self) -> bool {
        self > MASS_TOL
    }

    fn finish(w: f64, _l: i64, _m: &Meas, _nb: &Nbhd) -> (f64, i64, i64) {
        (1.0 - w, -1, 0)
    }
}
