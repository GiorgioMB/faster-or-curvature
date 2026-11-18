//! Maximum flow (Dinic, as `orcut._kernels._dinic`) on a CSR residual network, generic over the
//! capacity type: `i64` for exact integer masses, `f64` for real masses.
use std::ops::{Add, AddAssign, Sub, SubAssign};

pub trait Cap:
    Copy + Default + PartialOrd + Add<Output = Self> + Sub<Output = Self> + AddAssign + SubAssign + Send + Sync + 'static
{
    const ZERO: Self;
    const INF: Self;
    /// A residual capacity that can still carry flow (f64: above a small tolerance).
    fn pos(self) -> bool;
}

impl Cap for i64 {
    const ZERO: i64 = 0;
    const INF: i64 = 1 << 60;
    #[inline(always)]
    fn pos(self) -> bool {
        self > 0
    }
}

/// Residual capacities at or below this are treated as saturated (masses are probabilities).
pub const F64_TOL: f64 = 1e-13;

impl Cap for f64 {
    const ZERO: f64 = 0.0;
    const INF: f64 = f64::INFINITY;
    #[inline(always)]
    fn pos(self) -> bool {
        self > F64_TOL
    }
}

#[derive(Default)]
pub struct Net<C: Cap> {
    tail: Vec<u32>,
    head: Vec<u32>,
    capv: Vec<C>,
    start: Vec<u32>,
    to: Vec<u32>,
    cap: Vec<C>,
    rev: Vec<u32>,
    pos: Vec<u32>,
    fill: Vec<u32>,
    level: Vec<i32>,
    it: Vec<u32>,
    q: Vec<u32>,
    path: Vec<u32>,
}

impl<C: Cap> Net<C> {
    pub fn clear(&mut self) {
        self.tail.clear();
        self.head.clear();
        self.capv.clear();
    }

    #[inline(always)]
    pub fn arc(&mut self, u: usize, v: usize, c: C) -> u32 {
        let k = self.tail.len() as u32;
        self.tail.push(u as u32);
        self.head.push(v as u32);
        self.capv.push(c);
        k
    }

    #[inline(always)]
    pub fn n_arcs(&self) -> usize {
        self.tail.len()
    }

    /// CSR residual network of the arcs added so far (paired reverse arcs).
    pub fn build(&mut self, nv: usize) {
        let na = self.tail.len();
        self.start.clear();
        self.start.resize(nv + 1, 0);
        for k in 0..na {
            self.start[self.tail[k] as usize + 1] += 1;
            self.start[self.head[k] as usize + 1] += 1;
        }
        for v in 0..nv {
            self.start[v + 1] += self.start[v];
        }
        self.fill.clear();
        self.fill.extend_from_slice(&self.start[..nv]);
        self.to.clear();
        self.to.resize(2 * na, 0);
        self.cap.clear();
        self.cap.resize(2 * na, C::ZERO);
        self.rev.clear();
        self.rev.resize(2 * na, 0);
        self.pos.clear();
        self.pos.resize(na, 0);
        for k in 0..na {
            let u = self.tail[k] as usize;
            let v = self.head[k] as usize;
            let a = self.fill[u] as usize;
            self.fill[u] += 1;
            let b = self.fill[v] as usize;
            self.fill[v] += 1;
            self.to[a] = v as u32;
            self.cap[a] = self.capv[k];
            self.rev[a] = b as u32;
            self.to[b] = u as u32;
            self.cap[b] = C::ZERO;
            self.rev[b] = a as u32;
            self.pos[k] = a as u32;
        }
    }

    /// Flow on arc k (the residual capacity of its reverse).
    #[inline(always)]
    pub fn flow_of(&self, k: usize) -> C {
        self.cap[self.rev[self.pos[k] as usize] as usize]
    }

    /// Push f units along arc k (loads a warm start).
    #[inline(always)]
    pub fn push_arc(&mut self, k: usize, f: C) {
        let e = self.pos[k] as usize;
        let r = self.rev[e] as usize;
        self.cap[e] -= f;
        self.cap[r] += f;
    }

    /// Dinic from the current residual capacities (any feasible flow); returns the added flow.
    pub fn dinic(&mut self, nv: usize, s: usize, t: usize) -> C {
        self.level.clear();
        self.level.resize(nv, -1);
        self.it.clear();
        self.it.resize(nv, 0);
        self.q.clear();
        self.q.resize(nv, 0);
        self.path.clear();
        self.path.resize(nv + 1, 0);
        let mut flow = C::ZERO;
        loop {
            for v in 0..nv {
                self.level[v] = -1;
            }
            self.level[s] = 0;
            self.q[0] = s as u32;
            let (mut qh, mut qt) = (0usize, 1usize);
            // SAFETY: build() makes start (nv + 1 entries, ending at 2 na) and to/cap/rev (2 na
            // entries, heads < nv) consistent, and q/level/it/path have nv (+ 1) entries.
            while qh < qt {
                let u = unsafe { *self.q.get_unchecked(qh) } as usize;
                qh += 1;
                let lu = unsafe { *self.level.get_unchecked(u) } + 1;
                let (lo, hi) = unsafe { (*self.start.get_unchecked(u) as usize, *self.start.get_unchecked(u + 1) as usize) };
                for a in lo..hi {
                    unsafe {
                        if self.cap.get_unchecked(a).pos() {
                            let v = *self.to.get_unchecked(a) as usize;
                            if *self.level.get_unchecked(v) < 0 {
                                *self.level.get_unchecked_mut(v) = lu;
                                *self.q.get_unchecked_mut(qt) = v as u32;
                                qt += 1;
                            }
                        }
                    }
                }
                if self.level[t] >= 0 {
                    break;
                }
            }
            if self.level[t] < 0 {
                break;
            }
            let lt = self.level[t];
            for v in 0..nv {
                self.it[v] = self.start[v];
                if self.level[v] >= lt && v != t {
                    self.level[v] = -1;
                }
            }
            let mut top = 0usize;
            let mut u = s;
            loop {
                if u == t {
                    let mut f = C::INF;
                    for p in 0..top {
                        let c = self.cap[self.path[p] as usize];
                        if c < f {
                            f = c;
                        }
                    }
                    for p in 0..top {
                        let a = self.path[p] as usize;
                        let r = self.rev[a] as usize;
                        self.cap[a] -= f;
                        self.cap[r] += f;
                    }
                    flow += f;
                    let mut p0 = 0usize;
                    while self.cap[self.path[p0] as usize].pos() {
                        p0 += 1;
                    }
                    top = p0;
                    u = if top == 0 { s } else { self.to[self.path[top - 1] as usize] as usize };
                    continue;
                }
                let mut a = self.it[u] as usize;
                let end = self.start[u + 1] as usize;
                let lu = self.level[u] + 1;
                while a < end {
                    // SAFETY: a < end <= 2 na, and heads are < nv (see build()).
                    unsafe {
                        if self.cap.get_unchecked(a).pos() && *self.level.get_unchecked(*self.to.get_unchecked(a) as usize) == lu {
                            break;
                        }
                    }
                    a += 1;
                }
                self.it[u] = a as u32;
                if a < end {
                    self.path[top] = a as u32;
                    top += 1;
                    u = self.to[a] as usize;
                } else {
                    self.level[u] = -1;
                    if top == 0 {
                        break;
                    }
                    top -= 1;
                    u = self.to[self.rev[self.path[top] as usize] as usize] as usize;
                    self.it[u] += 1;
                }
            }
        }
        flow
    }
}
