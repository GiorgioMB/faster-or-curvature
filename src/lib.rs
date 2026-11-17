//! Native kernels of faster-or-curvature (``orcut._native``): the one-step closure network on
//! the hop metric (bit for bit the values of ``orcut._core``) and the primal-dual solver for edge
//! lengths, run over the edges in parallel with rayon, without the GIL.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Mutex;

mod flow;
mod hop;
mod lengths;
mod nbhd;

use nbhd::Meas;

pub struct Graph<'a> {
    pub indptr: &'a [i64],
    pub indices: &'a [i64],
}

impl<'a> Graph<'a> {
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.indptr.len() - 1
    }
    #[inline(always)]
    pub fn deg(&self, v: usize) -> i64 {
        self.indptr[v + 1] - self.indptr[v]
    }
    #[inline(always)]
    pub fn lo(&self, v: usize) -> usize {
        self.indptr[v] as usize
    }
    #[inline(always)]
    pub fn hi(&self, v: usize) -> usize {
        self.indptr[v + 1] as usize
    }
    #[inline(always)]
    pub fn at(&self, e: usize) -> usize {
        self.indices[e] as usize
    }
}

#[derive(Clone, Copy, Default)]
pub struct EdgeOut {
    pub val: f64,
    pub num: i64,
    pub den: i64,
    pub path: i8,
}

const BLOCK: usize = 16;

/// f(workspace, x) for x in 0..k, on nthreads threads with one workspace per thread.
pub fn run_edges<W, T, M, F>(k: usize, nthreads: usize, make: M, f: F) -> Vec<T>
where
    W: Send,
    T: Send + Clone + Default,
    M: Fn() -> W + Sync,
    F: Fn(&mut W, usize) -> T + Sync,
{
    let mut out = vec![T::default(); k];
    if k == 0 {
        return out;
    }
    if nthreads <= 1 {
        let mut ws = make();
        for (x, o) in out.iter_mut().enumerate() {
            *o = f(&mut ws, x);
        }
        return out;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().expect("rayon thread pool");
    let wss: Vec<Mutex<Option<W>>> = (0..nthreads).map(|_| Mutex::new(None)).collect();
    pool.install(|| {
        use rayon::prelude::*;
        out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, chunk)| {
            let t = rayon::current_thread_index().unwrap_or(0) % nthreads;
            let mut guard = wss[t].lock().unwrap();
            let ws = guard.get_or_insert_with(&make);
            for (o, slot) in chunk.iter_mut().enumerate() {
                *slot = f(ws, b * BLOCK + o);
            }
        });
    });
    out
}

fn make_meas<'a>(
    mode: i64, pn: i64, pd: i64, wi: &'a [i64], dp: &'a [i64], ns: &'a [i64], cf: &'a [i64], gd: &'a [i64],
    wf: &'a [f64], cff: &'a [f64], idf: &'a [f64], n: usize, nnz: usize,
) -> PyResult<Meas<'a>> {
    Ok(match mode {
        0 => Meas::Lazy,
        1 => Meas::NonLazy,
        2 => {
            if !(0 <= pn && pn < pd) {
                return Err(PyValueError::new_err("idleness pn/pd must lie in [0, 1)"));
            }
            Meas::Idle { a: pn, b: pd }
        }
        3 => Meas::Lly,
        4 => {
            if wi.len() != nnz || dp.len() != n || ns.len() != n || cf.len() != n || gd.len() != n {
                return Err(PyValueError::new_err("walk tables do not match the graph"));
            }
            Meas::WalkI { wts: wi, dp, nself: ns, coef: cf, gdv: gd }
        }
        5 => {
            if wf.len() != nnz || cff.len() != n || idf.len() != n {
                return Err(PyValueError::new_err("float walk tables do not match the graph"));
            }
            Meas::WalkF { wts: wf, coef: cff, idle: idf }
        }
        _ => return Err(PyValueError::new_err("unknown measure mode")),
    })
}

fn check(ip: &[i64], ix: &[i64], ev: &[i64], idv: &[i64]) -> PyResult<()> {
    if ip.is_empty() || ip[ip.len() - 1] as usize != ix.len() {
        return Err(PyValueError::new_err("indptr and indices do not match"));
    }
    let n = ip.len() - 1;
    let m = ev.len() / 2;
    for &e in idv {
        if e < 0 || e as usize >= m {
            return Err(PyValueError::new_err("edge id out of range"));
        }
        let (a, b) = (ev[2 * e as usize], ev[2 * e as usize + 1]);
        if a < 0 || b < 0 || a as usize >= n || b as usize >= n || a == b {
            return Err(PyValueError::new_err("invalid edge"));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, edges, ids, mode, pn, pd, wts_i, dp, nself, coef, gdv, wts_f, coef_f, idle_f, nthreads))]
fn hop_ids<'py>(
    py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>, ids: PyReadonlyArray1<'py, i64>, mode: i64, pn: i64, pd: i64,
    wts_i: PyReadonlyArray1<'py, i64>, dp: PyReadonlyArray1<'py, i64>, nself: PyReadonlyArray1<'py, i64>,
    coef: PyReadonlyArray1<'py, i64>, gdv: PyReadonlyArray1<'py, i64>, wts_f: PyReadonlyArray1<'py, f64>,
    coef_f: PyReadonlyArray1<'py, f64>, idle_f: PyReadonlyArray1<'py, f64>, nthreads: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i8>>)> {
    let (ip, ix, ev, idv) = (indptr.as_slice()?, indices.as_slice()?, edges.as_slice()?, ids.as_slice()?);
    check(ip, ix, ev, idv)?;
    let g = Graph { indptr: ip, indices: ix };
    let n = g.n();
    let meas = make_meas(
        mode, pn, pd, wts_i.as_slice()?, dp.as_slice()?, nself.as_slice()?, coef.as_slice()?, gdv.as_slice()?,
        wts_f.as_slice()?, coef_f.as_slice()?, idle_f.as_slice()?, n, ix.len(),
    )?;
    let k = idv.len();
    let out: Vec<EdgeOut> = py.detach(|| {
        let ends = |x: usize| {
            let e = idv[x] as usize;
            (ev[2 * e] as usize, ev[2 * e + 1] as usize)
        };
        if mode == 5 {
            run_edges(k, nthreads, || hop::HopWs::<f64>::new(n), |ws, x| {
                let (a, b) = ends(x);
                ws.edge(&g, &meas, a, b)
            })
        } else {
            run_edges(k, nthreads, || hop::HopWs::<i64>::new(n), |ws, x| {
                let (a, b) = ends(x);
                ws.edge(&g, &meas, a, b)
            })
        }
    });
    let val: Vec<f64> = out.iter().map(|o| o.val).collect();
    let num: Vec<i64> = out.iter().map(|o| o.num).collect();
    let den: Vec<i64> = out.iter().map(|o| o.den).collect();
    let path: Vec<i8> = out.iter().map(|o| o.path).collect();
    Ok((val.into_pyarray(py), num.into_pyarray(py), den.into_pyarray(py), path.into_pyarray(py)))
}

type LenResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
);

const MAX_BLOCK: usize = 4096;

/// Runs of consecutive positions of ``idv`` whose edges share their first endpoint.
fn blocks_of(ev: &[i64], idv: &[i64]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = 0usize;
    while a < idv.len() {
        let u = ev[2 * idv[a] as usize];
        let mut b = a + 1;
        while b < idv.len() && ev[2 * idv[b] as usize] == u && b - a < MAX_BLOCK {
            b += 1;
        }
        out.push((a, b));
        a = b;
    }
    out
}

/// f(workspace, start, end, out[start..end]) for every block, on nthreads threads.
pub fn run_blocks<W, T, M, F>(blocks: &[(usize, usize)], k: usize, nthreads: usize, make: M, f: F) -> Vec<T>
where
    W: Send,
    T: Send + Clone + Default,
    M: Fn() -> W + Sync,
    F: Fn(&mut W, usize, usize, &mut [T]) + Sync,
{
    let mut out = vec![T::default(); k];
    {
        let mut parts: Vec<(usize, usize, &mut [T])> = Vec::with_capacity(blocks.len());
        let mut rest: &mut [T] = &mut out[..];
        for &(a, b) in blocks {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(b - a);
            parts.push((a, b, head));
            rest = tail;
        }
        if nthreads <= 1 {
            let mut ws = make();
            for (a, b, s) in parts {
                f(&mut ws, a, b, s);
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().expect("rayon thread pool");
            let wss: Vec<Mutex<Option<W>>> = (0..nthreads).map(|_| Mutex::new(None)).collect();
            pool.install(|| {
                use rayon::prelude::*;
                parts.into_par_iter().for_each(|(a, b, s)| {
                    let t = rayon::current_thread_index().unwrap_or(0) % nthreads;
                    let mut guard = wss[t].lock().unwrap();
                    let ws = guard.get_or_insert_with(&make);
                    f(ws, a, b, s);
                });
            });
        }
    }
    out
}

/// Distance entries held at once by the shared single-source distances (128 MB of 8-byte values).
const GLOBAL_BUDGET: usize = 1 << 24;

fn pool_of(nthreads: usize) -> Option<rayon::ThreadPool> {
    (nthreads > 1).then(|| rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().expect("rayon thread pool"))
}

/// The edge-length solver over the edges ``idv``. Local mode: per run of edges sharing their first
/// endpoint, targeted Dijkstras (cheap when shortest paths stay local). Global mode, chosen when a
/// sample of local Dijkstras settles more than a fifth of the graph (small worlds): every needed
/// single-source distance is computed once per chunk of runs and shared by all their edges.
fn len_run<F: lengths::Prod<D>, D: lengths::Dist>(
    g: &Graph, lens: &[D], meas: &Meas, ev: &[i64], idv: &[i64], nthreads: usize,
) -> Vec<lengths::LenOut> {
    use rayon::prelude::*;
    let n = g.n();
    let k = idv.len();
    let blocks = blocks_of(ev, idv);
    let first = |a: usize| ev[2 * idv[a] as usize] as usize;
    let dmax = blocks.iter().map(|&(a, _)| g.deg(first(a)) as usize).max().unwrap_or(0);
    let mut global = false;
    if k > 64 && n.saturating_mul(dmax + 1) <= GLOBAL_BUDGET {
        let mut ws = lengths::LenWs::<F, D>::new(n);
        let mut tmp = vec![lengths::LenOut::default(); MAX_BLOCK];
        let step = (blocks.len() / 24).max(1);
        for b in (0..blocks.len()).step_by(step).take(24) {
            let (a, e) = blocks[b];
            ws.block(g, lens, meas, ev, &idv[a..e], &mut tmp[..e - a]);
        }
        global = ws.n_dij > 0 && (ws.n_settled as f64) > 0.2 * (n as f64) * (ws.n_dij as f64);
    }
    if !global {
        return run_blocks(&blocks, k, nthreads, || lengths::LenWs::<F, D>::new(n), |ws, a, b, s| {
            ws.block(g, lens, meas, ev, &idv[a..b], s)
        });
    }
    let pool = pool_of(nthreads);
    let max_rows = (GLOBAL_BUDGET / n.max(1)).max(dmax + 1);
    let mut out = vec![lengths::LenOut::default(); k];
    let mut grow = vec![u32::MAX; n];
    let mut srcs: Vec<u32> = Vec::new();
    let mut mat: Vec<D> = Vec::new();
    let dws: Vec<Mutex<Option<lengths::DijWs<D>>>> = (0..nthreads.max(1)).map(|_| Mutex::new(None)).collect();
    let lws: Vec<Mutex<Option<lengths::LenWs<F, D>>>> = (0..nthreads.max(1)).map(|_| Mutex::new(None)).collect();
    let mut bi = 0usize;
    while bi < blocks.len() {
        srcs.clear();
        let b0 = bi;
        while bi < blocks.len() {
            let u = first(blocks[bi].0);
            let mut new = (grow[u] == u32::MAX) as usize;
            for e in g.lo(u)..g.hi(u) {
                new += (grow[g.at(e)] == u32::MAX) as usize;
            }
            if !srcs.is_empty() && srcs.len() + new > max_rows {
                break;
            }
            for v in std::iter::once(u).chain((g.lo(u)..g.hi(u)).map(|e| g.at(e))) {
                if grow[v] == u32::MAX {
                    grow[v] = srcs.len() as u32;
                    srcs.push(v as u32);
                }
            }
            bi += 1;
        }
        let ns = srcs.len();
        mat.clear();
        mat.resize(ns * n, D::MAXV);
        let row_job = |si: usize, row: &mut [D]| {
            let t = rayon::current_thread_index().unwrap_or(0) % dws.len();
            let mut guard = dws[t].lock().unwrap();
            let ws = guard.get_or_insert_with(|| lengths::DijWs::new(n));
            lengths::full_dijkstra(g, lens, srcs[si] as usize, row, ws);
        };
        match &pool {
            Some(p) => p.install(|| mat.par_chunks_mut(n).enumerate().for_each(|(si, row)| row_job(si, row))),
            None => mat.chunks_mut(n).enumerate().for_each(|(si, row)| row_job(si, row)),
        }
        let (pa, pb) = (blocks[b0].0, blocks[bi - 1].1);
        let (mat_r, grow_r) = (&mat[..], &grow[..]);
        let edge_job = |c: usize, chunk: &mut [lengths::LenOut]| {
            let t = rayon::current_thread_index().unwrap_or(0) % lws.len();
            let mut guard = lws[t].lock().unwrap();
            let ws = guard.get_or_insert_with(|| lengths::LenWs::<F, D>::new(n));
            for (o, slot) in chunk.iter_mut().enumerate() {
                let e = idv[pa + c * BLOCK + o] as usize;
                *slot = ws.edge_global(g, meas, ev[2 * e] as usize, ev[2 * e + 1] as usize, mat_r, grow_r);
            }
        };
        match &pool {
            Some(p) => p.install(|| out[pa..pb].par_chunks_mut(BLOCK).enumerate().for_each(|(c, ch)| edge_job(c, ch))),
            None => out[pa..pb].chunks_mut(BLOCK).enumerate().for_each(|(c, ch)| edge_job(c, ch)),
        }
        for &v in &srcs {
            grow[v as usize] = u32::MAX;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, edges, ids, mode, pn, pd, wts_i, dp, nself, coef, gdv, wts_f, coef_f, idle_f, lens_i, lens_f, int_lengths, nthreads))]
fn len_ids<'py>(
    py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>, ids: PyReadonlyArray1<'py, i64>, mode: i64, pn: i64, pd: i64,
    wts_i: PyReadonlyArray1<'py, i64>, dp: PyReadonlyArray1<'py, i64>, nself: PyReadonlyArray1<'py, i64>,
    coef: PyReadonlyArray1<'py, i64>, gdv: PyReadonlyArray1<'py, i64>, wts_f: PyReadonlyArray1<'py, f64>,
    coef_f: PyReadonlyArray1<'py, f64>, idle_f: PyReadonlyArray1<'py, f64>, lens_i: PyReadonlyArray1<'py, i64>,
    lens_f: PyReadonlyArray1<'py, f64>, int_lengths: bool, nthreads: usize,
) -> PyResult<LenResult<'py>> {
    let (ip, ix, ev, idv) = (indptr.as_slice()?, indices.as_slice()?, edges.as_slice()?, ids.as_slice()?);
    check(ip, ix, ev, idv)?;
    let g = Graph { indptr: ip, indices: ix };
    let n = g.n();
    let meas = make_meas(
        mode, pn, pd, wts_i.as_slice()?, dp.as_slice()?, nself.as_slice()?, coef.as_slice()?, gdv.as_slice()?,
        wts_f.as_slice()?, coef_f.as_slice()?, idle_f.as_slice()?, n, ix.len(),
    )?;
    let (li, lf) = (lens_i.as_slice()?, lens_f.as_slice()?);
    let lens_len = if int_lengths { li.len() } else { lf.len() };
    if lens_len != ix.len() {
        return Err(PyValueError::new_err("one length per CSR entry is needed"));
    }
    let out: Vec<lengths::LenOut> = py.detach(|| match (mode == 5, int_lengths) {
        (false, true) => len_run::<i64, i64>(&g, li, &meas, ev, idv, nthreads),
        (false, false) => len_run::<i64, f64>(&g, lf, &meas, ev, idv, nthreads),
        (true, true) => len_run::<f64, i64>(&g, li, &meas, ev, idv, nthreads),
        (true, false) => len_run::<f64, f64>(&g, lf, &meas, ev, idv, nthreads),
    });
    let val: Vec<f64> = out.iter().map(|o| o.val).collect();
    let num: Vec<i64> = out.iter().map(|o| o.num).collect();
    let den: Vec<i64> = out.iter().map(|o| o.den).collect();
    let di: Vec<i64> = out.iter().map(|o| o.dist_i).collect();
    let w1: Vec<f64> = out.iter().map(|o| o.w1).collect();
    let dist: Vec<f64> = out.iter().map(|o| o.dist).collect();
    let aug: Vec<i32> = out.iter().map(|o| o.aug).collect();
    let rows: Vec<i32> = out.iter().map(|o| o.rows).collect();
    let cols: Vec<i32> = out.iter().map(|o| o.cols).collect();
    Ok((
        val.into_pyarray(py),
        num.into_pyarray(py),
        den.into_pyarray(py),
        di.into_pyarray(py),
        w1.into_pyarray(py),
        dist.into_pyarray(py),
        aug.into_pyarray(py),
        rows.into_pyarray(py),
        cols.into_pyarray(py),
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hop_ids, m)?)?;
    m.add_function(wrap_pyfunction!(len_ids, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
