"""The native backend: ``orcut._native``, compiled from the Rust crate in ``src/`` (PyO3).

It computes the same exact values as the numba kernel of :mod:`orcut._core` (bit for bit, tested)
with no JIT compilation, on all cores without the GIL, and adds float masses (real walk weights)
and edge lengths (:mod:`orcut.weighted`). The numba kernels stay for certificates, ball and
multi-step measures, diagnostics and ablation flags.
"""
from __future__ import annotations

import os

import numpy as np

try:
    from . import _native
except ImportError:  # a source tree without a build
    _native = None

__all__ = ["available", "threads_for", "hop", "lengths"]

_I = np.zeros(1, np.int64)
_F = np.zeros(1, np.float64)


def available() -> bool:
    """Whether the native backend is installed."""
    return _native is not None


def threads_for(n_jobs) -> int:
    """Threads for the native backend (scikit-learn semantics, as :func:`orcut.api.threads`)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpus
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    k = n_jobs if n_jobs > 0 else cpus + 1 + n_jobs
    return max(1, min(int(k), cpus))


def _tabs(spec):
    t = spec.tabs if spec.tabs is not None else (_I,) * 5
    f = spec.ftabs if spec.ftabs is not None else (_F,) * 3
    return (tuple(np.ascontiguousarray(a, dtype=np.int64) for a in t),
            tuple(np.ascontiguousarray(a, dtype=np.float64) for a in f))


def hop(g, ids, spec, nthreads):
    """(curvature, W1 numerator, denominator, path) on the hop metric."""
    t, f = _tabs(spec)
    return _native.hop_ids(g.indptr, g.indices, g.edges, np.ascontiguousarray(ids, dtype=np.int64),
                           int(spec.mode), int(spec.pn), int(spec.pd), *t, *f, int(nthreads))


def lengths(g, ids, spec, lens, integral, nthreads):
    """(curvature, W1 numerator, denominator, integer distance, W1, distance, network-simplex pivots,
    rows, columns) per edge, for lengths aligned with ``g.indices`` (int64 if ``integral``, else
    float64)."""
    t, f = _tabs(spec)
    li = np.ascontiguousarray(lens, dtype=np.int64) if integral else _I
    lf = _F if integral else np.ascontiguousarray(lens, dtype=np.float64)
    return _native.len_ids(g.indptr, g.indices, g.edges, np.ascontiguousarray(ids, dtype=np.int64),
                           int(spec.mode), int(spec.pn), int(spec.pd), *t, *f, li, lf, bool(integral), int(nthreads))
