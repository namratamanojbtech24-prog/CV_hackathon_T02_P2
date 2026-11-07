"""Blind estimation of Poisson-Gaussian noise (var = a*x + b) directly from a Bayer frame.

Green photosites are predicted from their four diagonal green neighbours; the residual is
(almost) pure noise. A robust (MAD) variance estimate inside intensity bins gives the noise
curve, from which a line is fitted.
This lets the ISP adapt to unknown ISO / illumination without calibration metadata.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .raw_processing import bayer_masks


def estimate_noise_params(mosaic, pattern="RGGB", clip_level=1.0, bins=10):
    """Return (a, b) for var = a*x + b on the (already gain-amplified) mosaic.

    Residual r = g - mean(4 diagonal green neighbours) is computed at every green site (diagonal
    neighbours of a green photosite are green in any Bayer layout). It cancels smooth structure
    (only curvature leaks through), has noise variance sigma^2 * (1 + 1/4), and is analysed with a
    robust MAD estimator inside intensity bins.
    """
    m = np.asarray(mosaic, np.float32)
    h, w = m.shape
    _, g, _ = bayer_masks((h, w), pattern)
    diag = np.zeros_like(m)
    cnt = np.zeros_like(m)
    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        sh = np.roll(np.roll(m, dy, 0), dx, 1)
        diag += sh
        cnt += 1
    diag /= cnt
    inner = np.zeros((h, w), bool)
    inner[2:-2, 2:-2] = True
    sel0 = g & inner
    r = (m - diag)[sel0] / np.sqrt(1.0 + 0.25)
    mean_sig = ((m + diag) / 2)[sel0]
    valid = mean_sig < 0.92 * clip_level
    r, mean_sig = r[valid], mean_sig[valid]
    if r.size < 400:
        return 1e-3, 1e-5
    edges = np.quantile(mean_sig, np.linspace(0.0, 1.0, bins + 1))
    xs, vs, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mean_sig >= lo) & (mean_sig <= hi)
        if sel.sum() < 100:
            continue
        rr = r[sel]
        sigma = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        xs.append(mean_sig[sel].mean())
        vs.append(sigma ** 2)
        ws.append(np.sqrt(sel.sum()))
    if len(xs) < 3:
        v = float(np.mean(vs)) if vs else 1e-5
        return 1e-4, max(v, 1e-8)
    xs, vs, ws = map(np.asarray, (xs, vs, ws))
    A = np.stack([xs, np.ones_like(xs)], 1) * ws[:, None]
    (a, b), *_ = np.linalg.lstsq(A, vs * ws, rcond=None)
    a = float(max(a, 1e-6))
    if b < 0:   # refit through the origin; keep the intercept physical
        a = float(max((xs * vs * ws ** 2).sum() / max((xs ** 2 * ws ** 2).sum(), 1e-12), 1e-6))
        b = 1e-8
    return a, float(max(b, 1e-8))


def noise_sigma(x, a, b):
    """Per-pixel noise standard deviation for signal x."""
    return np.sqrt(np.maximum(a * np.maximum(x, 0) + b, 1e-12))


def noise_level_at(a, b, level=0.18):
    """A single summary number: noise std at mid-grey (linear 0.18)."""
    return float(np.sqrt(max(a * level + b, 0)))
