"""Noise-adaptive denoising.

Main method ("nlm-bayer"): variance-stabilise the Bayer planes with the generalised
Anscombe transform (so noise becomes unit-variance, signal independent), run a
guided non-local-means filter whose weights come from the 4-plane average
(less noisy, shared by all planes -> no colour fringes), then invert the transform.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
from scipy.ndimage import uniform_filter

from .raw_processing import pack_bayer, unpack_bayer


# ---- generalised Anscombe transform --------------------------------------
def gat(x, a, b):
    return (2.0 / a) * np.sqrt(np.maximum(a * x + 0.375 * a * a + b, 0))


def gat_inverse(z, a, b):
    return ((a * z / 2.0) ** 2 - 0.375 * a * a - b) / a


def _box(x, k=3):
    if cv2 is not None:
        return cv2.boxFilter(x, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    return uniform_filter(x, k, mode="reflect")


def nlm_planes(z, guide_sigma2, strength=1.0, radius=2):
    """Guided NLM on stacked planes z (C,H,W) using their mean as guide."""
    g = z.mean(0).astype(np.float32)
    h, w = g.shape
    pad = radius
    gp = np.pad(g, pad, mode="reflect")
    zp = np.pad(z, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    lam = max(1.2 * strength, 1e-3) * guide_sigma2
    acc = np.zeros_like(z)
    wsum = np.zeros((h, w), np.float32)
    wmax = np.zeros((h, w), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            gq = gp[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            d = _box((g - gq) ** 2, 3)
            wt = np.exp(-np.maximum(d - 2 * guide_sigma2, 0) / lam).astype(np.float32)
            acc += wt[None] * zp[:, pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            wsum += wt
            wmax = np.maximum(wmax, wt)
    wmax = np.maximum(wmax, 1e-12)      # the centre pixel always keeps a non-zero weight (clean/edge case)
    acc += wmax[None] * z
    wsum += wmax
    return acc / wsum[None]


def denoise_bayer(mosaic, a, b, pattern="RGGB", strength=1.0, radius=2):
    """Denoise a Bayer mosaic (signed float, noise var = a*x+b). Returns a mosaic."""
    if strength <= 0:
        return np.asarray(mosaic, np.float32)
    p = pack_bayer(np.asarray(mosaic, np.float32), pattern)
    z = gat(p, a, b).astype(np.float32)
    zd = nlm_planes(z, guide_sigma2=0.25, strength=strength, radius=radius)
    out = gat_inverse(zd, a, b).astype(np.float32)
    return unpack_bayer(out, pattern)


# ---- RGB-domain filters (selectable in the UI / used for ablations) --------
def denoise_rgb(rgb, method="bilateral", strength=0.4):
    a = np.asarray(rgb, np.float32)
    s = float(np.clip(strength, 0, 1))
    if method == "none" or s == 0:
        return a.copy()
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, (0.3 + 1.6 * s, 0.3 + 1.6 * s, 0))
    if method == "median":
        from scipy.ndimage import median_filter
        return median_filter(a, size=(3 if s < .65 else 5,) * 2 + (1,))
    if method == "bilateral":
        if cv2 is None:
            raise RuntimeError("bilateral needs OpenCV")
        return cv2.bilateralFilter(a, 5, 0.02 + 0.25 * s, 3 + 3 * s)
    raise ValueError("Unknown RGB denoising method")


def denoise_chroma(rgb_srgb, sigma_px):
    """Smooth only chroma (Cb/Cr) in YCbCr: the eye is far less sensitive to chroma noise."""
    from scipy.ndimage import gaussian_filter
    from .color_correction import rgb_to_ycbcr, ycbcr_to_rgb
    if sigma_px <= 0.05:
        return rgb_srgb
    y = rgb_to_ycbcr(rgb_srgb)
    y[..., 1] = gaussian_filter(y[..., 1], sigma_px)
    y[..., 2] = gaussian_filter(y[..., 2], sigma_px)
    return ycbcr_to_rgb(y)
