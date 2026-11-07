"""Image-quality metrics. Everything is computed on display-referred sRGB in [0, 1].

PSNR  – fidelity (dB, higher better)          SSIM / MS-SSIM – structural similarity (higher better)
ΔE2000 – CIEDE2000 colour error (lower better; ~1 is just noticeable)
GMSD  – gradient-magnitude similarity deviation, a fast perceptual metric (lower better)
LPIPS – learned perceptual distance (lower better). Needs the optional `lpips` package and its
        pretrained AlexNet weights (downloaded on first use); reported as None when unavailable.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, convolve

from utils.image_utils import to_float_rgb

_LPIPS = {"model": None, "failed": False}


def _match(a, b):
    a, b = to_float_rgb(a), to_float_rgb(b)
    if a.shape[:2] != b.shape[:2]:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    return a, b


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def delta_e(a, b):
    from skimage.color import rgb2lab, deltaE_ciede2000
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


def _lum(a):
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def ms_ssim(a, b, scales=3):
    from skimage.metrics import structural_similarity
    x, y = _lum(a), _lum(b)
    vals = []
    for s in range(scales):
        if min(x.shape) < 16:
            break
        vals.append(max(structural_similarity(x, y, data_range=1.0), 1e-4))
        x = x[: x.shape[0] // 2 * 2, : x.shape[1] // 2 * 2].reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).mean((1, 3))
        y = y[: y.shape[0] // 2 * 2, : y.shape[1] // 2 * 2].reshape(y.shape[0] // 2, 2, y.shape[1] // 2, 2).mean((1, 3))
    return float(np.prod(np.power(vals, 1.0 / len(vals))))


def gmsd(a, b):
    """Gradient Magnitude Similarity Deviation (Xue et al. 2014)."""
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32) / 3
    x, y = _lum(a), _lum(b)
    x, y = x[::1, ::1], y[::1, ::1]
    def grad(z):
        z = uniform_filter(z, 2)[::2, ::2]
        return np.hypot(convolve(z, kx), convolve(z, kx.T))
    gx, gy = grad(x), grad(y)
    c = 0.0026
    gms = (2 * gx * gy + c) / (gx ** 2 + gy ** 2 + c)
    return float(gms.std())


def lpips_distance(a, b):
    if _LPIPS["failed"]:
        return None
    try:
        import torch
        if _LPIPS["model"] is None:
            import lpips
            _LPIPS["model"] = lpips.LPIPS(net="alex", verbose=False).eval()
        t = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)[None]) * 2 - 1).float()
        with torch.no_grad():
            return float(_LPIPS["model"](t(a), t(b)).item())
    except Exception:
        _LPIPS["failed"] = True
        return None


def calculate_metrics(output, reference=None, lpips_on=True):
    keys = ("PSNR", "SSIM", "Delta E", "MS-SSIM", "GMSD", "LPIPS")
    if reference is None:
        return {k: None for k in keys}
    a, b = _match(output, reference)
    return {"PSNR": psnr(a, b), "SSIM": ssim(a, b), "Delta E": delta_e(a, b),
            "MS-SSIM": ms_ssim(a, b), "GMSD": gmsd(a, b),
            "LPIPS": lpips_distance(a, b) if lpips_on else None}
