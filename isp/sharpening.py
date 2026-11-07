"""Edge-aware, noise-aware sharpening applied to luminance only."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .color_correction import rgb_to_ycbcr, ycbcr_to_rgb


def sharpen(srgb, amount=0.5, radius=1.2, noise_sigma=0.0):
    """Unsharp mask on luma with 'coring': detail below ~noise level is not amplified,
    so sharpening cannot turn residual noise into texture. Chroma is untouched (no colour fringes)."""
    a = np.asarray(srgb, np.float32)
    if amount <= 0:
        return a.copy()
    ycc = rgb_to_ycbcr(a)
    y = ycc[..., 0]
    detail = y - gaussian_filter(y, radius)
    tau = 1.5 * float(noise_sigma)
    detail = np.sign(detail) * np.maximum(np.abs(detail) - tau, 0)
    ycc[..., 0] = y + amount * detail
    return np.clip(ycbcr_to_rgb(ycc), 0, 1)
