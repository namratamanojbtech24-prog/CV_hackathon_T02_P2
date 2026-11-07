"""Gamma encoding and tone mapping (linear light -> display-referred sRGB)."""
from __future__ import annotations

import numpy as np


def srgb_oetf(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def gamma_correction(rgb, gamma=2.2):
    return np.clip(np.asarray(rgb, np.float32), 0, 1) ** (1.0 / max(float(gamma), 0.05))


def filmic_curve(x, strength=0.5):
    """Soft S-curve (blend between identity and smoothstep) applied on display-referred values."""
    x = np.clip(x, 0, 1)
    return (1 - strength) * x + strength * (x * x * (3 - 2 * x))


def highlight_compress(lin, knee=0.8):
    """Roll off values above `knee` smoothly toward 1 (extended Reinhard shoulder) instead of hard clipping."""
    lin = np.asarray(lin, np.float32)
    over = np.maximum(lin - knee, 0)
    head = 1 - knee
    comp = knee + head * (1 - np.exp(-over / head))
    return np.where(lin > knee, comp, lin)


def tone_map(lin, mode="srgb", strength=0.5):
    """mode: 'srgb' (faithful OETF) | 'filmic' (OETF + mild S-curve) | 'adaptive' (OETF; highlight roll-off
    only if a meaningful fraction of pixels is over-exposed)."""
    lin = np.asarray(lin, np.float32)
    if mode == "adaptive":
        if (lin.max(2) > 1.0).mean() > 0.002:
            lin = highlight_compress(lin)
        return srgb_oetf(lin)
    out = srgb_oetf(lin)
    if mode == "filmic":
        out = filmic_curve(out, strength)
    elif mode != "srgb":
        raise ValueError("tone mode must be srgb, filmic or adaptive")
    return out
