"""Colour correction matrix (camera RGB -> linear sRGB) and colour-space transforms."""
from __future__ import annotations

import numpy as np

from config import SENSOR_CCM

_YCC = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], np.float32)
_YCC_INV = np.linalg.inv(_YCC).astype(np.float32)


def apply_ccm(rgb, matrix=None):
    """Apply a 3x3 matrix (no clipping: callers decide when to clip)."""
    m = np.asarray(SENSOR_CCM if matrix is None else matrix, np.float32)
    if m.shape != (3, 3):
        raise ValueError("Colour correction matrix must be 3x3.")
    return np.asarray(rgb, np.float32) @ m.T


def rgb_to_ycbcr(rgb):
    return np.asarray(rgb, np.float32) @ _YCC.T


def ycbcr_to_rgb(ycc):
    return np.asarray(ycc, np.float32) @ _YCC_INV.T


def rgb_to_lab(rgb):
    from skimage.color import rgb2lab
    return rgb2lab(np.clip(rgb, 0, 1))


def color_space(rgb, space):
    """sRGB (0..1) -> 'YCbCr' | 'LAB' | 'HSV' (colour-space transformation stage)."""
    if space == "YCbCr":
        return rgb_to_ycbcr(rgb)
    if space == "LAB":
        return rgb_to_lab(rgb)
    if space == "HSV":
        from skimage.color import rgb2hsv
        return rgb2hsv(np.clip(rgb, 0, 1))
    raise ValueError("space must be YCbCr, LAB or HSV")
