"""Bayer RAW handling: normalisation, packing, mosaicing and demosaicing.

All arrays are float32. A "mosaic" is a 2-D array (H, W) with one colour sample
per photosite. Normalised mosaics are *signed*: after black-level subtraction
noise may push a pixel below zero and that information is kept (clipping it
would bias dark regions brighter).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import correlate

from config import BAYER_PATTERN, BLACK_DN, WHITE_DN

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


@dataclass
class RawImage:
    """A normalised Bayer frame plus the metadata a camera ISP normally has."""
    mosaic: np.ndarray                       # (H, W) float32, black-subtracted, 1.0 = white level
    pattern: str = BAYER_PATTERN
    gain: Optional[float] = None             # exposure compensation (AE) known to the camera; None -> estimate
    noise_a: Optional[float] = None          # optional sensor calibration: var = a*x + b  (None -> estimate blind)
    noise_b: Optional[float] = None
    wb_gains: Optional[Tuple[float, float, float]] = None   # camera as-shot WB, optional
    ccm: Optional[np.ndarray] = None         # camera->sRGB 3x3, None -> config.SENSOR_CCM
    truth: dict = field(default_factory=dict)  # simulator ground truth (never used by pipelines)

    @property
    def shape(self):
        return self.mosaic.shape


# --------------------------------------------------------------------------- #
# Normalisation / layout
# --------------------------------------------------------------------------- #
def normalize_raw(dn, black=BLACK_DN, white=WHITE_DN) -> np.ndarray:
    """Digital numbers -> signed float with black at 0 and white at 1."""
    if white <= black:
        raise ValueError("RAW white level must exceed black level.")
    x = (np.asarray(dn, dtype=np.float32) - float(black)) / float(white - black)
    return np.minimum(x, 1.0)


def crop_even(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    return a[: h - h % 2, : w - w % 2]


def _check(pattern: str):
    if pattern not in PATTERNS:
        raise ValueError(f"Bayer pattern must be one of {PATTERNS}.")


def bayer_masks(shape, pattern=BAYER_PATTERN):
    """Boolean masks (R, G, B) for a mosaic of the given shape."""
    _check(pattern)
    h, w = shape
    masks = {c: np.zeros((h, w), bool) for c in "RGB"}
    for k, c in enumerate(pattern):
        masks[c][k // 2::2, k % 2::2] = True
    return masks["R"], masks["G"], masks["B"]


def pack_bayer(mosaic: np.ndarray, pattern=BAYER_PATTERN) -> np.ndarray:
    """(…, H, W) mosaic -> (…, 4, H/2, W/2) planes ordered [R, G1, G2, B]."""
    _check(pattern)
    m = mosaic[..., : mosaic.shape[-2] // 2 * 2, : mosaic.shape[-1] // 2 * 2]
    planes = {k: m[..., k // 2::2, k % 2::2] for k in range(4)}
    order, seen_g = [None] * 4, 0
    for k, c in enumerate(pattern):
        if c == "R":
            order[0] = planes[k]
        elif c == "B":
            order[3] = planes[k]
        else:
            order[1 + seen_g] = planes[k]
            seen_g += 1
    return np.stack(order, axis=-3)


def unpack_bayer(packed: np.ndarray, pattern=BAYER_PATTERN) -> np.ndarray:
    _check(pattern)
    *lead, _, h, w = packed.shape
    out = np.zeros((*lead, h * 2, w * 2), packed.dtype)
    seen_g = 0
    for k, c in enumerate(pattern):
        idx = 0 if c == "R" else 3 if c == "B" else 1 + seen_g
        seen_g += c == "G"
        out[..., k // 2::2, k % 2::2] = packed[..., idx, :, :]
    return out


def mosaic_from_rgb(rgb: np.ndarray, pattern=BAYER_PATTERN) -> np.ndarray:
    """Sample a full RGB image (…, H, W, 3) through a colour filter array."""
    h, w = rgb.shape[-3:-1]
    r, g, b = bayer_masks((h, w), pattern)
    out = np.zeros(rgb.shape[:-1], rgb.dtype)
    out[..., r] = rgb[..., r, 0]
    out[..., g] = rgb[..., g, 1]
    out[..., b] = rgb[..., b, 2]
    return out


# --------------------------------------------------------------------------- #
# Demosaicing
# --------------------------------------------------------------------------- #
_MHC = {
    "g_at_rb": np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                         [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8,
    "rb_at_g_same_row": np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                                  [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], np.float32) / 8,
    "rb_at_g_same_col": np.array([[0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [.5, 0, 5, 0, .5],
                                  [0, -1, 4, -1, 0], [0, 0, -1, 0, 0]], np.float32) / 8,
    "rb_at_br": np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                          [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8,
}
_BILINEAR_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
_BILINEAR_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4


def _corr(img, k):
    """Correlate the last two axes of img with 2-D kernel k (reflect borders)."""
    kk = k.reshape((1,) * (img.ndim - 2) + k.shape)
    return correlate(img, kk, mode="mirror")


def demosaic(mosaic: np.ndarray, pattern: str = BAYER_PATTERN, method: str = "mhc") -> np.ndarray:
    """(…, H, W) mosaic -> (…, H, W, 3) linear RGB.

    method: 'bilinear' | 'mhc' (Malvar-He-Cutler, gradient-corrected linear
    interpolation: sharper than bilinear with far fewer zipper artefacts) |
    'opencv' (edge-aware, needs OpenCV; single image only).
    """
    _check(pattern)
    m = np.asarray(mosaic, np.float32)
    h, w = m.shape[-2:]
    r, g, b = bayer_masks((h, w), pattern)

    if method == "opencv":
        import cv2
        # OpenCV names Bayer codes by the *second* row/column offset, so RGGB maps to "BG" (and so on).
        code = {"RGGB": cv2.COLOR_BayerBG2RGB_EA, "BGGR": cv2.COLOR_BayerRG2RGB_EA,
                "GRBG": cv2.COLOR_BayerGB2RGB_EA, "GBRG": cv2.COLOR_BayerGR2RGB_EA}[pattern]
        u = np.clip(m * 65535 + 0.0, 0, 65535).astype(np.uint16)
        return cv2.cvtColor(u, code).astype(np.float32) / 65535

    if method == "bilinear":
        out = np.zeros((*m.shape, 3), np.float32)
        out[..., 1] = _corr(m * g, _BILINEAR_G)
        out[..., 0] = _corr(m * r, _BILINEAR_RB)
        out[..., 2] = _corr(m * b, _BILINEAR_RB)
        for c, mk in ((0, r), (1, g), (2, b)):          # keep measured samples exact
            out[..., c][..., mk] = m[..., mk]
        return out
    if method != "mhc":
        raise ValueError("demosaic method must be 'mhc', 'bilinear' or 'opencv'.")

    # Position classes. "same row" = the red (or blue) neighbours lie on the same row as this green.
    row_of_r = np.zeros((h, w), bool)
    for k, c in enumerate(pattern):
        if c == "R":
            row_of_r[k // 2::2, :] = True
    gr_row = g & row_of_r          # green sitting on a row that also contains R  -> R neighbours horizontal
    gb_row = g & ~row_of_r         # green sitting on a row that contains B       -> B neighbours horizontal

    g_est = _corr(m, _MHC["g_at_rb"])
    rb_row = _corr(m, _MHC["rb_at_g_same_row"])
    rb_col = _corr(m, _MHC["rb_at_g_same_col"])
    rb_diag = _corr(m, _MHC["rb_at_br"])

    out = np.empty((*m.shape, 3), np.float32)
    out[..., 1] = np.where(g, m, g_est)
    # red channel
    R = np.where(r, m, 0.0).astype(np.float32)
    R = np.where(gr_row, rb_row, R)
    R = np.where(gb_row, rb_col, R)
    R = np.where(b, rb_diag, R)
    # blue channel
    B = np.where(b, m, 0.0).astype(np.float32)
    B = np.where(gb_row, rb_row, B)
    B = np.where(gr_row, rb_col, B)
    B = np.where(r, rb_diag, B)
    out[..., 0], out[..., 2] = R, B
    return out


# --------------------------------------------------------------------------- #
# Loading real RAW data
# --------------------------------------------------------------------------- #
def load_dng(path) -> RawImage:
    """Load a DNG/other RAW via rawpy (optional dependency)."""
    try:
        import rawpy
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("RAW/DNG support needs the optional 'rawpy' package (pip install rawpy).") from e
    with rawpy.imread(str(path)) as r:
        a = r.raw_image_visible.astype(np.float32)
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
        desc = r.color_desc.decode() if isinstance(r.color_desc, bytes) else str(r.color_desc)
        pat = "".join(desc[i] for i in r.raw_pattern.flatten()) if desc.startswith("RGB") else BAYER_PATTERN
        wb = np.array(r.camera_whitebalance[:3], np.float32)
        wb = wb / max(wb[1], 1e-6)
        cm = np.array(r.rgb_xyz_matrix[:3, :3], np.float32) if r.rgb_xyz_matrix is not None else None
    mosaic = crop_even(normalize_raw(a, black, white))
    return RawImage(mosaic=mosaic, pattern=pat if pat in PATTERNS else BAYER_PATTERN,
                    wb_gains=tuple(float(v) for v in wb))


def load_raw_array(path_or_array, black=BLACK_DN, white=WHITE_DN, pattern=BAYER_PATTERN) -> RawImage:
    """Load a 2-D mosaic from .npy/.tif/.png (uint16 DN) or an array."""
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        p = str(path_or_array)
        if p.lower().endswith(".npy"):
            a = np.load(p)
        else:
            from PIL import Image
            a = np.asarray(Image.open(p))
    else:
        a = np.asarray(path_or_array)
    if a.ndim != 2:
        raise ValueError("A Bayer RAW mosaic must be a 2-D array (one sample per photosite).")
    if a.dtype != np.float32 and a.max() > 1.5:
        mosaic = normalize_raw(a, black, white)
    else:
        mosaic = a.astype(np.float32)
    return RawImage(mosaic=crop_even(mosaic), pattern=pattern)
