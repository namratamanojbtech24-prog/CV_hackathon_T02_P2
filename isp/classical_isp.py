"""Traditional (fully classical) ISP: RAW mosaic -> display-referred sRGB.

Stage order (each stage is a separate, inspectable function):
  1 black-level / normalisation         (done when the RAW is loaded)
  2 exposure compensation (AE gain)     metadata, else estimated from image statistics
  3 blind noise-model estimation        var = a*x + b
  4 Bayer-domain denoising              GAT + guided NLM  (strength adapts to noise)
  5 demosaicing                         Malvar-He-Cutler
  6 white balance                       adaptive consensus of illuminant estimators
  7 colour correction matrix            camera RGB -> linear sRGB
  8 tone mapping + gamma                sRGB OETF (optional filmic / highlight roll-off)
  9 chroma denoise + luma sharpening    noise-aware coring
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .color_correction import apply_ccm
from .denoising import denoise_bayer, denoise_chroma, denoise_rgb
from .noise_estimation import estimate_noise_params, noise_level_at
from .raw_processing import RawImage, demosaic, pack_bayer, crop_even
from .sharpening import sharpen
from .tone_mapping import tone_map
from .white_balance import apply_gains, estimate_gains


@dataclass
class ISPResult:
    rgb: np.ndarray
    info: Dict = field(default_factory=dict)


def estimate_exposure_gain(mosaic, pattern="RGGB", target=0.18, max_gain=16.0):
    """Fallback auto-exposure: scale so the scene's mean luminance sits at mid-grey."""
    p = pack_bayer(mosaic, pattern)
    lum = np.maximum(p, 0).mean()
    return float(np.clip(target / max(lum, 1e-4), 1.0, max_gain))


def prepare(raw: RawImage):
    """Shared front-end: AE gain + blind noise estimate. Returns (amplified mosaic, a, b, gain)."""
    m = crop_even(np.asarray(raw.mosaic, np.float32))
    gain = float(raw.gain) if raw.gain else estimate_exposure_gain(m, raw.pattern)
    m = m * gain
    if raw.noise_a is not None and raw.noise_b is not None:
        a, b = raw.noise_a * gain, raw.noise_b * gain ** 2
    else:
        a, b = estimate_noise_params(m, raw.pattern, clip_level=gain)
    return m, a, b, gain


def traditional_isp(raw: RawImage, white_balance="adaptive", denoise_method="nlm-bayer", denoise_strength=1.0,
                    sharpen_amount=0.5, tone="srgb", demosaic_method="mhc", adaptive=True, ccm=None) -> ISPResult:
    t = {}
    t0 = time.perf_counter()
    m, a, b, gain = prepare(raw)
    t["prepare"] = time.perf_counter() - t0
    sigma_mid = noise_level_at(a, b)

    # noise-adaptive strength: stronger smoothing when the frame is noisier
    strength = denoise_strength
    if adaptive:
        strength = denoise_strength * float(np.clip(0.55 + sigma_mid / 0.06, 0.55, 1.6))

    t0 = time.perf_counter()
    if denoise_method == "nlm-bayer":
        m = denoise_bayer(m, a, b, raw.pattern, strength)
    t["denoise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rgb = demosaic(m, raw.pattern, demosaic_method)
    t["demosaic"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gains = np.asarray(raw.wb_gains, np.float32) if raw.wb_gains is not None and white_balance == "camera" \
        else estimate_gains(np.maximum(rgb, 0), "adaptive" if white_balance == "camera" else white_balance)
    rgb = apply_gains(rgb, gains)
    t["white_balance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rgb = apply_ccm(rgb, raw.ccm if raw.ccm is not None else ccm)
    t["ccm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = tone_map(np.clip(rgb, 0, None), tone)
    t["tone"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if denoise_method in ("bilateral", "gaussian", "median"):
        out = denoise_rgb(out, denoise_method, min(1.0, 0.25 + strength * 0.4))
    if adaptive:
        out = denoise_chroma(out, float(np.clip(sigma_mid * 25 * min(strength, 1.5), 0, 2.5)))
    sig_out = sigma_mid * 1.2 * np.sqrt(1.0 / max(strength, 0.5))
    out = sharpen(out, sharpen_amount * (0.5 if adaptive and sigma_mid > 0.08 else 1.0), noise_sigma=min(sig_out, 0.2) * 0.35)
    t["sharpen"] = time.perf_counter() - t0
    info = dict(pipeline="traditional", gain=gain, noise_a=a, noise_b=b, noise_sigma_mid=sigma_mid,
                wb_gains=[float(g) for g in gains], denoise_strength=strength, stage_ms={k: v * 1000 for k, v in t.items()},
                total_ms=sum(t.values()) * 1000)
    return ISPResult(np.clip(out, 0, 1).astype(np.float32), info)
