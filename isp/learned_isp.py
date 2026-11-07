"""Learned & hybrid inference: feature construction, model loading, tiled execution.

Two learned configurations share the same U-Net (models.unet_isp.ISPUNet):

  learned : packed RAW (no WB, no demosaic, no CCM)  ->  U-Net  ->  sRGB          (end-to-end)
  hybrid  : classical front-end (gain, Bayer NLM denoise, MHC demosaic, AWB, CCM, sRGB curve) -> U-Net residual correction
            (fed with the WB'd RAW planes + noise map) -> classical luma sharpening       (this project's design)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

from config import CHECKPOINTS, SENSOR_CCM
from models.unet_isp import CTX_DIM, IN_CH, ISPUNet
from .classical_isp import ISPResult, prepare
from .denoising import denoise_bayer, denoise_chroma
from .color_correction import apply_ccm
from .noise_estimation import noise_level_at
from .raw_processing import RawImage, crop_even, demosaic, pack_bayer
from .sharpening import sharpen
from .tone_mapping import srgb_oetf
from .white_balance import apply_gains, estimate_gains

SIGMA_SCALE = 4.0
CCM_T = torch.from_numpy(SENSOR_CCM.astype(np.float32))


# --------------------------------------------------------------------------- features
IN_CH_BY_MODE = {"learned": 8, "hybrid": 12}
CTX_BY_MODE = {"learned": 8, "hybrid": 10}


def adaptive_strength(a, b, base=1.0):
    """Classical denoise strength as a function of the estimated noise (shared by traditional & hybrid)."""
    return base * float(np.clip(0.55 + noise_level_at(a, b) / 0.06, 0.55, 1.6))


def chroma_sigma(a, b):
    return float(np.clip(noise_level_at(a, b) * 25 * min(adaptive_strength(a, b), 1.5), 0, 2.5))


def build_features(m, a, b, pattern, mode, gains=None, gain=1.0, ccm=None):
    """m: amplified signed mosaic (H,W). Returns dict(x, ctx, skip) as float32 numpy arrays.

    x    (8|12, H/2, W/2)  packed planes (WB'd for 'hybrid') + noise-level maps
                           (+ classically denoised planes for 'hybrid')
    ctx  (8,)           global descriptor
    skip (3, H, W)      display-referred base image the network residual is added to
                        hybrid : classical result  oetf(CCM @ (WB * MHC-demosaic))   (noisy, but colour-managed)
                        learned: gamma-encoded binned RGB (no WB / CCM: the net must learn them)
    """
    packed = pack_bayer(m, pattern).astype(np.float32)                         # (4,h,w)
    g = np.ones(3, np.float32) if gains is None or mode == "learned" else np.asarray(gains, np.float32)
    pg = np.array([g[0], g[1], g[1], g[2]], np.float32)[:, None, None]
    smooth = uniform_filter(packed, size=(1, 3, 3), mode="reflect")
    sigma = np.sqrt(np.maximum(a * np.maximum(smooth, 0) + b, 1e-12)) * pg     # noise std AFTER the white-balance gains
    planes = packed * pg
    chans = [planes, sigma * SIGMA_SCALE]
    md = None
    if mode == "hybrid":      # cheap classical Bayer-domain denoise: the network refines it instead of starting from noise
        md = denoise_bayer(m, a, b, pattern, adaptive_strength(a, b))
        chans.append(pack_bayer(md, pattern) * pg)
    x = np.concatenate(chans, 0).astype(np.float32)

    rgb = np.stack([planes[0], 0.5 * (planes[1] + planes[2]), planes[3]], 0)    # binned RGB (h,w)
    mean = np.maximum(rgb.reshape(3, -1).mean(1), 1e-3)
    p99 = np.maximum(np.percentile(rgb.reshape(3, -1), 99, axis=1), 1e-3)
    sig_mid = noise_level_at(a, b)
    ctx = np.concatenate([np.log(mean), np.log(p99), [np.log(max(sig_mid, 1e-4)), np.log(max(gain, 1e-3))]]).astype(np.float32)

    if mode == "hybrid":
        lin = demosaic(md, pattern, "mhc") * g
        lin = apply_ccm(lin, SENSOR_CCM if ccm is None else ccm)
        base = denoise_chroma(srgb_oetf(np.maximum(lin, 0)), chroma_sigma(a, b))      # same cheap chroma clean-up as the classical ISP
        skip = np.moveaxis(base, -1, 0)
        ctx = np.concatenate([ctx, np.log(g[[0, 2]])]).astype(np.float32)
    else:
        binned = np.clip(rgb, 0, 1)
        skip = np.repeat(np.repeat(srgb_oetf(np.moveaxis(binned, 0, -1)).transpose(2, 0, 1), 2, 1), 2, 2)
    return {"x": x, "ctx": ctx, "skip": skip.astype(np.float32)}


# --------------------------------------------------------------------------- composition
def compose(residual, skip, mode=None, ccm=None):
    """Display-referred output = skip + learned residual (differentiable; `mode`/`ccm` kept for API symmetry)."""
    return skip + residual


# --------------------------------------------------------------------------- model loading
_CACHE: Dict[str, tuple] = {}


def build_model(arch: dict) -> ISPUNet:
    return ISPUNet(base=arch.get("base", 16), in_ch=arch.get("in_ch", IN_CH), ctx_dim=arch.get("ctx_dim", CTX_DIM),
                   zero_init=False)


def load_model(name_or_path="hybrid", device="cpu", use_cache=True):
    """Return (model, meta). `name_or_path` is a key of config.CHECKPOINTS or a file path."""
    path = CHECKPOINTS.get(name_or_path, name_or_path)
    key = f"{path}:{device}"
    if use_cache and key in _CACHE:
        return _CACHE[key]
    path = Path(path)
    if path.name.endswith("_int8.pt") and path.exists():       # TorchScript INT8 model
        from models.optimize import load_quantized
        out = (load_quantized(path), {"quantized": True})
        _CACHE[key] = out
        return out
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Train one with `python -m models.train` "
                                f"(see README) or run `python run_experiments.py`.")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_model(blob["arch"]).to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    meta = {k: v for k, v in blob.items() if k not in ("state", "scripted", "quantized_state")}
    _CACHE[key] = (model, meta)
    return model, meta


def available_models():
    return {k: Path(v).exists() for k, v in CHECKPOINTS.items()}


# --------------------------------------------------------------------------- tiled inference
@torch.inference_mode()
def run_network(model, x, ctx, tile=None, overlap=12):
    """x (8,h,w) numpy, ctx (8,) -> residual (3,2h,2w) numpy. Optional tiling bounds peak memory;
    the global context vector is shared by every tile, so seams are not introduced."""
    xt = torch.from_numpy(x)[None]
    ct = torch.from_numpy(ctx)[None]
    h, w = x.shape[1:]

    def pad8(t):
        ph, pw = (-t.shape[-2]) % 8, (-t.shape[-1]) % 8
        return F.pad(t, (0, pw, 0, ph), mode="reflect") if (ph or pw) else t

    if tile is None or (h <= tile and w <= tile):
        y = model(pad8(xt), ct)
        return y[0, :, : 2 * h, : 2 * w].float().numpy()
    out = torch.zeros(1, 3, 2 * h, 2 * w)
    step = max(tile - 2 * overlap, 8)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            ya, yb = max(y0 - overlap, 0), min(y0 + step + overlap, h)
            xa, xb = max(x0 - overlap, 0), min(x0 + step + overlap, w)
            tt = xt[:, :, ya:yb, xa:xb]
            th, tw = tt.shape[-2:]
            r = model(pad8(tt), ct)[:, :, : 2 * th, : 2 * tw]
            cy0, cx0 = y0 - ya, x0 - xa
            ch, cw = min(step, h - y0), min(step, w - x0)
            out[:, :, 2 * y0:2 * (y0 + ch), 2 * x0:2 * (x0 + cw)] = r[:, :, 2 * cy0:2 * (cy0 + ch), 2 * cx0:2 * (cx0 + cw)]
    return out[0].numpy()


# --------------------------------------------------------------------------- pipelines
def learned_isp(raw: RawImage, model_name="learned", model=None, tile=None, sharpen_amount=0.0) -> ISPResult:
    t0 = time.perf_counter()
    mdl = model if model is not None else load_model(model_name)[0]
    m, a, b, gain = prepare(raw)
    f = build_features(m, a, b, raw.pattern, "learned", gain=gain)
    t1 = time.perf_counter()
    res = run_network(mdl, f["x"], f["ctx"], tile)
    t2 = time.perf_counter()
    out = np.clip(np.moveaxis(f["skip"] + res, 0, -1), 0, 1)
    if sharpen_amount > 0:
        out = sharpen(out, sharpen_amount, noise_sigma=0.01)
    t3 = time.perf_counter()
    info = dict(pipeline="learned", gain=gain, noise_a=a, noise_b=b, noise_sigma_mid=noise_level_at(a, b),
                stage_ms={"features": (t1 - t0) * 1e3, "network": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3},
                total_ms=(t3 - t0) * 1e3)
    return ISPResult(out.astype(np.float32), info)


def hybrid_isp(raw: RawImage, model_name="hybrid", model=None, white_balance="blend", sharpen_amount=0.5,
               tone="srgb", strength=1.0, tile=None, adaptive=True, skip_threshold=None, ccm=None) -> ISPResult:
    """Classical front-end (gain, MHC demosaic, AWB, CCM, sRGB curve) -> learned residual -> classical sharpening.

    strength        scales the learned residual (0 = purely classical demosaic/CCM/tone, 1 = full model)
    skip_threshold  if set and the estimated noise at mid-grey is below it, the network is skipped
                    (adaptive compute: clean, well-lit frames need no learned denoiser)
    """
    t = {}
    t0 = time.perf_counter()
    m, a, b, gain = prepare(raw)
    sig = noise_level_at(a, b)
    t["prepare"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    rgb0 = demosaic(m, raw.pattern, "mhc")
    if white_balance == "camera" and raw.wb_gains is not None:       # as-shot gains from the camera / DNG metadata
        gains = np.asarray(raw.wb_gains, np.float32)
    else:
        gains = estimate_gains(np.maximum(rgb0, 0), "blend" if white_balance == "camera" else white_balance)
    t["awb"] = (time.perf_counter() - t0) * 1e3

    use_net = not (skip_threshold is not None and sig < skip_threshold) and strength > 0
    t0 = time.perf_counter()
    ccm_m = np.asarray(raw.ccm if raw.ccm is not None else (ccm if ccm is not None else SENSOR_CCM), np.float32)
    f = build_features(m, a, b, raw.pattern, "hybrid", gains=gains, gain=gain, ccm=ccm_m)
    t["classical front-end"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    if use_net:
        mdl = model if model is not None else load_model(model_name)[0]
        res = run_network(mdl, f["x"], f["ctx"], tile) * float(strength)
    else:
        res = np.zeros_like(f["skip"])
    t["network"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    out = np.clip(np.moveaxis(f["skip"] + res, 0, -1), 0, 1)
    if tone != "srgb":                      # optional creative tone curve on the display-referred image
        from .tone_mapping import filmic_curve
        out = filmic_curve(out, 0.5)
    sig_out = sig * (0.5 if use_net else 1.0)
    amount = sharpen_amount * (0.5 if adaptive and sig > 0.08 else 1.0)
    out = sharpen(out, amount, noise_sigma=min(sig_out * 0.45, 0.08))
    t["backend"] = (time.perf_counter() - t0) * 1e3
    info = dict(pipeline="hybrid", gain=gain, noise_a=a, noise_b=b, noise_sigma_mid=sig, wb_gains=[float(g) for g in gains],
                used_network=bool(use_net), stage_ms=t, total_ms=sum(t.values()))
    return ISPResult(np.clip(out, 0, 1).astype(np.float32), info)
