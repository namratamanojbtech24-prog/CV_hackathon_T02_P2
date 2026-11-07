"""Model compression for mobile deployment: knowledge distillation, FP16 / INT8 post-training quantisation.

* Knowledge distillation   -> `python -m models.train --base 8 --distill <teacher>`  (student has ~4x fewer parameters)
* INT8 static PTQ (FX)     -> `quantize_int8()`   weights + activations in 8 bit, calibrated on simulated RAW frames
* FP16 / BF16              -> `to_half()`          (GPU / NPU precision; CPUs without native FP16 gain nothing)
Every technique is *measured* (size, latency, quality) by run_experiments.py: nothing is assumed to be faster.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from config import CKPT_DIR
from isp.learned_isp import build_model


def calibration_batches(mode="hybrid", n=24, seed=7, size=128):
    """Simulated RAW patches (from the TRAIN split) for activation-range calibration."""
    from models.train import make_sample_fn
    from utils.datasets import BatchSampler, load_pool, split_pool
    train, _, _ = split_pool(load_pool())
    s = BatchSampler(train, make_sample_fn(mode), batch=8, size=size, seed=seed)
    batches = [s.get() for _ in range(max(1, n // 8))]
    s.close()
    return [(torch.from_numpy(b["x"]), torch.from_numpy(b["ctx"])) for b in batches]


def quantize_int8(model, calib, backend="x86"):
    """Static post-training quantisation with torch.ao FX graph mode. Returns a quantised module."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        torch.backends.quantized.engine = "fbgemm"
    m = copy.deepcopy(model).eval().cpu()
    qmap = get_default_qconfig_mapping(torch.backends.quantized.engine)
    ex = calib[0]
    prepared = prepare_fx(m, qmap, ex)
    with torch.no_grad():
        for x, c in calib:
            prepared(x, c)
    return convert_fx(prepared)


def to_half(model, dtype=torch.float16):
    return copy.deepcopy(model).to(dtype)


def save_quantized(qmodel, path, example):
    """Serialise the INT8 model as TorchScript (FX graph modules do not pickle reliably)."""
    traced = torch.jit.trace(qmodel, example, strict=False)
    torch.jit.save(traced, str(path))
    return traced


def load_quantized(path):
    m = torch.jit.load(str(path), map_location="cpu")
    return m.eval()


def state_dict_bytes(model):
    """Serialized size of the model (what would ship in an app bundle)."""
    import io
    buf = io.BytesIO()
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, buf)
    else:
        torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
