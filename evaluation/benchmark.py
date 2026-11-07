"""Efficiency measurements: parameters, size, MACs, latency, peak memory."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict

import numpy as np
import torch

from models.unet_isp import count_macs, model_info


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return None


class PeakMemory:
    """Samples process RSS in a background thread; `delta_mb` = peak - baseline while the context is active."""

    def __enter__(self):
        self.base = _rss()
        self.peak = self.base
        self._stop = False
        if self.base is not None:
            def loop():
                while not self._stop:
                    r = _rss()
                    if r is not None and r > self.peak:
                        self.peak = r
                    time.sleep(0.002)
            self.t = threading.Thread(target=loop, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        if self.base is not None:
            self.t.join()

    @property
    def delta_mb(self):
        return None if self.base is None else max(self.peak - self.base, 0) / 1e6


def time_fn(fn: Callable, warmup=2, runs=7):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "min_ms": float(np.min(ts))}


@torch.inference_mode()
def benchmark_model(model, megapixels=1.0, runs=7, dtype=torch.float32, ctx_dim=None, in_ch=None, tile=None, name="model"):
    """Model-only cost at a given image size (full-resolution MP; the network runs at half-res packed)."""
    ctx_dim = ctx_dim or getattr(_REF.get('ref'), 'ctx_dim', 10)
    in_ch = in_ch or getattr(_REF.get('ref'), 'in_ch', 12)
    side = int(round((megapixels * 1e6) ** 0.5 / 16) * 16)
    h = w = side // 2
    x = torch.randn(1, in_ch, h, w).to(dtype)
    c = torch.randn(1, ctx_dim).to(dtype)
    model = model.eval()
    from isp.learned_isp import run_network

    def fwd():
        if tile:
            run_network(model, x[0].float().numpy(), c[0].float().numpy(), tile)
        else:
            model(x, c)

    with PeakMemory() as pm:
        lat = time_fn(fwd, runs=runs)
    out = {"name": name, "megapixels": side * side / 1e6, "latency_ms": lat["median_ms"], "latency_p90_ms": lat["p90_ms"],
           "peak_memory_mb": pm.delta_mb, "tile": tile, "threads": torch.get_num_threads()}
    try:
        out["gmacs_per_mp"] = count_macs(model_fp32_ref(model), (h, w)) / 1e9 / (side * side / 1e6)
    except Exception:
        pass
    return out


_REF = {}


def model_fp32_ref(m):
    return _REF.get("ref", m)


def set_macs_reference(m):
    _REF["ref"] = m


def model_summary(model, name="model"):
    info = model_info(model)
    info["name"] = name
    return info


def pipeline_stage_timing(fn, raw, runs=3):
    """Run a full ISP several times; return median total and per-stage milliseconds."""
    fn(raw)
    results = [fn(raw).info for _ in range(runs)]
    stages = {k: float(np.median([r["stage_ms"][k] for r in results])) for k in results[0]["stage_ms"]}
    return {"total_ms": float(np.median([r["total_ms"] for r in results])), "stage_ms": stages}


# --------------------------------------------------------------------------- reliable peak-memory measurement
_PEAK_SNIPPET = r"""
import sys, json, torch
torch.set_num_threads(int(sys.argv[4]))
from isp.learned_isp import load_model, run_network
key, mp, tile = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
model = load_model(key)[0].eval()
in_ch, ctx_dim = getattr(model, 'in_ch', 12), getattr(model, 'ctx_dim', 10)
side = int(round((mp * 1e6) ** 0.5 / 16) * 16); h = w = side // 2
x, c = torch.randn(1, in_ch, h, w), torch.randn(1, ctx_dim)
def rss_kb():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS:'): rss = int(l.split()[1])
        if l.startswith('VmHWM:'): hwm = int(l.split()[1])
    return rss, hwm
with torch.inference_mode():
    model(x[:, :, :32, :32], c)                       # warm-up: allocator pools, JIT, thread pools
    open('/proc/self/clear_refs', 'w').write('5')     # reset the high-water mark
    base, _ = rss_kb()
    if tile: run_network(model, x[0].numpy(), c[0].numpy(), tile)
    else: model(x, c)
    _, peak = rss_kb()
print(json.dumps({"peak_mb": max(peak - base, 0) / 1024}))
"""


def peak_memory_mb(key: str, megapixels: float = 1.0, tile: int = 0, threads: int = 2):
    """Extra resident memory (MB) a forward pass needs, measured in a fresh process so that earlier
    allocations cannot hide it. Uses the kernel's high-water mark (Linux); returns None elsewhere."""
    import json
    import subprocess
    import sys
    from config import ROOT
    if not sys.platform.startswith("linux"):
        return None
    try:
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", _PEAK_SNIPPET, key, str(megapixels), str(tile), str(threads)],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        return float(json.loads(out.stdout.strip().splitlines()[-1])["peak_mb"])
    except Exception:
        return None
