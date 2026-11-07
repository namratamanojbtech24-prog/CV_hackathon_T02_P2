"""Robustness sweeps: how each ISP degrades (and recovers) as noise, light level and illuminant get harder."""
from __future__ import annotations

import json
import time
from typing import Dict, Iterable

import numpy as np

from evaluation.comparison import run_pipeline
from evaluation.metrics import calculate_metrics
from utils.datasets import center_crop
from utils.degradation import Scenario, naive_preview, simulate_raw

SWEEPS = {
    "noise": dict(label="Sensor noise (ISO)", xs=[100, 400, 1600, 3200, 6400, 12800],
                  make=lambda v: Scenario(iso=v, ev_under=1.0, kelvin=5000), fmt="ISO {:g}"),
    "light": dict(label="Light level (stops under-exposed)", xs=[0.0, 1.0, 2.0, 3.0, 4.0],
                  make=lambda v: Scenario(iso=1600, ev_under=v, kelvin=5000), fmt="-{:g} EV"),
    "illuminant": dict(label="Illuminant colour temperature (K)", xs=[2500, 3200, 4500, 6500, 8500],
                       make=lambda v: Scenario(iso=1600, ev_under=1.0, kelvin=v), fmt="{:g} K"),
}


def run_sweeps(test_imgs: Dict[str, np.ndarray], models, names: Iterable[str], size=256, seed=0, sweeps=None, keys=("PSNR", "SSIM", "Delta E", "GMSD")):
    out = {}
    for sk in (sweeps or SWEEPS):
        sw = SWEEPS[sk]
        res = {n: {k: [] for k in keys} for n in names}
        res["Unprocessed (naive)"] = {k: [] for k in keys}
        for v in sw["xs"]:
            acc = {n: {k: [] for k in keys} for n in res}
            for i, (nm, img) in enumerate(test_imgs.items()):
                gt = center_crop(img, size).astype(np.float32) / 255
                raw = simulate_raw(gt, sw["make"](v), np.random.default_rng(seed + i))
                for n in names:
                    m = calculate_metrics(run_pipeline(n, raw, models).rgb, gt, False)
                    for k in keys:
                        acc[n][k].append(m[k])
                m = calculate_metrics(naive_preview(raw), gt, False)
                for k in keys:
                    acc["Unprocessed (naive)"][k].append(m[k])
            for n in res:
                for k in keys:
                    res[n][k].append(float(np.mean(acc[n][k])))
        out[sk] = {"label": sw["label"], "x": sw["xs"], "xlabels": [sw["fmt"].format(v) for v in sw["xs"]], "series": res}
    return out
