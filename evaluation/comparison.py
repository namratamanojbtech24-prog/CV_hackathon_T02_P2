"""Run every ISP on the same RAW frame and score it against the clean reference."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from isp.classical_isp import traditional_isp
from isp.raw_processing import RawImage
from .metrics import calculate_metrics

PIPELINES = ("Traditional ISP", "Learned ISP", "Hybrid ISP")


def run_pipeline(name: str, raw: RawImage, models: Optional[dict] = None, **kw):
    """name in PIPELINES (+ 'Hybrid-Lite', 'Hybrid INT8' when those checkpoints exist). Returns ISPResult."""
    from isp.learned_isp import hybrid_isp, learned_isp
    models = models or {}
    if name == "Traditional ISP":
        return traditional_isp(raw, **kw)
    if name == "Learned ISP":
        return learned_isp(raw, model=models.get("learned"), **kw)
    if name == "Hybrid ISP":
        return hybrid_isp(raw, model=models.get("hybrid"), **kw)
    if name == "Hybrid-Lite":
        return hybrid_isp(raw, model=models.get("hybrid_lite"), **kw)
    if name == "Hybrid INT8":
        return hybrid_isp(raw, model=models.get("hybrid_int8"), **kw)
    raise ValueError(name)


def compare(raw: RawImage, reference: Optional[np.ndarray] = None, names=PIPELINES, models=None, lpips_on=False):
    outputs, rows = {}, []
    for n in names:
        t = time.perf_counter()
        res = run_pipeline(n, raw, models)
        ms = (time.perf_counter() - t) * 1e3
        outputs[n] = res
        rows.append({"Method": n, **calculate_metrics(res.rgb, reference, lpips_on), "Time (ms)": ms,
                     "info": res.info})
    return outputs, rows
