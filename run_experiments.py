"""Reproduce every number, figure and checkpoint in results/.

    python run_experiments.py                 # train (if needed) + evaluate + figures  (~1 h on a laptop CPU)
    python run_experiments.py --skip-train    # evaluation only, using existing checkpoints
    python run_experiments.py --steps 1500    # shorter training
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINTS, RESULTS
from download_data import main as fetch_data
from evaluation.benchmark import PeakMemory, benchmark_model, peak_memory_mb, pipeline_stage_timing, set_macs_reference
from evaluation.comparison import run_pipeline
from evaluation.metrics import calculate_metrics
from evaluation.robustness import run_sweeps
from isp.learned_isp import load_model
from models.unet_isp import count_macs, model_info
from utils.datasets import center_crop, load_pool, split_pool
from utils.degradation import SCENARIOS, naive_preview, simulate_raw, raw_to_dn
from utils.visualization import save_comparison, save_efficiency, save_sweeps

NAMES = ["Traditional ISP", "Learned ISP", "Hybrid ISP"]
EXTRA = {"Hybrid-Lite": "hybrid_lite", "Hybrid INT8": "hybrid_int8"}


def ensure_models(steps, base, student_base, skip_train):
    from models.train import train
    from models.optimize import calibration_batches, quantize_int8, save_quantized
    C = CHECKPOINTS
    if not skip_train:
        if not C["hybrid"].exists():
            train("hybrid", base, steps, out=C["hybrid"])
        if not C["learned"].exists():
            train("learned", base, steps, out=C["learned"])
        if not C["hybrid_lite"].exists():
            train("hybrid", student_base, steps, out=C["hybrid_lite"], distill=C["hybrid"])
    if C["hybrid_lite"].exists() and not C["hybrid_int8"].exists():
        m, meta = load_model("hybrid_lite")
        try:
            calib = calibration_batches("hybrid")
            q = quantize_int8(m, calib)
            save_quantized(q, C["hybrid_int8"], calib[0])
        except Exception as e:                       # pragma: no cover
            print("INT8 quantisation failed:", e)


def load_all():
    models = {}
    for k in ("learned", "hybrid", "hybrid_lite", "hybrid_int8"):
        if CHECKPOINTS[k].exists():
            models[k] = load_model(k)[0]
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--student-base", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--size", type=int, default=256, help="evaluation crop size")
    a = ap.parse_args()
    fetch_data()
    ensure_models(a.steps, a.base, a.student_base, a.skip_train)
    models = load_all()
    names = NAMES + [n for n, k in EXTRA.items() if k in models]
    pool = load_pool()
    _, _, test = split_pool(pool)
    print(f"{len(test)} held-out test images; pipelines: {names}", flush=True)

    # ---- 1. headline table over scenarios ---------------------------------
    keys = ("PSNR", "SSIM", "Delta E", "MS-SSIM", "GMSD", "LPIPS")
    table = {}
    samples = {}
    for sc_name, sc in SCENARIOS.items():
        acc = {n: {k: [] for k in keys} for n in names + ["Unprocessed (naive)"]}
        for i, (nm, img) in enumerate(test.items()):
            gt = center_crop(img, a.size).astype(np.float32) / 255
            raw = simulate_raw(gt, sc, np.random.default_rng(100 + i))
            outs = {n: run_pipeline(n, raw, models) for n in names}
            outs["Unprocessed (naive)"] = type("R", (), {"rgb": naive_preview(raw), "info": {}})
            for n, r in outs.items():
                m = calculate_metrics(r.rgb, gt, lpips_on=(n in names))
                for k in keys:
                    if m[k] is not None:
                        acc[n][k].append(m[k])
            if i == 0:
                samples[sc_name] = (gt, {"Unprocessed (naive)": outs["Unprocessed (naive)"].rgb, **{n: outs[n].rgb for n in names}},
                                    {n: calculate_metrics(outs[n].rgb, gt, False) for n in names})
        table[sc_name] = {n: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} for n, d in acc.items()}
        print(sc_name, {n: round(table[sc_name][n]["PSNR"], 2) for n in table[sc_name]}, flush=True)
    for sc_name, (gt, imgs, met) in samples.items():
        imgs = {**imgs, "Ground truth": gt}
        save_comparison(imgs, RESULTS / f"compare_{sc_name.lower().replace(' ', '_').replace(',', '')}.png", met, sc_name)
    gt, imgs, met = samples["Low light"]
    save_comparison({**imgs, "Ground truth": gt}, RESULTS / "comparison.png", met, "Low light · ISO 3200 · −2 EV · 4200 K")

    # ---- 2. robustness sweeps ---------------------------------------------
    sweeps = run_sweeps(test, models, names, size=a.size)
    save_sweeps(sweeps, RESULTS / "robustness.png", "PSNR")
    save_sweeps(sweeps, RESULTS / "robustness_deltae.png", "Delta E")

    # ---- 3. efficiency ----------------------------------------------------
    eff = []
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    ref = models.get("hybrid", next(iter(models.values())))
    set_macs_reference(ref)
    overall = {n: float(np.mean([table[s][n]["PSNR"] for s in table])) for n in names}
    for label, key in (("Learned ISP", "learned"), ("Hybrid ISP", "hybrid"), ("Hybrid-Lite", "hybrid_lite"), ("Hybrid INT8", "hybrid_int8")):
        if key not in models:
            continue
        m = models[key]
        row = {"name": label, "key": key}
        try:
            info = model_info(m)
        except Exception:
            info = {}
        if key == "hybrid_int8":
            from models.optimize import state_dict_bytes
            set_macs_reference(models["hybrid_lite"])
            info = model_info(models["hybrid_lite"]); info["size_mb"] = state_dict_bytes(m) / 1e6
        else:
            set_macs_reference(m)
        b1 = benchmark_model(m, 1.0, name=label)
        b12 = benchmark_model(m, 4.0, name=label)
        bt = benchmark_model(m, 4.0, tile=128, name=label) if key != "hybrid_int8" else {}
        row.update(info)
        # memory: peak extra RSS of a forward pass in a fresh process (in-process sampling is hidden by allocator reuse)
        pm1, pm4 = peak_memory_mb(key, 1.0), peak_memory_mb(key, 4.0)
        pmt = peak_memory_mb(key, 4.0, tile=128) if key != "hybrid_int8" else None
        row.update({"latency_ms": b1["latency_ms"], "latency_4mp_ms": b12["latency_ms"],
                    "peak_memory_mb": pm1 if pm1 is not None else b1["peak_memory_mb"],
                    "peak_memory_4mp_mb": pm4 if pm4 is not None else b12["peak_memory_mb"],
                    "peak_memory_4mp_tiled_mb": pmt if pmt is not None else bt.get("peak_memory_mb"),
                    "gmacs_per_mp": b1.get("gmacs_per_mp"), "threads": b1["threads"],
                    "psnr": overall.get("Hybrid ISP" if key == "hybrid_int8" and False else label, float("nan"))})
        eff.append(row)
        print(row, flush=True)
    # full pipelines at ~1 MP
    # exactly 1.0 MP (1000 x 1000) so that "ms per frame" and "ms per MP" are the same number for every pipeline
    import cv2
    gt = center_crop(next(iter(test.values())), 960).astype(np.float32) / 255
    gt = cv2.resize(gt, (1000, 1000), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    big = simulate_raw(gt, SCENARIOS["Low light"], 3)
    assert abs(big.mosaic.size / 1e6 - 1.0) < 1e-6
    pipe = {}
    for n in names:
        pipe[n] = pipeline_stage_timing(lambda r, n=n: run_pipeline(n, r, models), big, runs=3)
    trad_row = {"name": "Traditional ISP", "key": "traditional", "latency_ms": pipe["Traditional ISP"]["total_ms"] * 1e6 / (big.mosaic.size),
                "psnr": overall["Traditional ISP"], "parameters": 0, "size_mb": 0.0}
    eff.insert(0, trad_row)
    for r in eff:
        if r["name"] in pipe:
            r["pipeline_ms_per_mp"] = pipe[r["name"]]["total_ms"] * 1e6 / big.mosaic.size
    save_efficiency([{**r, "latency_ms": r.get("pipeline_ms_per_mp", r["latency_ms"])} for r in eff if r["name"] in overall or r["name"] in COLORS_KEYS(names)], RESULTS / "efficiency.png")

    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "test_images": sorted(test), "eval_size": a.size, "scenarios": table,
           "sweeps": sweeps, "efficiency": eff, "pipeline_timing_1mp": pipe, "overall_psnr": overall,
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "pipeline_frame_mp": big.mosaic.size / 1e6,
           "lpips_available": bool(any(table[s][n].get("LPIPS") is not None for s in table for n in names)),
           "lpips_note": "LPIPS needs the optional `lpips` package and downloadable AlexNet weights; null when unavailable (GMSD / MS-SSIM are always reported)."}
    (RESULTS / "experiments.json").write_text(json.dumps(out, indent=1, default=float))
    # sample RAW for the dashboard / demos
    gt0 = center_crop(test["sk_chelsea"] if "sk_chelsea" in test else next(iter(test.values())), 256).astype(np.float32) / 255
    raw0 = simulate_raw(gt0, SCENARIOS["Tungsten indoor"], 5)
    np.save(Path("data/sample") / "sample_raw_bayer_uint16.npy", raw_to_dn(raw0))
    from PIL import Image
    Image.fromarray(np.uint8(gt0 * 255)).save("data/sample/clean.png")
    Image.fromarray(np.uint8(np.clip(naive_preview(raw0), 0, 1) * 255)).save("data/sample/degraded.png")
    print("done ->", RESULTS)


def COLORS_KEYS(names):
    return names


if __name__ == "__main__":
    main()
