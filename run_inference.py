"""Command-line RAW -> RGB inference.

    python run_inference.py --input photo.jpg --scenario "Low light"            # simulate a degraded RAW from a clean photo
    python run_inference.py --input frame.dng                                   # real RAW (needs `pip install rawpy`)
    python run_inference.py --input frame.npy --pipeline "Hybrid INT8"          # 2-D Bayer mosaic as uint16 DN or float
    python run_inference.py --input photo.jpg --pipeline all --output out/      # every pipeline + metrics table

Pipelines: "Traditional ISP", "Learned ISP", "Hybrid ISP", "Hybrid-Lite", "Hybrid INT8", or "all".
When the input is a clean photo the result is scored against it (PSNR, SSIM, Delta E, MS-SSIM, GMSD, LPIPS if installed).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from config import CHECKPOINTS
from utils.degradation import SCENARIOS, naive_preview, simulate_raw
from utils.image_utils import save_image

ALL = ["Traditional ISP", "Learned ISP", "Hybrid ISP", "Hybrid-Lite", "Hybrid INT8"]
KEY = {"Learned ISP": "learned", "Hybrid ISP": "hybrid", "Hybrid-Lite": "hybrid_lite", "Hybrid INT8": "hybrid_int8"}


def load_input(path: Path, scenario: str, size: int, seed: int):
    """Return (RawImage, clean reference or None)."""
    ext = path.suffix.lower()
    if ext in (".dng", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2"):
        from isp.raw_processing import load_dng
        return load_dng(path), None
    if ext == ".npy":
        a = np.load(path)
        if a.ndim == 2:
            from isp.raw_processing import load_raw_array
            return load_raw_array(a), None
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
    if size:
        from utils.datasets import center_crop
        img = center_crop((img * 255).astype(np.uint8), size).astype(np.float32) / 255
    if scenario not in SCENARIOS:
        sys.exit(f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}")
    return simulate_raw(img, SCENARIOS[scenario], seed), img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--pipeline", default="Hybrid ISP", help='one of %s, or "all"' % ALL)
    ap.add_argument("--scenario", default="Low light", help="used when --input is a clean photo: " + ", ".join(SCENARIOS))
    ap.add_argument("--size", type=int, default=384, help="centre-crop size for photos (0 = full frame)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("outputs"), help="output folder")
    a = ap.parse_args()

    from evaluation.comparison import run_pipeline
    from evaluation.metrics import calculate_metrics
    from isp.learned_isp import load_model

    names = ALL if a.pipeline.lower() == "all" else [a.pipeline]
    for n in names:
        if n not in ALL:
            sys.exit(f"Unknown pipeline '{n}'. Choose from {ALL} or 'all'.")
    models = {}
    for n in names:
        k = KEY.get(n)
        if k and CHECKPOINTS[k].exists():
            models[k] = load_model(k)[0]
        elif k:
            sys.exit(f"Checkpoint for '{n}' not found ({CHECKPOINTS[k]}). Run run_experiments.py to create it.")

    raw, ref = load_input(a.input, a.scenario, a.size, a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.input.stem
    save_image(a.output / f"{stem}_input_naive.png", naive_preview(raw))
    if ref is not None:
        save_image(a.output / f"{stem}_reference.png", ref)
    print(f"{'Pipeline':16s} {'ms':>8s}" + ("   PSNR   SSIM  dE2000  LPIPS" if ref is not None else ""))
    for n in names:
        run_pipeline(n, raw, models)                      # warm-up (TorchScript/INT8 compile on first call)
        res = run_pipeline(n, raw, models)
        out_path = a.output / f"{stem}_{n.lower().replace(' ', '_')}.png"
        save_image(out_path, res.rgb)
        line = f"{n:16s} {res.info.get('total_ms', float('nan')):8.1f}"
        if ref is not None:
            m = calculate_metrics(res.rgb, ref, lpips_on=True)
            lp = "  n/a" if m.get("LPIPS") is None else f"{m['LPIPS']:6.3f}"
            line += f"  {m['PSNR']:5.2f}  {m['SSIM']:.3f}  {m['Delta E']:6.2f}  {lp}"
        print(line)
    print("saved to", a.output.resolve())


if __name__ == "__main__":
    main()
