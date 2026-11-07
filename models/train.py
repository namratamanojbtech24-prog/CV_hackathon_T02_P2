"""Training of the learned / hybrid ISP networks on synthetic RAW pairs (+ knowledge distillation).

Every step draws random patches from the clean-photo pool, runs them through the sensor simulator
(random ISO, exposure, illuminant) and trains the U-Net so that

    compose( classical front-end -> U-Net -> classical back-end )  ≈  clean sRGB     (L1 loss).

For the hybrid model the network corrects the *display-referred* classical result (after CCM + sRGB curve), so the loss
acts directly on the final picture and noise is handled in a perceptually uniform domain.

    python -m models.train --mode hybrid  --base 16 --steps 3000 --out models/checkpoints/hybrid_teacher.pt
    python -m models.train --mode learned --base 16 --steps 3000 --out models/checkpoints/learned_e2e.pt
    python -m models.train --mode hybrid  --base 8  --steps 3000 --distill models/checkpoints/hybrid_teacher.pt \
                           --out models/checkpoints/hybrid_student.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from config import CKPT_DIR, CHECKPOINTS
from isp.learned_isp import CTX_BY_MODE, IN_CH_BY_MODE, build_features, build_model, compose, load_model
from models.unet_isp import ISPUNet, model_info
from utils.datasets import BatchSampler, center_crop, load_pool, split_pool
from utils.degradation import SCENARIOS, simulate_raw

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_sample_fn(mode):
    def fn(gt, raw, rng):
        tr = raw.truth
        a = tr["a_eff"] * float(np.exp(rng.normal(0, 0.15)))          # the noise map is an *estimate* at inference
        b = tr["b_eff"] * float(np.exp(rng.normal(0, 0.25)))
        cast = np.asarray(tr["cast"], np.float64)
        g = 1.0 / cast
        # white-balance gains come from an imperfect AWB: mostly small errors, sometimes large ones
        # Small AWB jitter only: large colour errors are not fixable locally and would swamp the restoration
        # signal (denoise / demosaic) in the L1 loss. Global colour is handled by the classical AWB.
        err = rng.normal(0, 0.02, 2)
        gains = np.array([g[0] * np.exp(err[0]), 1.0, g[2] * np.exp(err[1])], np.float32)
        f = build_features(raw.mosaic * raw.gain, a, b, raw.pattern, mode, gains=gains, gain=raw.gain)
        return {"x": f["x"], "ctx": f["ctx"], "skip": f["skip"], "gt": gt.transpose(2, 0, 1).astype(np.float32)}
    return fn


def _t(batch, key):
    return torch.from_numpy(batch[key])


def make_val_set(val_imgs, mode, n_per=3, size=192, seed=1234):
    rng = np.random.default_rng(seed)
    scen = [SCENARIOS["Tungsten indoor"], SCENARIOS["Low light"], SCENARIOS["Night, extreme"]][:n_per]
    items, fn = [], make_sample_fn(mode)
    for name, img in val_imgs.items():
        gt = center_crop(img, size).astype(np.float32) / 255
        for sc in scen:
            raw = simulate_raw(gt, sc, rng)
            items.append(fn(gt, raw, rng))
    return {k: np.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def evaluate(model, vb, mode):
    model.eval()
    out = compose(model(_t(vb, "x"), _t(vb, "ctx")), _t(vb, "skip"), mode).clamp(0, 1)
    mse = ((out - _t(vb, "gt")) ** 2).mean((1, 2, 3))
    return float((10 * torch.log10(1 / mse)).mean())


def train(mode="hybrid", base=16, steps=3000, batch=12, size=128, lr=1e-3, out=None, distill=None, kd_weight=1.0,
          seed=0, hard=0.7, log_every=100, progress=None, pool=None, verbose=True):
    assert mode in ("hybrid", "learned")
    out = Path(out or (CKPT_DIR / ("hybrid_teacher.pt" if mode == "hybrid" else "learned_e2e.pt")))
    torch.manual_seed(seed)
    pool = pool or load_pool()
    train_imgs, val_imgs, _ = split_pool(pool)
    if not train_imgs:
        raise RuntimeError("No training images found. Run `python download_data.py` or add photos to data/reference/.")
    model = ISPUNet(base=base, in_ch=IN_CH_BY_MODE[mode], ctx_dim=CTX_BY_MODE[mode], zero_init=True)
    teacher = None
    if distill:
        teacher = load_model(distill, use_cache=False)[0].eval()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    vb = make_val_set(val_imgs, mode) if val_imgs else None
    sampler = BatchSampler(train_imgs, make_sample_fn(mode), batch, size, seed, hard)
    hist, best, t0 = [], -1.0, time.time()
    if verbose:
        info = model_info(model)
        print(f"[{mode}] base={base} params={info['parameters']:,} train_imgs={len(train_imgs)} val_imgs={len(val_imgs)}"
              f"{' distill' if teacher else ''}", flush=True)
    try:
        for step in range(1, steps + 1):
            warm = min(1.0, step / 100)
            cos = 0.5 * (1 + math.cos(math.pi * step / steps))
            for g in opt.param_groups:
                g["lr"] = lr * warm * (0.02 + 0.98 * cos)
            b_ = sampler.get()
            x, c, s, gt = (_t(b_, k) for k in ("x", "ctx", "skip", "gt"))
            model.train()
            pred = compose(model(x, c), s, mode)
            loss = (pred - gt).abs().mean()
            if teacher is not None:
                with torch.no_grad():
                    tp = compose(teacher(x, c), s, mode)
                loss = loss + kd_weight * (pred - tp).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            d = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            hist.append(float(loss))
            if step % log_every == 0 or step == steps:
                vp = evaluate(ema, vb, mode) if vb is not None else float("nan")
                if verbose:
                    print(f"step {step:5d}/{steps}  loss {np.mean(hist[-log_every:]):.4f}  val PSNR(EMA) {vp:.2f} dB  "
                          f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                if progress:
                    progress(step, steps, float(np.mean(hist[-log_every:])), vp)
                if vp > best:
                    best = vp
                    _save(ema, out, mode, base, step, hist, vp, sorted(train_imgs), distill)
    finally:
        sampler.close()
    if vb is None:
        _save(ema, out, mode, base, steps, hist, float("nan"), sorted(train_imgs), distill)
    return {"history": hist, "best_val_psnr": best, "path": str(out)}


def _save(model, path, mode, base, step, hist, val, names, distill):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": {"base": base, "in_ch": model.in_ch, "ctx_dim": model.ctx_dim}, "mode": mode,
                "state": {k: v.clone() for k, v in model.state_dict().items()}, "steps": step, "val_psnr": val,
                "loss_history": hist[-2000:], "train_images": names, "distilled_from": str(distill) if distill else None},
               path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "learned"])
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--distill", default=None, help="teacher checkpoint (name or path) for knowledge distillation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = train(a.mode, a.base, a.steps, a.batch, a.size, a.lr, a.out, a.distill, seed=a.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}))
