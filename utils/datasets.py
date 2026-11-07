"""Clean-photo datasets (ground truth) and the on-the-fly RAW training-pair sampler."""
from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from config import ROOT
from utils.degradation import Scenario, sample_scenario, simulate_raw

PUBLIC = ROOT / "data" / "public"
# Images are split by *name* so that no test photo (or near-duplicate of it) is ever seen in training.
TEST_NAMES = ["sk_astronaut", "sk_chelsea", "sk_coffee", "sk_rocket", "sl_flower", "ex_airplane", "cv_baboon",
              "cv_building", "cv_butterfly", "ex_teddy"]
VAL_NAMES = ["cv_lena", "ex_cones"]
MAX_SIDE = 960


def _load(path: Path, max_side=MAX_SIDE):
    im = Image.open(path)
    if im.mode != "RGB":
        if im.mode in ("L", "1", "I;16"):
            return None                       # skip greyscale: colour processing needs colour data
        im = im.convert("RGB")
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1:
        # large camera JPEGs are area-downsampled: it removes compression artefacts and keeps real detail
        im = im.resize((max(8, round(w * s)), max(8, round(h * s))), Image.Resampling.LANCZOS)
    return np.asarray(im, np.uint8)


def load_pool(folders=None) -> Dict[str, np.ndarray]:
    """name -> uint8 RGB image, from data/public and data/reference (user supplied)."""
    folders = folders or [PUBLIC, ROOT / "data" / "reference"]
    pool = {}
    for f in folders:
        for p in sorted(Path(f).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
                a = _load(p)
                if a is not None and min(a.shape[:2]) >= 96:
                    pool[p.stem] = a
    return pool


def split_pool(pool):
    train = {k: v for k, v in pool.items() if k not in TEST_NAMES and k not in VAL_NAMES}
    val = {k: v for k, v in pool.items() if k in VAL_NAMES}
    test = {k: v for k, v in pool.items() if k in TEST_NAMES}
    return train, val, test


def center_crop(img, size):
    h, w = img.shape[:2]
    s = min(size, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def random_patch(rng, img, size):
    """Random scale (0.5..1.0, area-resampled), flip, crop -> float32 sRGB patch."""
    import cv2
    h, w = img.shape[:2]
    s = rng.uniform(0.3, 1.0)
    if min(h, w) * s < size:
        s = min(1.0, size / min(h, w) + 1e-3)
    if s < 0.999:
        img = cv2.resize(img, (max(size, int(w * s)), max(size, int(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    p = img[y:y + size, x:x + size].astype(np.float32) / 255.0
    if rng.random() < 0.5:
        p = p[:, ::-1]
    if rng.random() < 0.5:
        p = p[::-1]
    if rng.random() < 0.5:
        p = p.transpose(1, 0, 2)
    # strong photometric diversity: with a tiny photo pool the network would otherwise memorise image content.
    # Any recoloured image is still a valid "scene", so permute channels and jitter per-channel gains.
    p = p[..., rng.permutation(3)] if rng.random() < 0.7 else p
    p = np.clip(p * rng.uniform(0.8, 1.2, 3).astype(np.float32), 0, 1)
    gam = float(np.exp(rng.normal(0, 0.12)))
    p = np.clip(p, 0, 1) ** gam
    sat = float(rng.uniform(0.75, 1.25))
    lum = p.mean(2, keepdims=True)
    p = np.clip(lum + (p - lum) * sat, 0, 1)
    if rng.random() < 0.3:                                  # vary the amount of fine texture
        import cv2
        p = cv2.GaussianBlur(p, (0, 0), float(rng.uniform(0.4, 1.2)))
    return np.ascontiguousarray(p)


class BatchSampler:
    """Background-thread producer of training batches (numpy dicts)."""

    def __init__(self, pool, make_sample, batch=12, size=128, seed=0, hard=1.0, prefetch=6):
        self.names = list(pool)
        self.pool, self.make_sample = pool, make_sample
        self.batch, self.size, self.hard = batch, size, hard
        self.q = queue.Queue(prefetch)
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, args=(seed,), daemon=True)
        self.th.start()

    def _run(self, seed):
        rng = np.random.default_rng(seed)
        while not self.stop.is_set():
            items = []
            for _ in range(self.batch):
                img = self.pool[self.names[int(rng.integers(len(self.names)))]]
                gt = random_patch(rng, img, self.size)
                sc = sample_scenario(rng, self.hard)
                raw = simulate_raw(gt, sc, rng)
                items.append(self.make_sample(gt, raw, rng))
            batch = {k: np.stack([it[k] for it in items]) for k in items[0]}
            while not self.stop.is_set():
                try:
                    self.q.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    pass

    def get(self):
        return self.q.get()

    def close(self):
        self.stop.set()
