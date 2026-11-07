"""Automatic white balance on linear camera-RGB statistics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _prep(rgb, factor=4):
    """Block-average (cuts noise) and drop clipped / near-black pixels."""
    a = np.asarray(rgb, np.float32)
    h, w = a.shape[:2]
    f = max(1, min(factor, h // 8, w // 8))
    a = a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f, 3).mean((1, 3))
    lum = a.mean(2)
    hi = np.percentile(lum, 99.0)
    lo = max(np.percentile(lum, 5.0), 1e-4)
    keep = (lum > lo) & (lum < max(0.98 * a.max(), hi))
    if keep.sum() < 16:
        keep = np.ones_like(keep)
    return a, keep


def gains_gray_world(rgb):
    a, keep = _prep(rgb)
    mean = a[keep].mean(0)
    return _norm(mean[1] / np.maximum(mean, 1e-6))


def gains_shades_of_gray(rgb, p=6):
    a, keep = _prep(rgb)
    mink = (np.maximum(a[keep], 0) ** p).mean(0) ** (1 / p)
    return _norm(mink[1] / np.maximum(mink, 1e-6))


def gains_gray_edge(rgb, p=1):
    a, keep = _prep(rgb, 2)
    gy, gx = np.gradient(uniform_filter(a, (3, 3, 1)), axis=(0, 1))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = (mag[keep] ** p).mean(0) ** (1 / p)
    return _norm(m[1] / np.maximum(m, 1e-6))


def gains_white_patch(rgb, q=99.0):
    a, _ = _prep(rgb)
    pk = np.percentile(a.reshape(-1, 3), q, axis=0)
    return _norm(pk[1] / np.maximum(pk, 1e-6))


def _norm(g):
    g = np.clip(np.asarray(g, np.float32), 0.25, 4.0)
    return g / g[1]


def project_to_locus(g, prior_strength=0.15):
    """Constrain gains to the physical illuminant locus (daylight/Planckian: R and B gains move in
    opposite directions with colour temperature). Returns the nearest locus point, softly shrunk
    toward neutral so that ambiguous scenes do not produce extreme casts."""
    lr, lb = np.log(g[0]), np.log(g[2])
    # locus: lr = -k * lb  with k = 0.55/0.75 ; nearest point by orthogonal projection onto direction d
    d = np.array([-0.55, 0.75])
    d = d / np.linalg.norm(d)
    t = np.array([lr, lb]) @ d
    t = np.clip(t, -1.15, 0.85) / (1.0 + prior_strength)
    lr, lb = d * t
    return _norm(np.array([np.exp(lr), 1.0, np.exp(lb)], np.float32))


def estimate_gains(rgb, method="adaptive"):
    """Per-channel gains (R, G=1, B) from linear camera RGB."""
    if method == "none":
        return np.ones(3, np.float32)
    if method == "gray-world":
        return gains_gray_world(rgb)
    if method == "shades-of-gray":
        return gains_shades_of_gray(rgb)
    if method == "gray-edge":
        return gains_gray_edge(rgb)
    if method == "white-patch":
        return gains_white_patch(rgb)
    if method == "learned":
        return learned_gains(rgb)
    if method == "blend":       # classical consensus averaged (log domain) with the learned regressor
        ga, gl = estimate_gains(rgb, "adaptive"), learned_gains(rgb)
        return _norm(np.exp(0.5 * (np.log(ga) + np.log(gl))))
    if method == "adaptive":
        # Robust consensus of three complementary illuminant estimators (geometric mean in log space)
        # weighted by how reliable each is: gray-world needs colourful scenes, gray-edge needs texture,
        # shades-of-gray sits in between. White-patch only when a bright, unclipped patch exists.
        cands = [gains_gray_world(rgb), gains_shades_of_gray(rgb), gains_gray_edge(rgb)]
        logs = np.log(np.stack(cands))
        med = np.median(logs, axis=0)
        w = np.exp(-((logs - med) ** 2).sum(1) / 0.08)         # down-weight outlier estimators
        g = np.exp((w[:, None] * logs).sum(0) / w.sum())
        return project_to_locus(_norm(g))
    raise ValueError(f"Unknown white-balance method '{method}'.")


def apply_gains(rgb, gains):
    return np.asarray(rgb, np.float32) * np.asarray(gains, np.float32).reshape(1, 1, 3)


def apply_white_balance(rgb, method="adaptive", gains=None):
    g = np.asarray(gains, np.float32) if gains is not None else estimate_gains(rgb, method)
    return apply_gains(rgb, g), g


# --------------------------------------------------------------------------- learned AWB
def gain_features(rgb):
    """16 illuminant-evidence features from linear (noisy) camera RGB: the four classical estimators'
    log-gains, chroma percentiles and two scene statistics. Used by the learned white-balance regressor."""
    a, keep = _prep(rgb)
    est = [gains_gray_world(rgb), gains_shades_of_gray(rgb), gains_gray_edge(rgb), gains_white_patch(rgb)]
    f = [np.log(g[[0, 2]]) for g in est]
    px = np.maximum(a[keep], 1e-4)
    lc = np.stack([np.log(px[:, 0] / px[:, 1]), np.log(px[:, 2] / px[:, 1])], 1)
    for q in (10, 50, 90):
        f.append(np.percentile(lc, q, axis=0))
    lum = px.mean(1)
    f.append(np.array([np.log(lum.mean()), np.abs(lc).mean()]))
    return np.concatenate(f).astype(np.float32)


class LearnedAWB:
    """Tiny MLP (numpy inference) mapping gain_features -> log gains (R, B). Trained by models/train_awb.py."""
    _cache = {}

    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd = z["mu"], z["sd"]
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n"]))]

    @classmethod
    def load(cls, path=None):
        from config import CKPT_DIR
        path = path or (CKPT_DIR / "learned_awb.npz")
        if not path.exists():
            return None
        if str(path) not in cls._cache:
            cls._cache[str(path)] = cls(path)
        return cls._cache[str(path)]

    def predict(self, feats):
        x = (feats - self.mu) / self.sd
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return x


def learned_gains(rgb):
    m = LearnedAWB.load()
    if m is None:
        return estimate_gains(rgb, "adaptive")
    lg = m.predict(gain_features(rgb))
    return _norm(np.array([np.exp(lg[0]), 1.0, np.exp(lg[1])], np.float32))
