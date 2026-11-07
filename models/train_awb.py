"""Train the learned white-balance regressor (a small MLP on illuminant-evidence features).

Training pairs are unlimited: any photo x random illuminant x random noise/exposure -> true log gains.
Takes ~2 minutes on a laptop CPU.   python -m models.train_awb
"""
import time

import numpy as np

from config import CKPT_DIR
from isp.classical_isp import prepare
from isp.raw_processing import demosaic
from isp.white_balance import gain_features
from utils.datasets import load_pool, random_patch, split_pool
from utils.degradation import sample_scenario, simulate_raw


def make_dataset(imgs, n, seed):
    rng = np.random.default_rng(seed)
    names = list(imgs)
    X, Y = [], []
    for _ in range(n):
        img = imgs[names[int(rng.integers(len(names)))]]
        size = int(rng.choice([128, 192, 256, 320]))
        size = min(size, min(img.shape[:2]))
        gt = random_patch(rng, img, size)
        raw = simulate_raw(gt, sample_scenario(rng), rng)
        m, a, b, g = prepare(raw)
        rgb = demosaic(m, raw.pattern, "mhc")
        X.append(gain_features(np.maximum(rgb, 0)))
        Y.append(np.log(1.0 / np.asarray(raw.truth["cast"]))[[0, 2]])
    return np.array(X), np.array(Y)


def main(n_train=6000, seed=0):
    from sklearn.neural_network import MLPRegressor
    pool = load_pool()
    train, val, test = split_pool(pool)
    t0 = time.time()
    X, Y = make_dataset(train, n_train, seed)
    Xv, Yv = make_dataset({**val}, 600, seed + 1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    mlp = MLPRegressor(hidden_layer_sizes=(48, 48), alpha=1e-2, max_iter=600, early_stopping=True, random_state=seed, learning_rate_init=2e-3)
    mlp.fit((X - mu) / sd, Y)
    pred = mlp.predict((Xv - mu) / sd)
    rms = lambda e: float(np.sqrt((e ** 2).mean()))
    base = np.stack([Xv[:, :2][:, 0], Xv[:, :2][:, 1]], 1)       # gray-world features are the first 2 columns
    print(f"val rms log-gain error: learned {rms(pred - Yv):.3f}  gray-world {rms(base - Yv):.3f}  ({time.time() - t0:.0f}s)")
    out = CKPT_DIR / "learned_awb.npz"
    arrs = {"mu": mu, "sd": sd, "n": len(mlp.coefs_)}
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrs[f"W{i}"], arrs[f"b{i}"] = W, b
    np.savez(out, **arrs)
    print("saved", out)


if __name__ == "__main__":
    main()
