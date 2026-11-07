import numpy as np
import pytest

from isp.raw_processing import (PATTERNS, RawImage, demosaic, mosaic_from_rgb, pack_bayer, unpack_bayer)
from isp.white_balance import estimate_gains
from isp.color_correction import apply_ccm, rgb_to_ycbcr, ycbcr_to_rgb
from isp.denoising import gat, gat_inverse, denoise_bayer
from isp.tone_mapping import tone_map
from isp.sharpening import sharpen
from isp.noise_estimation import estimate_noise_params
from isp.classical_isp import traditional_isp
from utils.degradation import Scenario, simulate_raw, unprocess
from evaluation.metrics import calculate_metrics


def photo(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x / w, y / h, 0.5 + 0.3 * np.sin(x / 7.0)], -1) * 0.8 + 0.1
    return np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_pack_roundtrip(pattern):
    m = np.random.default_rng(0).random((16, 18), dtype=np.float32)
    assert np.array_equal(unpack_bayer(pack_bayer(m, pattern), pattern), m)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_demosaic_beats_bilinear(pattern):
    cam = unprocess(photo(64, 64))
    mos = mosaic_from_rgb(cam, pattern)
    err = lambda d: float(np.mean((d - cam) ** 2))
    assert err(demosaic(mos, pattern, "mhc")) < err(demosaic(mos, pattern, "bilinear"))


def test_gat_inverse():
    x = np.linspace(0, 1, 100)
    assert np.allclose(gat_inverse(gat(x, 1e-3, 1e-4), 1e-3, 1e-4), x, atol=1e-6)


def test_noise_estimator_accuracy():
    gt = photo(192, 192)
    raw = simulate_raw(gt, Scenario(iso=3200, ev_under=0, kelvin=6500), 1)
    a, b = estimate_noise_params(raw.mosaic, raw.pattern)
    assert 0.5 < a / raw.truth["a"] < 2.0


def test_denoise_reduces_noise():
    gt = photo(96, 96)
    raw = simulate_raw(gt, Scenario(iso=6400, ev_under=0, kelvin=6500), 2)
    a, b = raw.truth["a"], raw.truth["b"]
    clean = simulate_raw(gt, Scenario(iso=100, ev_under=0, kelvin=6500), 2, quantize=False).mosaic
    den = denoise_bayer(raw.mosaic, a, b, raw.pattern)
    assert np.mean((den - clean) ** 2) < 0.6 * np.mean((raw.mosaic - clean) ** 2)


def test_white_balance_finds_cast():
    gt = photo(96, 96)
    raw = simulate_raw(gt, Scenario(iso=100, ev_under=0, kelvin=3000), 3)
    rgb = demosaic(raw.mosaic * raw.gain, raw.pattern)
    g = estimate_gains(np.maximum(rgb, 0))
    assert g[0] < 1 < g[2]                       # warm light -> reduce red, boost blue


def test_stage_shapes_and_ranges():
    x = np.random.default_rng(2).random((31, 29, 3), dtype=np.float32)
    for y in (apply_ccm(x), tone_map(x), sharpen(x, 0.5)):
        assert y.shape == x.shape and np.isfinite(y).all()
    assert np.allclose(ycbcr_to_rgb(rgb_to_ycbcr(x)), x, atol=1e-4)


def test_traditional_isp_improves_over_naive():
    from utils.degradation import naive_preview
    gt = photo(96, 96)
    raw = simulate_raw(gt, Scenario(iso=3200, ev_under=2, kelvin=3000), 4)
    out = traditional_isp(raw).rgb
    assert out.shape == gt.shape and 0 <= out.min() and out.max() <= 1
    assert calculate_metrics(out, gt, False)["PSNR"] > calculate_metrics(naive_preview(raw), gt, False)["PSNR"]


def test_odd_sizes():
    raw = RawImage(mosaic=np.random.default_rng(0).random((33, 35), dtype=np.float32) * 0.5, gain=1.0)
    out = traditional_isp(raw).rgb
    assert out.shape[0] % 2 == 0


torch = pytest.importorskip("torch")


def test_unet_shapes_and_tiling():
    import torch
    from models.unet_isp import ISPUNet
    from isp.learned_isp import run_network
    torch.manual_seed(0)
    rng = np.random.default_rng(0)                       # deterministic: random weights make this otherwise flaky
    m = ISPUNet(4, zero_init=False).eval()
    x = rng.random((8, 96, 104), dtype=np.float32)
    c = rng.random(8, dtype=np.float32)
    full = run_network(m, x, c)
    assert full.shape == (3, 192, 208)
    tiled = run_network(m, x, c, tile=64, overlap=24)    # overlap close to the receptive field -> near-identical
    assert np.abs(full - tiled).mean() < 0.05 * np.abs(full).mean()


def test_hybrid_zero_strength_equals_classical_backend():
    from models.unet_isp import ISPUNet
    from isp.learned_isp import hybrid_isp
    raw = simulate_raw(photo(64, 64), Scenario(iso=800, ev_under=1, kelvin=5000), 5)
    out = hybrid_isp(raw, model=ISPUNet(4, in_ch=12, ctx_dim=10).eval(), strength=0.0)
    assert out.rgb.shape == (64, 64, 3) and not out.info["used_network"]
