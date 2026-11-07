"""Physically motivated sensor simulator: clean sRGB photo  ->  degraded Bayer RAW.

Pipeline ("unprocessing", Brooks et al. CVPR'19, extended with illuminant & exposure):

    sRGB (ground truth) -> inverse gamma -> inverse CCM -> camera RGB
        -> scene illuminant colour cast  -> exposure (dim light)
        -> Bayer mosaic -> Poisson-Gaussian noise (shot + read) -> 12-bit ADC

Noise model: var(x) = a * x + b   (x = noise-free signal in [0,1])
    a = shot-noise slope (grows with ISO / analogue gain),  b = read-noise variance.
The camera then applies a digital exposure-compensation gain 1/exposure, which
also amplifies the noise: the realistic low-light failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import BAYER_PATTERN, BIT_DEPTH, BLACK_DN, SENSOR_CCM, WHITE_DN
from isp.raw_processing import RawImage, crop_even, mosaic_from_rgb, normalize_raw

_INV_CCM = np.linalg.inv(SENSOR_CCM).astype(np.float32)


# ---------------------------------------------------------------- transfer curves
def srgb_to_linear(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


# ---------------------------------------------------------------- scenario parameters
def iso_to_shot(iso: float) -> float:
    """ISO-equivalent -> shot-noise slope a (ISO 100 -> 8e-5 ... ISO 25600 -> 2e-2)."""
    return float(np.clip(8e-5 * iso / 100.0, 1e-5, 5e-2))


def read_from_shot(a: float, jitter: float = 0.0) -> float:
    """Typical read-noise variance for a given shot-noise slope (Brooks et al. fit)."""
    return float(np.exp(2.18 * np.log(a) + 1.2 + jitter))


def kelvin_to_cast(kelvin: float):
    """Colour temperature of the scene light -> (R, G, B) multipliers the sensor sees (6500 K = neutral)."""
    t = np.log(kelvin / 6500.0)
    return (float(np.exp(-0.55 * t)), 1.0, float(np.exp(0.75 * t)))


@dataclass
class Scenario:
    iso: float = 800.0                 # noise level (ISO-equivalent)
    ev_under: float = 0.0              # stops of under-exposure (0 = well lit, 3 = very dark)
    kelvin: float = 6500.0             # scene illuminant colour temperature
    read_jitter: float = 0.0

    @property
    def shot(self):
        return iso_to_shot(self.iso)

    @property
    def read(self):
        return read_from_shot(self.shot, self.read_jitter)

    @property
    def exposure(self):
        return float(2.0 ** (-self.ev_under))

    @property
    def cast(self):
        return kelvin_to_cast(self.kelvin)

    def label(self):
        return f"ISO {self.iso:.0f} · -{self.ev_under:.1f} EV · {self.kelvin:.0f} K"


SCENARIOS = {
    "Clean":            Scenario(iso=100, ev_under=0.0, kelvin=6500),
    "Daylight, ISO 800": Scenario(iso=800, ev_under=0.0, kelvin=6500),
    "Tungsten indoor":  Scenario(iso=1600, ev_under=1.0, kelvin=2900),
    "Overcast shade":   Scenario(iso=1600, ev_under=0.5, kelvin=8500),
    "Low light":        Scenario(iso=3200, ev_under=2.0, kelvin=4200),
    "Night, extreme":   Scenario(iso=6400, ev_under=3.0, kelvin=3200),
}


def sample_scenario(rng: np.random.Generator, hard: float = 1.0) -> Scenario:
    """Random scenario for training. `hard` scales the log-ISO and EV ranges."""
    iso = float(np.exp(rng.uniform(np.log(100), np.log(100 + 12700 * hard))))
    ev = float(rng.uniform(0, 3.0 * hard)) if rng.random() < 0.8 else 0.0
    kel = float(np.exp(rng.uniform(np.log(2500), np.log(9500))))
    return Scenario(iso=iso, ev_under=ev, kelvin=kel, read_jitter=float(rng.normal(0, 0.26)))


# ---------------------------------------------------------------- core simulator
def unprocess(srgb) -> np.ndarray:
    """Ground-truth sRGB -> linear camera RGB under neutral light."""
    lin = srgb_to_linear(srgb)
    return np.clip(lin @ _INV_CCM.T, 0, None).astype(np.float32)


def simulate_raw(srgb, scenario: Optional[Scenario] = None, rng=None, pattern=BAYER_PATTERN,
                 quantize=True) -> RawImage:
    """Create a degraded Bayer RAW frame from a clean sRGB image (H, W, 3) in [0, 1]."""
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    sc = scenario or Scenario()
    cam = unprocess(crop_even(np.asarray(srgb, np.float32)))
    cast = np.asarray(sc.cast, np.float32)
    # headroom: scale so that the strongest channel gain does not push white straight into clipping
    exposure = sc.exposure / max(1.0, float(cast.max()))
    scene = cam * cast * exposure                         # photons reaching each photosite (normalised)
    mosaic = mosaic_from_rgb(scene, pattern)
    a, b = sc.shot, sc.read
    var = np.maximum(a * mosaic + b, 0)
    noisy = mosaic + rng.standard_normal(mosaic.shape).astype(np.float32) * np.sqrt(var, dtype=np.float32)
    noisy = np.minimum(noisy, 1.0)                        # sensor saturation
    if quantize:
        dn = np.round(noisy * (WHITE_DN - BLACK_DN) + BLACK_DN)
        dn = np.clip(dn, 0, WHITE_DN)
        noisy = normalize_raw(dn)
    gain = 1.0 / exposure
    # Noise parameters in the *amplified* domain are known only to a calibrated camera; we expose them
    # for the optional "metadata" mode but pipelines estimate them blind by default.
    raw = RawImage(mosaic=noisy.astype(np.float32), pattern=pattern, gain=float(gain))
    raw.truth = dict(a=a, b=b, a_eff=a * gain, b_eff=b * gain ** 2, cast=tuple(float(c) for c in cast),
                     exposure=exposure, scenario=sc.label(), kelvin=sc.kelvin, iso=sc.iso, ev_under=sc.ev_under)
    return raw


def raw_to_dn(raw: RawImage) -> np.ndarray:
    """Normalised mosaic -> uint16 digital numbers (for saving as a RAW-like file)."""
    return np.clip(np.round(raw.mosaic * (WHITE_DN - BLACK_DN) + BLACK_DN), 0, WHITE_DN).astype(np.uint16)


# ---------------------------------------------------------------- back-compat helper
def degrade(clean, seed=0, scenario: Optional[Scenario] = None):
    """Legacy helper: clean sRGB -> a *naively* processed degraded sRGB preview of the RAW frame."""
    raw = simulate_raw(clean, scenario or sample_scenario(np.random.default_rng(seed)), rng=seed)
    return naive_preview(raw)


def naive_preview(raw: RawImage) -> np.ndarray:
    """What you would see without any ISP intelligence: bilinear demosaic, no WB, plain gamma, auto-gain."""
    from isp.raw_processing import demosaic
    rgb = demosaic(raw.mosaic * float(raw.gain or 1.0), raw.pattern, "bilinear")
    return linear_to_srgb(np.clip(rgb, 0, 1))
