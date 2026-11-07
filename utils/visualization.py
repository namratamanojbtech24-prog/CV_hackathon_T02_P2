"""Static result figures (matplotlib) in the dashboard's dark, minimal visual identity."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.theme import ACCENT, COLORS, INK, INK2, LINE, MPL_DASH, DASH, MUTED, PANEL, SURFACE


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": LINE, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": LINE, "grid.linewidth": 0.7, "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK, "axes.titlelocation": "left",
    })


def _u8(img):
    return np.uint8(np.clip(img, 0, 1) * 255 + 0.5)


def save_comparison(images: dict, path, metrics: dict | None = None, title=None):
    """Row of images with bold names and PSNR chips. images: name -> HxWx3 float."""
    _style()
    n = len(images)
    h, w = next(iter(images.values())).shape[:2]
    fig_w = 3.4 * n
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.4 * h / w + 0.9))
    axes = np.atleast_1d(axes)
    for ax, (name, im) in zip(axes, images.items()):
        ax.imshow(_u8(im))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        col = COLORS.get(name, INK)
        ax.set_title(name.replace(" ISP", ""), color=col, fontsize=13, pad=8)
        if metrics and name in metrics and metrics[name].get("PSNR") is not None:
            m = metrics[name]
            ax.text(0.02, 0.03, f"{m['PSNR']:.1f} dB  ·  ΔE {m['Delta E']:.1f}", transform=ax.transAxes, color=INK,
                    fontsize=10, fontweight="bold", bbox=dict(facecolor=SURFACE, alpha=0.78, edgecolor="none", pad=4))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sweeps(sweeps: dict, path, metric="PSNR"):
    _style()
    keys = list(sweeps)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        sw = sweeps[k]
        xs = np.arange(len(sw["x"]))
        for name, ser in sw["series"].items():
            ax.plot(xs, ser[metric], MPL_DASH[DASH.get(name, "solid")], color=COLORS.get(name, MUTED),
                    lw=3.0 if name == "Hybrid ISP" else 2.0, marker="o", ms=6, mec=SURFACE, mew=1.5, label=name)
        ax.set_xticks(xs); ax.set_xticklabels(sw["xlabels"], fontsize=9)
        ax.set_title(sw["label"], fontsize=12)
        ax.grid(axis="y")
    axes[0].set_ylabel(metric + (" (dB)" if metric == "PSNR" else ""))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_efficiency(rows: list, path):
    """rows: dicts with name, parameters, size_mb, latency_ms, psnr."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    names = [r["name"] for r in rows]
    cols = [COLORS.get(n, MUTED) for n in names]
    for ax, key, ttl, fmt in ((axes[0], "size_mb", "Model size (MB)", "{:.2f}"),
                              (axes[1], "latency_ms", "Latency per megapixel (ms)", "{:.0f}")):
        vals = [r[key] for r in rows]
        bars = ax.barh(names, vals, color=cols, height=0.62)
        ax.invert_yaxis()
        ax.set_title(ttl)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.015, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.18)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    ax = axes[2]
    for r in rows:
        ax.scatter(r["latency_ms"], r["psnr"], s=180, color=COLORS.get(r["name"], MUTED), ec=SURFACE, lw=2, zorder=3)
        ax.annotate(r["name"].replace(" ISP", ""), (r["latency_ms"], r["psnr"]), xytext=(8, 7), textcoords="offset points", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("Latency per megapixel (ms)  →  slower")
    ax.set_ylabel("PSNR (dB)  ↑  better")
    ax.set_title("Quality vs speed")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
