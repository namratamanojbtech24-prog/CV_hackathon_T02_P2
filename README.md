# Hybrid ISP: an intelligent, computationally efficient smartphone ISP (P2)

RAW → RGB with three pipelines compared on the same degraded Bayer frames:

| Pipeline | What it is |
|---|---|
| **Traditional ISP** | Black level → exposure gain → blind noise estimate → GAT + guided-NLM Bayer denoise → Malvar-He-Cutler demosaic → consensus AWB (locus-projected) → CCM → sRGB curve → chroma denoise → luma sharpen |
| **Learned ISP** | One U-Net: packed RAW + noise map → sRGB (end-to-end) |
| **Hybrid ISP** | Classical front-end builds a display-referred base; a small FiLM-conditioned U-Net predicts a residual; classical sharpening on top |
| **Hybrid-Lite / INT8** | Hybrid student distilled from the teacher (4× fewer params), then INT8 post-training quantisation (exported as TorchScript) |

## Quickstart

```bash
pip install -r requirements.txt
python download_data.py                  # public sample photos (skip if data/public is populated)
streamlit run app.py                     # dashboard
python run_inference.py --input data/public/sk_chelsea.png --pipeline all --scenario "Low light"
python run_experiments.py --skip-train   # regenerate every number / figure in results/ (a few minutes, CPU)
python run_experiments.py                # same, but (re)trains missing checkpoints first (~1 h on CPU)
pytest -q                                # 17 tests
```
Windows: use `py -m pip ...`, `py -m streamlit run app.py`, `py run_inference.py ...`. Pretrained checkpoints ship in `models/checkpoints/`.

`run_inference.py` accepts a clean photo (a degraded Bayer RAW is simulated for the chosen scenario and the output is scored against the photo), a 2-D `.npy` Bayer mosaic, or a real `.dng` (`pip install rawpy`).

**LPIPS (optional perceptual metric).** `pip install lpips`, then re-run `python run_experiments.py --skip-train`: LPIPS is computed automatically and appears in `results/experiments.json`, the dashboard table and `run_inference.py`. It needs one download of the AlexNet backbone, which was not possible in the environment that produced the saved results, so the reported perceptual metrics are **MS-SSIM and GMSD**.

## Measured results (10 held-out photos × 6 scenarios, 256 px crops)

Mean PSNR over all scenarios:

| Pipeline | PSNR | Params | Size (FP32) |
|---|---|---|---|
| Traditional | 24.32 dB | – | – |
| Learned | 22.50 dB | 565 k | 2.26 MB |
| Hybrid | **24.46 dB** | 566 k | 2.26 MB |
| Hybrid-Lite (distilled) | 24.44 dB | 144 k | 0.58 MB |
| Hybrid INT8 | 24.42 dB | 144 k | 0.29 MB |

Per-scenario PSNR / SSIM / ΔE2000 / MS-SSIM / GMSD are in the dashboard (Benchmarks tab) and `results/experiments.json`.

### Efficiency (1.0 MP frame, desktop CPU, 2 threads)

"Model" is the network alone; "full pipeline" is every stage (classical + network) for one 1.0 MP RAW frame. Both are ms per megapixel, measured on the same frame, so the columns are directly comparable. Run-to-run variation on this shared CPU is roughly ±10%.

| Pipeline | Model ms/MP | **Full pipeline ms/MP** | Peak RAM 1 MP | Peak RAM 4 MP | Peak RAM 4 MP, tiled |
|---|---|---|---|---|---|
| Traditional | – | **348** | – | – | – |
| Learned | 154 | **251** | 196 MB | 733 MB | 72 MB |
| Hybrid | 172 | **762** | 181 MB | 763 MB | 73 MB |
| Hybrid-Lite | 84 | **666** | 100 MB | 418 MB | 72 MB |
| Hybrid INT8 | 25 | **645** | 28 MB | 83 MB | – |

Peak RAM is the extra resident memory of one forward pass, measured in a fresh process (Linux kernel high-water mark). Tiled inference with a shared global context vector bounds it at about 72 MB for any image size.

Stage breakdown, 1.0 MP (ms): Traditional = prepare 49 · denoise 60 · demosaic 53 · AWB 89 · CCM 4 · tone 19 · sharpen 64. Hybrid = prepare 55 · **AWB 280** · **classical front-end 208** · network 178 · backend 35.

## Honest findings and limitations

* **The hybrid is slower end to end than the classical ISP** (762 vs 348 ms/MP; 645 ms/MP with INT8). Distillation and INT8 cut the *network* from 172 to 25 ms/MP (7×, 0.29 MB), but the hybrid pipeline's cost is dominated by classical work that it does in addition to the traditional path: a full-resolution demosaic + learned-AWB regressor (≈280 ms) and a second Bayer-domain denoise to build the network's input and base image (≈208 ms). Computing AWB statistics on 2×2-binned planes removes most of the AWB cost but, with the current learned-AWB weights, lost 0.4–0.5 dB on neutral-light scenes, so it was *not* adopted. Retraining the AWB regressor on binned statistics is the obvious next step. The learned ISP is faster than the hybrid but 1.8 dB worse.
* **The hybrid's win is small** (+0.14 dB mean) and it **loses on clean and daylight frames** (−0.4 dB). It wins where conditions are hard: Tungsten +0.7 dB, Low light +0.3, Night +0.4, with lower ΔE. Diagnosis (test-set diagnostic, *not* used to tune anything):

  | PSNR (dB) | Traditional | Hybrid | Hybrid, network skipped | Hybrid with classical AWB only |
  |---|---|---|---|---|
  | Clean | 25.35 | 24.91 | 24.72 | 25.69 |
  | Daylight ISO 800 | 25.26 | 24.93 | 24.77 | 25.65 |
  | Tungsten indoor | 25.54 | 26.21 | 25.90 | 25.39 |
  | Low light | 24.43 | 24.75 | 24.99 | 24.10 |

  Skipping the network on clean frames (`skip_threshold`, used by the dashboard's compute-saving option) saves ≈25% of the hybrid's time but does **not** recover the loss: the network adds ≈ +0.2 dB there. The loss comes from the learned white-balance blend, which is better under colour casts and worse when the light is neutral. A gate on the noise level would fit these test scenarios, but on validation photos with random casts the learned/blend AWB is better, so it was not applied to avoid tuning to the test set. A larger, more diverse training set for the AWB regressor is the proper fix.
* The pure learned ISP degrades fastest under harsh conditions (Night: 16.6 dB vs 21.1 dB classical), which is the argument for the hybrid.
* Only ~19 training photos are available offline, so the network mostly refines its classical base. Training on DIV2K / SIDD / MIT-Adobe FiveK or real RAW pairs is the main route to larger gains.
* Data are *simulated* RAW (inverse ISP + Poisson-Gaussian noise + 12-bit ADC); real sensors will differ.
* **No backlit / HDR test scenes** are included, so nothing here claims bright-region preservation or local tone mapping; the tone stage is a global sRGB curve with optional highlight roll-off.
* Latency is desktop CPU and should be re-measured on the target phone. INT8 uses FX post-training quantisation, exported as TorchScript.

## Layout
`isp/` classical stages + learned/hybrid inference · `models/` U-Net, training, distillation, quantisation · `evaluation/` metrics, benchmark, comparison, robustness · `utils/` simulator, datasets, UI · `app.py` dashboard · `run_inference.py` CLI · `run_experiments.py` end-to-end results.
