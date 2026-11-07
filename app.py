"""Hybrid Smartphone ISP — dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from config import CHECKPOINTS, RESULTS
from utils.theme import ACCENT, COLORS, DASH, INK, INK2, LINE, MUTED
from utils.ui import CSS, compare_html, kpi, plotly_layout

st.set_page_config(page_title="Hybrid ISP", page_icon="◐", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

PIPES_ALL = ["Traditional ISP", "Learned ISP", "Hybrid ISP", "Hybrid-Lite", "Hybrid INT8"]
CKPT_KEY = {"Learned ISP": "learned", "Hybrid ISP": "hybrid", "Hybrid-Lite": "hybrid_lite", "Hybrid INT8": "hybrid_int8"}


def show_img(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:                                  # older Streamlit
        st.image(img, caption=caption, use_container_width=True)


def stretch_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def plot(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_models():
    import torch
    from isp.learned_isp import load_model
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = {}
    for name, key in CKPT_KEY.items():
        if CHECKPOINTS[key].exists():
            try:
                out[name] = load_model(key)[0]
            except Exception as e:                       # noqa
                st.warning(f"Could not load {key}: {e}")
    return out


@st.cache_data(show_spinner=False)
def get_pool():
    from utils.datasets import load_pool, split_pool
    pool = load_pool()
    _, _, test = split_pool(pool)
    return test or pool


@st.cache_data(show_spinner=False)
def get_results():
    p = RESULTS / "experiments.json"
    return json.loads(p.read_text()) if p.exists() else None


def pipe_models(models):
    return {CKPT_KEY[k]: v for k, v in models.items()}


# --------------------------------------------------------------------------- header
models = get_models()
res = get_results()
st.markdown(
    """<div class="hero"><div><div class="tag">RAW → RGB · classical + learned</div>
    <h1>Hybrid <em>ISP</em></h1>
    <p>A smartphone image-signal-processing pipeline that keeps the interpretable classical stages and hands
    only denoising and demosaic refinement to a tiny neural network.</p></div>
    <div style="text-align:right"><span class="chip">12-bit Bayer RAW</span><span class="chip">U-Net · &lt;0.6 M params</span></div></div>""",
    unsafe_allow_html=True)
if "Hybrid ISP" not in models:
    st.warning("No trained checkpoints yet: Learned / Hybrid pipelines are disabled. Open **Train** (or run "
               "`python run_experiments.py`) to create them. The classical ISP works without them.")

tab_studio, tab_bench, tab_eff, tab_pipe, tab_train = st.tabs(["Studio", "Benchmarks", "Efficiency", "Pipeline", "Train"])

# =========================================================================== STUDIO
with tab_studio:
    left, right = st.columns([1, 2.7], gap="large")
    with left:
        st.markdown('<div class="tag">1 · Source</div>', unsafe_allow_html=True)
        source = st.radio("Source", ["Sample photo", "Upload photo", "Upload RAW"], label_visibility="collapsed", horizontal=False)
        gt = raw_in = None
        if source == "Sample photo":
            pool = get_pool()
            name = st.selectbox("Held-out photo", sorted(pool), index=sorted(pool).index("sk_chelsea") if "sk_chelsea" in pool else 0)
            size = st.select_slider("Crop", [256, 384, 512], value=384)
            from utils.datasets import center_crop
            gt = center_crop(pool[name], size).astype(np.float32) / 255
        elif source == "Upload photo":
            up = st.file_uploader("Photo (treated as the clean reference)", type=["png", "jpg", "jpeg", "tif", "tiff", "webp"])
            if up:
                im = Image.open(up).convert("RGB")
                im.thumbnail((640, 640))
                gt = np.asarray(im, np.float32) / 255
        else:
            up = st.file_uploader("Bayer RAW mosaic (.npy / .tif / .png 16-bit, or .dng)", type=["npy", "tif", "tiff", "png", "dng"])
            pat = st.selectbox("Bayer pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
            c1, c2 = st.columns(2)
            blk = c1.number_input("Black level", 0, 4095, 256)
            wht = c2.number_input("White level", 1, 65535, 4095)
            if up:
                try:
                    import tempfile
                    from isp.raw_processing import load_dng, load_raw_array
                    suffix = Path(up.name).suffix.lower()
                    if suffix == ".dng":
                        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as f:
                            f.write(up.getvalue())
                        raw_in = load_dng(f.name)
                    elif suffix == ".npy":
                        raw_in = load_raw_array(np.load(io.BytesIO(up.getvalue())), blk, wht, pat)
                    else:
                        raw_in = load_raw_array(np.asarray(Image.open(io.BytesIO(up.getvalue()))), blk, wht, pat)
                except Exception as e:
                    st.error(f"Could not read RAW: {e}")
            elif (BASE / "data/sample/sample_raw_bayer_uint16.npy").exists():
                if st.button("Use bundled sample RAW"):
                    from isp.raw_processing import load_raw_array
                    st.session_state["bundled_raw"] = load_raw_array(np.load(BASE / "data/sample/sample_raw_bayer_uint16.npy"))
                raw_in = st.session_state.get("bundled_raw")

        if source != "Upload RAW":
            st.markdown('<div class="tag" style="margin-top:1.2rem">2 · Capture conditions</div>', unsafe_allow_html=True)
            from utils.degradation import SCENARIOS
            preset = st.selectbox("Preset", list(SCENARIOS) + ["Custom"], index=4, label_visibility="collapsed")
            sc0 = SCENARIOS.get(preset, SCENARIOS["Low light"])
            iso = st.select_slider("Noise · ISO", [100, 200, 400, 800, 1600, 3200, 6400, 12800], value=int(min(max(sc0.iso, 100), 12800)) if int(sc0.iso) in (100, 200, 400, 800, 1600, 3200, 6400, 12800) else 3200)
            ev = st.slider("Light · stops under-exposed", 0.0, 4.0, float(sc0.ev_under), 0.5)
            kel = st.slider("Illuminant · Kelvin", 2500, 9500, int(sc0.kelvin), 100)
            seed = st.number_input("Noise seed", 0, 9999, 1)
        st.markdown('<div class="tag" style="margin-top:1.2rem">3 · Pipelines</div>', unsafe_allow_html=True)
        avail = [p for p in PIPES_ALL if p == "Traditional ISP" or p in models]
        chosen = st.multiselect("Pipelines", avail, default=[p for p in ["Traditional ISP", "Learned ISP", "Hybrid ISP"] if p in avail], label_visibility="collapsed")
        with st.expander("Hybrid settings"):
            strength = st.slider("Learned strength", 0.0, 1.0, 1.0, 0.05)
            sharp = st.slider("Sharpening", 0.0, 1.0, 0.5, 0.05)
            skip_thr = st.checkbox("Adaptive compute: skip the network on clean frames", False)
            tile = st.selectbox("Tile size (memory bound)", [None, 128, 256], format_func=lambda v: "off" if v is None else f"{v}px")
        try:
            go = st.button("Process", type="primary", width="stretch")
        except TypeError:
            go = st.button("Process", type="primary", use_container_width=True)

    with right:
        ready = gt is not None or raw_in is not None
        if not ready:
            st.markdown('<div class="note" style="padding:5rem 0;text-align:center">Pick a photo or RAW frame on the left, then press <b>Process</b>.</div>', unsafe_allow_html=True)
        elif go or st.session_state.get("last_key"):
            from evaluation.comparison import run_pipeline
            from evaluation.metrics import calculate_metrics
            from utils.degradation import Scenario, naive_preview, simulate_raw

            if go:
                if gt is not None:
                    sc = Scenario(iso=iso, ev_under=ev, kelvin=kel)
                    raw = simulate_raw(gt, sc, int(seed))
                else:
                    raw = raw_in
                outs, mets = {}, {}
                pm = pipe_models(models)
                with st.spinner("Running pipelines…"):
                    for n in chosen:
                        kw = {}
                        if n == "Traditional ISP":
                            kw = dict(sharpen_amount=sharp)
                        else:
                            kw = dict(tile=tile)
                            if n != "Learned ISP":
                                kw.update(strength=strength, sharpen_amount=sharp, skip_threshold=0.02 if skip_thr else None)
                        t = time.perf_counter()
                        r = run_pipeline(n, raw, pm, **kw)
                        r.info["wall_ms"] = (time.perf_counter() - t) * 1e3
                        outs[n] = r
                        if gt is not None:
                            g = gt[: r.rgb.shape[0], : r.rgb.shape[1]]
                            mets[n] = calculate_metrics(r.rgb, g, lpips_on=False)
                naive = naive_preview(raw)
                if gt is not None:
                    mets["Unprocessed (naive)"] = calculate_metrics(naive, gt[: naive.shape[0], : naive.shape[1]], lpips_on=False)
                st.session_state.update(last_key=True, outs=outs, mets=mets, naive=naive, gt=gt, truth=getattr(raw, "truth", {}))
            outs, mets, naive, gt_, truth = (st.session_state[k] for k in ("outs", "mets", "naive", "gt", "truth"))
            if not outs:
                st.info("Select at least one pipeline.")
            else:
                names = list(outs)
                # ---- KPI strip -------------------------------------------------
                if gt_ is not None and mets:
                    hero = "Hybrid ISP" if "Hybrid ISP" in outs else names[-1]
                    base = "Traditional ISP" if "Traditional ISP" in outs and hero != "Traditional ISP" else None
                    m, mb = mets[hero], mets.get(base) if base else None
                    k = st.columns(4)
                    dp = f"{m['PSNR'] - mb['PSNR']:+.2f} dB vs traditional" if mb else f"{hero}"
                    k[0].markdown(kpi("PSNR", f"{m['PSNR']:.1f}", "dB", dp, (m['PSNR'] > mb['PSNR']) if mb else None, True), unsafe_allow_html=True)
                    ds = f"{m['SSIM'] - mb['SSIM']:+.3f} vs traditional" if mb else None
                    k[1].markdown(kpi("SSIM", f"{m['SSIM']:.3f}", "", ds, (m['SSIM'] > mb['SSIM']) if mb else None), unsafe_allow_html=True)
                    dd = f"{m['Delta E'] - mb['Delta E']:+.2f} vs traditional" if mb else None
                    k[2].markdown(kpi("Colour error ΔE", f"{m['Delta E']:.2f}", "", dd, (m['Delta E'] < mb['Delta E']) if mb else None), unsafe_allow_html=True)
                    k[3].markdown(kpi("Time", f"{outs[hero].info['wall_ms']:.0f}", "ms", f"{hero}"), unsafe_allow_html=True)
                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                # ---- before / after slider ------------------------------------
                st.markdown('<div class="tag">Compare</div>', unsafe_allow_html=True)
                options = ["Sensor (unprocessed)"] + names + (["Ground truth"] if gt_ is not None else [])
                c1, c2 = st.columns(2)
                a_name = c1.selectbox("Left", options, index=0)
                b_name = c2.selectbox("Right", options, index=min(len(options) - 1, options.index("Hybrid ISP") if "Hybrid ISP" in options else 1))

                def get(nm):
                    return naive if nm.startswith("Sensor") else gt_ if nm == "Ground truth" else outs[nm].rgb
                A, B = get(a_name), get(b_name)
                h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
                components.html(compare_html(A[:h, :w], B[:h, :w], a_name, b_name, COLORS.get(a_name, INK2), COLORS.get(b_name, ACCENT)),
                                height=int(min(760, 1000 * h / w)) + 30)
                # ---- tiles -----------------------------------------------------
                st.markdown('<div class="tag" style="margin-top:.6rem">All outputs</div>', unsafe_allow_html=True)
                tiles = [("Sensor (unprocessed)", naive, mets.get("Unprocessed (naive)"))] + [(n, outs[n].rgb, mets.get(n)) for n in names]
                if gt_ is not None:
                    tiles.append(("Ground truth", gt_, None))
                cols = st.columns(min(3, len(tiles)))
                for i, (n, im, m) in enumerate(tiles):
                    with cols[i % len(cols)]:
                        col = COLORS.get(n, COLORS["Unprocessed (naive)"] if n.startswith("Sensor") else INK)
                        chip = f"{m['PSNR']:.1f} dB" if m else ("reference" if n == "Ground truth" else "")
                        st.markdown(f'<div class="tile-t"><b><span class="dot" style="background:{col}"></span>{n}</b><span>{chip}</span></div>', unsafe_allow_html=True)
                        show_img(np.clip(im, 0, 1))
                # ---- table -----------------------------------------------------
                if mets:
                    rows = []
                    for n, m in mets.items():
                        rows.append({"Pipeline": n, "PSNR dB": m["PSNR"], "SSIM": m["SSIM"], "ΔE 2000": m["Delta E"], "MS-SSIM": m["MS-SSIM"],
                                     "GMSD ↓": m["GMSD"], "ms": outs[n].info["wall_ms"] if n in outs else None})
                    st.markdown('<div class="tag" style="margin-top:1rem">Metrics</div>', unsafe_allow_html=True)
                    stretch_df(pd.DataFrame(rows).round(3))
                    st.caption("PSNR/SSIM higher is better · ΔE 2000 and GMSD lower is better. LPIPS is reported in the offline benchmark when the optional `lpips` package is installed.")
                else:
                    st.info("No reference for real RAW input: reference-based scores are unavailable.")
                # ---- diagnostics -----------------------------------------------
                with st.expander("What the ISP estimated about this frame"):
                    info = next(iter(outs.values())).info
                    dcols = st.columns(3)
                    dcols[0].metric("Noise σ at mid-grey", f"{info['noise_sigma_mid']:.3f}", f"true {np.sqrt(truth['a_eff'] * .18 + truth['b_eff']):.3f}" if truth else None, delta_color="off")
                    if "wb_gains" in info:
                        dcols[1].metric("White-balance gains R / B", f"{info['wb_gains'][0]:.2f} / {info['wb_gains'][2]:.2f}",
                                        f"true {1 / truth['cast'][0]:.2f} / {1 / truth['cast'][2]:.2f}" if truth else None, delta_color="off")
                    dcols[2].metric("Exposure gain applied", f"×{info['gain']:.2f}")
                    import plotly.graph_objects as go_
                    fig = go_.Figure()
                    for n in names:
                        sm = outs[n].info["stage_ms"]
                        for stage, v in sm.items():
                            fig.add_bar(y=[n], x=[v], name=stage, orientation="h", showlegend=False, hovertemplate=f"{stage}: %{{x:.1f}} ms<extra></extra>",
                                        marker=dict(color=COLORS.get(n, MUTED), line=dict(color="#0B0B0C", width=2)))
                    fig.update_layout(barmode="stack", title=dict(text="Where the time goes (ms per stage)", font=dict(size=14, color=INK)))
                    plot(plotly_layout(fig, 210, False))

# =========================================================================== BENCHMARKS
with tab_bench:
    if not res:
        st.info("Run `python run_experiments.py` to generate benchmark results.")
    else:
        names = [n for n in PIPES_ALL if n in res["scenarios"]["Low light"]]
        ov = res["overall_psnr"]
        naive = np.mean([res["scenarios"][s]["Unprocessed (naive)"]["PSNR"] for s in res["scenarios"]])
        hy, tr = ov.get("Hybrid ISP"), ov.get("Traditional ISP")
        st.markdown(f"""<div class="tag">Held-out photos · {len(res['test_images'])} images · 6 capture scenarios</div>
        <div class="big-claim" style="margin:.5rem 0 .3rem">{hy - tr:+.1f} dB <em>vs the classical ISP</em></div>
        <div class="note">Mean PSNR — Traditional {tr:.1f} dB · Learned {ov.get('Learned ISP', float('nan')):.1f} dB · <b style="color:{ACCENT}">Hybrid {hy:.1f} dB</b> ·
        unprocessed sensor {naive:.1f} dB. Test photos are never seen in training.</div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        metric = st.radio("Metric", ["PSNR", "SSIM", "Delta E", "GMSD"], horizontal=True, label_visibility="collapsed")
        import plotly.graph_objects as go_
        cols = st.columns(3)
        for col, (sk, sw) in zip(cols, res["sweeps"].items()):
            fig = go_.Figure()
            for n, ser in sw["series"].items():
                fig.add_scatter(x=sw["xlabels"], y=ser[metric], name=n, mode="lines+markers",
                                line=dict(color=COLORS.get(n, MUTED), width=4 if n == "Hybrid ISP" else 2.4, dash=DASH.get(n, "solid")),
                                marker=dict(size=8, line=dict(color="#0B0B0C", width=2)))
            fig.update_layout(title=dict(text=sw["label"], font=dict(size=14, color=INK)), hovermode="x unified")
            with col:
                plot(plotly_layout(fig, 340, sk == "noise"))
        st.markdown('<div class="tag">Scenario table</div>', unsafe_allow_html=True)
        rows = []
        for sc, d in res["scenarios"].items():
            for n, m in d.items():
                rows.append({"Scenario": sc, "Pipeline": n, "PSNR": m["PSNR"], "SSIM": m["SSIM"], "ΔE 2000": m["Delta E"], "MS-SSIM": m["MS-SSIM"], "GMSD ↓": m["GMSD"], "LPIPS ↓": m.get("LPIPS")})
        df = pd.DataFrame(rows)
        if df["LPIPS ↓"].isna().all():          # LPIPS is optional (pip install lpips): hide the empty column
            df = df.drop(columns=["LPIPS ↓"])
        sel = st.multiselect("Scenarios", list(res["scenarios"]), default=list(res["scenarios"]))
        stretch_df(df[df["Scenario"].isin(sel)].round(3))
        for f, cap in (("comparison.png", "Low light · same RAW frame through every pipeline"),):
            if (RESULTS / f).exists():
                show_img(str(RESULTS / f), cap)

# =========================================================================== EFFICIENCY
with tab_eff:
    if not res or not res.get("efficiency"):
        st.info("Run `python run_experiments.py` to generate efficiency results.")
    else:
        import plotly.graph_objects as go_
        eff = res["efficiency"]
        e = {r["name"]: r for r in eff}
        hero = e.get("Hybrid-Lite") or e.get("Hybrid ISP")
        st.markdown('<div class="tag">Mobile budget · measured on this machine, ' + f"{res.get('threads', '?')} CPU threads · torch {res.get('torch', '')}</div>", unsafe_allow_html=True)
        k = st.columns(4)
        k[0].markdown(kpi("Parameters", f"{hero['parameters'] / 1e3:.0f}", "K", hero["name"], None, True), unsafe_allow_html=True)
        k[1].markdown(kpi("Model size", f"{hero['size_mb']:.2f}", "MB", f"{e['Hybrid ISP']['size_mb']:.2f} MB teacher" if 'Hybrid ISP' in e else None), unsafe_allow_html=True)
        k[2].markdown(kpi("Compute", f"{hero.get('gmacs_per_mp', 0):.1f}", "GMAC/MP"), unsafe_allow_html=True)
        k[3].markdown(kpi("Latency", f"{hero['latency_ms']:.0f}", "ms/MP", "model only"), unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        rows = [{"Model": r["name"], "Params": r.get("parameters"), "Size MB": r.get("size_mb"),
                 "FP16 MB": r.get("size_mb_fp16"), "INT8 MB": r.get("size_mb_int8"), "GMAC/MP": r.get("gmacs_per_mp"),
                 "Latency 1 MP ms": r.get("latency_ms"), "Latency 4 MP ms": r.get("latency_4mp_ms"),
                 "Peak RAM 1 MP MB": r.get("peak_memory_mb"), "Peak RAM 4 MP MB": r.get("peak_memory_4mp_mb"),
                 "Peak RAM 4 MP tiled MB": r.get("peak_memory_4mp_tiled_mb"), "Full pipeline ms/MP": r.get("pipeline_ms_per_mp"),
                 "PSNR dB": r.get("psnr")} for r in eff]
        stretch_df(pd.DataFrame(rows).round(2).apply(lambda col: col.map(lambda v: v if isinstance(v, str) else ("—" if pd.isna(v) else f"{v:g}"))))
        c1, c2, c3 = st.columns(3)
        for col, key, ttl in ((c1, "size_mb", "Model size (MB)"), (c2, "latency_ms", "Model latency per MP (ms)")):
            fig = go_.Figure(go_.Bar(y=[r["name"] for r in eff if r.get(key)], x=[r[key] for r in eff if r.get(key)], orientation="h",
                                     marker=dict(color=[COLORS.get(r["name"], MUTED) for r in eff if r.get(key)]), text=[f"{r[key]:.2f}" for r in eff if r.get(key)], textposition="outside", cliponaxis=False))
            fig.update_xaxes(range=[0, max(r[key] for r in eff if r.get(key)) * 1.25])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title=dict(text=ttl, font=dict(size=14, color=INK)))
            with col:
                plot(plotly_layout(fig, 300, False))
        fig = go_.Figure()
        for r in eff:
            lat = r.get("pipeline_ms_per_mp") or r["latency_ms"]
            if r.get("psnr") and r["name"] in COLORS:
                fig.add_scatter(x=[lat], y=[r["psnr"]], mode="markers+text", text=[r["name"].replace(" ISP", "")], textposition={"Hybrid ISP": "middle right", "Hybrid-Lite": "bottom center"}.get(r["name"], "top center"), name=r["name"], showlegend=False,
                                marker=dict(size=20, color=COLORS[r["name"]], line=dict(color="#0B0B0C", width=3)))
        fig.update_layout(title=dict(text="Quality vs speed (ms/MP, full pipeline where measured)", font=dict(size=14, color=INK)),
                          xaxis_title="ms per megapixel", yaxis_title="PSNR dB", margin=dict(t=60), xaxis_range=[150, 900])
        with c3:
            plot(plotly_layout(fig, 300, False))
        st.caption("Optimisations applied: knowledge distillation (Hybrid-Lite = 4× fewer parameters), INT8 post-training quantisation, FP16/INT8 storage estimates, and tiled inference to bound peak RAM. Peak RAM is the extra resident memory of one forward pass, measured in a fresh process (Linux high-water mark); tiled inference bounds it. Latency 1 MP is model-only; Full pipeline includes every classical stage on a 1.0 MP frame. Latencies are wall-clock on the machine that ran the benchmark; re-measure on the target phone.")
        if (RESULTS / "efficiency.png").exists():
            show_img(str(RESULTS / "efficiency.png"))

# =========================================================================== PIPELINE
with tab_pipe:
    st.markdown("""<div class="tag">Hybrid architecture</div>
<div class="pipe">
<span class="s c">Black level</span><span class="a">→</span><span class="s c">Exposure gain</span><span class="a">→</span>
<span class="s c">Noise-model estimate</span><span class="a">→</span><span class="s c">Malvar-He-Cutler demosaic</span><span class="a">→</span>
<span class="s c">Auto white balance</span><span class="a">→</span><span class="s l">U-Net residual · denoise + refine</span><span class="a">→</span>
<span class="s c">Colour matrix</span><span class="a">→</span><span class="s c">Tone + gamma</span><span class="a">→</span><span class="s c">Luma sharpen</span></div>
<span class="chip" style="border-color:#3987E5">classical</span><span class="chip" style="border-color:#FF5A1F">learned</span>""", unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="large")
    a.markdown("#### Traditional\nEvery stage is hand-designed: generalised-Anscombe variance stabilisation + guided non-local-means in the Bayer domain, gradient-corrected demosaic, consensus white-balance projected on the daylight locus. Strong, explainable, but denoising is the cost centre and colour fails on scenes with dominant hues.")
    b.markdown("#### Learned\nOne U-Net maps packed RAW + a noise map straight to sRGB. It learns the sensor colour transform and denoising together, but it has to rediscover optics, gamma and white balance from data and can drift in colour or hallucinate texture.")
    c.markdown("#### Hybrid\nClassical stages do what has a closed form (exposure, demosaic, AWB, CCM, tone curve, sharpening). The network predicts only a *residual* in the linear domain and is trained **through** the differentiable CCM + sRGB curve, so it optimises the final picture while the pipeline stays physically grounded.")
    st.markdown('<div class="tag" style="margin-top:1.5rem">Adaptive behaviour</div>', unsafe_allow_html=True)
    st.markdown("""
- **Noise** — `var = a·x + b` is estimated blind from the frame (no ISO metadata needed) and given to the network as a per-pixel noise map; classical denoise strength and sharpen coring scale with it.
- **Illumination** — exposure gain is applied before noise estimation (so low light is modelled as amplified noise); AWB gains are constrained to the daylight/Planckian locus; the network also receives a global descriptor (channel means, highlights, noise, gain) so it can correct residual casts.
- **Compute** — optional early exit skips the network on clean frames; tiled inference with a shared global descriptor bounds RAM without seams.
""")

# =========================================================================== TRAIN
with tab_train:
    st.markdown('<div class="tag">Train / fine-tune</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Training draws patches from <code>data/public</code> and <code>data/reference</code> (drop your own clean photos there), simulates a degraded Bayer RAW for each, and optimises the network. '
                'Test photos are excluded from training by name.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Model", ["hybrid", "learned"])
    base_ch = c2.select_slider("Width (base channels)", [4, 8, 16, 24], value=16)
    steps = c3.slider("Steps", 50, 6000, 600, 50)
    teacher = c4.selectbox("Distil from", ["none"] + [k for k in ("hybrid",) if CHECKPOINTS[k].exists()])
    out_name = st.text_input("Save as", f"models/checkpoints/{mode}_custom.pt")
    ups = st.file_uploader("Extra clean photos (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Start training"):
        for i, f in enumerate(ups):
            (BASE / "data/reference").mkdir(exist_ok=True)
            (BASE / "data/reference" / f"user_{i}_{f.name}").write_bytes(f.getvalue())
        from models.train import train
        bar, status, chart = st.progress(0.0), st.empty(), st.empty()
        hist = []

        def cb(step, total, loss, val):
            bar.progress(step / total)
            hist.append({"step": step, "loss": loss})
            status.markdown(f"step **{step}/{total}** · loss `{loss:.4f}` · val PSNR `{val:.2f} dB`")
            chart.line_chart(pd.DataFrame(hist).set_index("step"))
        try:
            r = train(mode, base_ch, steps, out=BASE / out_name, distill=None if teacher == "none" else teacher, progress=cb, log_every=25, verbose=False)
            st.success(f"Saved {out_name} · best validation PSNR {r['best_val_psnr']:.2f} dB")
        except Exception as e:
            st.error(f"Training failed: {e}")
