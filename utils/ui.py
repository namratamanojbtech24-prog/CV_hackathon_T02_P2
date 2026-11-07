"""Streamlit UI helpers: CSS, KPI tiles, before/after slider, Plotly theme."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from utils.theme import ACCENT, COLORS, INK, INK2, LINE, MUTED, PANEL, SURFACE

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root {{ --accent:{ACCENT}; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --line:{LINE}; --panel:{PANEL}; }}
html, body, [class*="css"], .stApp {{ font-family:'Inter',system-ui,sans-serif; }}
.stApp {{ background:{SURFACE}; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none!important; }}
[data-testid="stHeader"] {{ background:transparent; height:0; }}
.block-container {{ max-width:1280px; padding:2.2rem 2rem 4rem; }}
h1,h2,h3,h4 {{ letter-spacing:-0.02em; color:var(--ink); }}
.hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; padding-bottom:1.2rem;
        border-bottom:1px solid var(--line); margin-bottom:.6rem; }}
.hero h1 {{ font-size:3.1rem; font-weight:800; line-height:.95; margin:0; letter-spacing:-0.045em; }}
.hero h1 em {{ color:var(--accent); font-style:normal; }}
.hero p {{ color:var(--ink2); margin:.7rem 0 0; font-size:1.02rem; max-width:560px; }}
.tag {{ font:700 .72rem 'JetBrains Mono',monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }}
.stTabs [data-baseweb="tab-list"] {{ gap:2.2rem; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding:.9rem 0; background:transparent; font-weight:700; font-size:.95rem; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:var(--ink)!important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent); height:3px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1.05rem 1.2rem; height:100%; }}
.kpi .l {{ font:700 .7rem 'JetBrains Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.kpi .v {{ font:800 2.4rem/1.05 'Inter'; letter-spacing:-.04em; margin-top:.35rem; color:var(--ink); }}
.kpi .v small {{ font-size:1rem; font-weight:600; color:var(--ink2); margin-left:.25rem; letter-spacing:0; }}
.kpi .d {{ font:700 .8rem 'JetBrains Mono',monospace; margin-top:.35rem; color:var(--muted); }}
.kpi .d.good {{ color:#3FB68B; }} .kpi .d.bad {{ color:#E66767; }}
.kpi.hero-k {{ border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }}
.chip {{ display:inline-block; padding:.18rem .6rem; border-radius:99px; font:700 .72rem 'JetBrains Mono',monospace;
        border:1px solid var(--line); color:var(--ink2); margin-right:.35rem; }}
.dot {{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.4rem; }}
.tile-t {{ display:flex; justify-content:space-between; align-items:baseline; margin:.55rem 0 .3rem; }}
.tile-t b {{ font-weight:800; letter-spacing:-.02em; }}
.tile-t span {{ font:700 .8rem 'JetBrains Mono',monospace; color:var(--ink2); }}
.stButton>button {{ background:var(--accent); color:#0B0B0C; border:0; border-radius:10px; font-weight:800; padding:.65rem 1.2rem; letter-spacing:-.01em; }}
.stButton>button:hover {{ background:#FF7A47; color:#0B0B0C; }}
.stButton>button:focus {{ box-shadow:0 0 0 3px rgba(255,90,31,.35); color:#0B0B0C; }}
[data-testid="stImage"] img {{ border-radius:12px; }}
[data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:12px; background:var(--panel); }}
hr {{ border-color:var(--line); }}
.big-claim {{ font:800 clamp(2.2rem,5vw,4rem)/1 'Inter'; letter-spacing:-.05em; }}
.big-claim em {{ color:var(--accent); font-style:normal; }}
.stage {{ display:flex; flex-direction:column; gap:.4rem; }}
.pipe {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.4rem 0 1.2rem; }}
.pipe .s {{ padding:.55rem .85rem; border-radius:10px; font-weight:700; font-size:.9rem; border:1px solid var(--line); background:var(--panel); }}
.pipe .s.c {{ border-color:{COLORS['Traditional ISP']}; }}
.pipe .s.l {{ border-color:{ACCENT}; background:rgba(255,90,31,.12); }}
.pipe .a {{ color:var(--muted); font-weight:800; }}
.note {{ color:var(--ink2); font-size:.9rem; }}
</style>
"""


def kpi(label, value, unit="", delta=None, good=None, hero=False):
    d = ""
    if delta is not None:
        cls = "" if good is None else ("good" if good else "bad")
        d = f'<div class="d {cls}">{delta}</div>'
    return (f'<div class="kpi {"hero-k" if hero else ""}"><div class="l">{label}</div>'
            f'<div class="v">{value}<small>{unit}</small></div>{d}</div>')


def _jpeg_b64(img, max_w=1000, q=90):
    im = Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255 + 0.5))
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode(), im.size


def compare_html(a, b, label_a, label_b, col_a=INK2, col_b=ACCENT):
    """Self-contained before/after slider (drag, or use the arrow keys)."""
    ia, (w, h) = _jpeg_b64(a)
    ib, _ = _jpeg_b64(b)
    return f"""
<div id="c" style="position:relative;width:100%;max-width:{w}px;aspect-ratio:{w}/{h};margin:auto;border-radius:14px;overflow:hidden;
     user-select:none;touch-action:none;background:#000;font-family:Inter,system-ui,sans-serif">
  <img src="data:image/jpeg;base64,{ib}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
  <div id="top" style="position:absolute;inset:0;clip-path:inset(0 50% 0 0)">
    <img src="data:image/jpeg;base64,{ia}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover"></div>
  <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#F2F0EB;transform:translateX(-1.5px)">
    <div style="position:absolute;top:50%;left:50%;width:42px;height:42px;border-radius:50%;background:#0B0B0C;border:3px solid #F2F0EB;
      transform:translate(-50%,-50%);display:flex;align-items:center;justify-content:center;color:#F2F0EB;font-weight:800;font-size:15px">⇄</div></div>
  <span style="position:absolute;left:12px;top:12px;background:rgba(11,11,12,.82);color:{col_a};padding:5px 11px;border-radius:99px;font-weight:800;font-size:13px">{label_a}</span>
  <span style="position:absolute;right:12px;top:12px;background:rgba(11,11,12,.82);color:{col_b};padding:5px 11px;border-radius:99px;font-weight:800;font-size:13px">{label_b}</span>
</div>
<script>
const c=document.getElementById('c'),t=document.getElementById('top'),b=document.getElementById('bar');
function set(x){{const r=c.getBoundingClientRect();let p=Math.max(0,Math.min(1,(x-r.left)/r.width))*100;t.style.clipPath='inset(0 '+(100-p)+'% 0 0)';b.style.left=p+'%';}}
let d=false;c.addEventListener('pointerdown',e=>{{d=true;c.setPointerCapture(e.pointerId);set(e.clientX)}});
c.addEventListener('pointermove',e=>{{if(d)set(e.clientX)}});c.addEventListener('pointerup',()=>d=false);
</script>"""


def plotly_layout(fig, height=360, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK2, size=13),
        margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK)),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK2)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK2)),
    )
    return fig
