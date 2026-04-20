import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="PortCast · Kenya Maritime Intelligence",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# Navy #0d1b2e  |  Slate #1c2e45  |  Cream #f5f0e8
# Coral #e85d4a |  Gold #d4a853   |  Sky #4a9eca
# Text primary #0d1b2e | muted #6b7a8d
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{
  font-family:'DM Sans',sans-serif;
  background:#f5f0e8;
  color:#0d1b2e;
}
[data-testid="stHeader"],[data-testid="stToolbar"],footer,
[data-testid="stDecoration"]{display:none!important}
[data-testid="block-container"]{padding:0 2rem 4rem!important;max-width:1400px}

/* ── HERO ─────────────────────────────────────────────────────────── */
.hero-wrap{
  display:grid;
  grid-template-columns:1fr 420px;
  gap:2rem;
  background:#0d1b2e;
  border-radius:0 0 32px 32px;
  padding:3rem 3rem 2.5rem;
  margin:-1rem -2rem 2.5rem;
}
.hero-left{}
.hero-eyebrow{
  font-family:'DM Sans',sans-serif;
  font-size:.72rem;font-weight:600;
  letter-spacing:.18em;text-transform:uppercase;
  color:#e85d4a;margin-bottom:.9rem;
}
.hero-title{
  font-family:'Playfair Display',serif;
  font-size:2.9rem;font-weight:800;
  color:#f5f0e8;
  line-height:1.1;margin-bottom:1rem;
}
.hero-title span{color:#d4a853}
.hero-desc{
  color:rgba(245,240,232,.65);
  font-size:.96rem;line-height:1.7;
  max-width:520px;margin-bottom:1.5rem;
}
.hero-tags{display:flex;flex-wrap:wrap;gap:.5rem}
.hero-tag{
  background:rgba(245,240,232,.08);
  border:1px solid rgba(245,240,232,.18);
  color:rgba(245,240,232,.75);
  border-radius:4px;
  padding:.25rem .75rem;
  font-size:.75rem;font-weight:500;
  letter-spacing:.04em;
}
.hero-right{
  display:flex;flex-direction:column;
  justify-content:center;gap:1rem;
}
.stat-card{
  background:rgba(245,240,232,.06);
  border:1px solid rgba(245,240,232,.12);
  border-radius:12px;
  padding:1rem 1.3rem;
  display:flex;align-items:center;gap:1rem;
}
.stat-card .sc-val{
  font-family:'Playfair Display',serif;
  font-size:1.9rem;font-weight:700;
  line-height:1;
}
.stat-card .sc-val.coral{color:#e85d4a}
.stat-card .sc-val.gold{color:#d4a853}
.stat-card .sc-val.sky{color:#4a9eca}
.stat-card .sc-val.lime{color:#7ec98e}
.stat-card .sc-lbl{
  font-size:.72rem;font-weight:500;
  letter-spacing:.06em;text-transform:uppercase;
  color:rgba(245,240,232,.45);
  margin-top:.15rem;
}
.stat-card .sc-icon{font-size:1.5rem;opacity:.7}

/* ── PIPELINE BAR ─────────────────────────────────────────────────── */
.pipeline{
  display:flex;align-items:center;
  gap:0;margin-bottom:2.5rem;
  background:white;
  border-radius:14px;
  border:1px solid #e2ddd4;
  overflow:hidden;
}
.pipe-step{
  flex:1;padding:1rem 1.2rem;
  border-right:1px solid #e2ddd4;
  position:relative;
}
.pipe-step:last-child{border-right:none}
.pipe-num{
  font-size:.65rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;
  color:#e85d4a;margin-bottom:.2rem;
}
.pipe-name{
  font-size:.88rem;font-weight:600;color:#0d1b2e;
}
.pipe-desc{font-size:.75rem;color:#6b7a8d;margin-top:.15rem}
.pipe-arr{
  color:#6b7a8d;font-size:.9rem;
  padding:0 .2rem;flex-shrink:0;
}

/* ── SECTION LABELS ───────────────────────────────────────────────── */
.sec-label{
  font-size:.68rem;font-weight:700;
  letter-spacing:.15em;text-transform:uppercase;
  color:#e85d4a;margin-bottom:.5rem;
}
.sec-title{
  font-family:'Playfair Display',serif;
  font-size:1.6rem;font-weight:700;
  color:#0d1b2e;margin-bottom:1.5rem;
  padding-bottom:.7rem;
  border-bottom:2px solid #0d1b2e;
}

/* ── INPUT CARDS ──────────────────────────────────────────────────── */
.icard{
  background:white;
  border:1px solid #e2ddd4;
  border-radius:16px;
  padding:1.4rem 1.5rem 1rem;
  height:100%;
}
.icard-head{
  display:flex;align-items:center;gap:.6rem;
  font-size:.75rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;
  color:#6b7a8d;margin-bottom:1.1rem;
  padding-bottom:.7rem;
  border-bottom:1px solid #ede8df;
}
.icard-head .dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
}
.dot-coral{background:#e85d4a}
.dot-gold{background:#d4a853}
.dot-sky{background:#4a9eca}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label{
  color:#6b7a8d!important;font-size:.82rem!important;font-weight:500!important;
}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"]>div>div,
[data-testid="stDateInput"] input{
  background:#faf8f4!important;
  border:1px solid #e2ddd4!important;
  color:#0d1b2e!important;
  border-radius:8px!important;
}

/* ── FORECAST BUTTON ──────────────────────────────────────────────── */
[data-testid="stButton"]>button[kind="primary"]{
  background:#0d1b2e!important;
  color:#f5f0e8!important;
  border:none!important;
  border-radius:10px!important;
  font-weight:600!important;font-size:1rem!important;
  padding:.8rem 1.5rem!important;
  width:100%;letter-spacing:.04em;
  font-family:'DM Sans',sans-serif!important;
}
[data-testid="stButton"]>button[kind="primary"]:hover{
  background:#1c2e45!important;
}

/* ── RESULT CARD ──────────────────────────────────────────────────── */
.result-card{
  background:#0d1b2e;
  border-radius:20px;
  padding:2.2rem 2rem 1.8rem;
  display:flex;flex-direction:column;
  align-items:center;text-align:center;
  margin-bottom:1.2rem;
  position:relative;overflow:hidden;
}
.result-card::before{
  content:'';position:absolute;
  bottom:-40px;right:-40px;
  width:180px;height:180px;border-radius:50%;
  background:rgba(212,168,83,.08);
}
.result-card::after{
  content:'';position:absolute;
  top:-30px;left:-30px;
  width:120px;height:120px;border-radius:50%;
  background:rgba(232,93,74,.06);
}
.result-eyebrow{
  font-size:.68rem;font-weight:600;
  letter-spacing:.15em;text-transform:uppercase;
  color:rgba(245,240,232,.45);margin-bottom:.6rem;
}
.result-num{
  font-family:'Playfair Display',serif;
  font-size:4.2rem;font-weight:800;
  color:#f5f0e8;line-height:1;
  margin-bottom:.3rem;position:relative;z-index:1;
}
.result-unit{color:#d4a853;font-size:1.1rem;font-weight:600;letter-spacing:.04em}
.result-sub{
  color:rgba(245,240,232,.5);font-size:.82rem;margin-top:.8rem;
}
.result-delta{
  margin-top:1rem;
  background:rgba(126,201,142,.12);
  border:1px solid rgba(126,201,142,.25);
  border-radius:8px;padding:.5rem .9rem;
  font-size:.82rem;font-weight:600;color:#7ec98e;
}
.result-zero-card{
  background:#f0ebe0;border:2px dashed #c9c0b0;
  border-radius:20px;padding:2.2rem 2rem 1.8rem;
  text-align:center;margin-bottom:1.2rem;
}
.result-zero-card .big{
  font-family:'Playfair Display',serif;
  font-size:4rem;font-weight:700;color:#b0a898;
}

/* ── STAGE PANELS ─────────────────────────────────────────────────── */
.stage-panel{
  background:white;border:1px solid #e2ddd4;
  border-radius:16px;padding:1.3rem 1.4rem;
  margin-bottom:1rem;
}
.stage-head{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:1rem;padding-bottom:.7rem;
  border-bottom:1px solid #ede8df;
}
.stage-badge{
  font-size:.68rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;
  background:#0d1b2e;color:#f5f0e8;
  border-radius:4px;padding:.25rem .65rem;
}
.stage-status-ok{
  font-size:.78rem;font-weight:600;color:#2d7d46;
  background:#e8f5ec;border-radius:20px;padding:.2rem .7rem;
}
.stage-status-skip{
  font-size:.78rem;font-weight:600;color:#9ca3af;
  background:#f3f4f6;border-radius:20px;padding:.2rem .7rem;
}

/* ── CONF BAR ─────────────────────────────────────────────────────── */
.cbar-wrap{margin:.5rem 0 1rem}
.cbar-labels{display:flex;justify-content:space-between;font-size:.78rem;color:#6b7a8d;margin-bottom:.35rem}
.cbar-track{background:#ede8df;border-radius:99px;height:8px;overflow:hidden}
.cbar-fill{height:100%;border-radius:99px;transition:width .5s ease}

/* ── BENCH TILES ──────────────────────────────────────────────────── */
.bench-row{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-top:.8rem}
.btile{
  background:#faf8f4;border:1px solid #ede8df;
  border-radius:10px;padding:.65rem .85rem;
}
.btile .bv{font-size:.98rem;font-weight:700;color:#0d1b2e}
.btile .bl{font-size:.7rem;color:#6b7a8d;margin-top:.1rem}

/* ── HOW CARDS ────────────────────────────────────────────────────── */
.how-card{
  background:white;border:1px solid #e2ddd4;
  border-radius:16px;padding:1.4rem 1.5rem;
  border-top:3px solid #0d1b2e;
  height:100%;
}
.how-num{
  font-size:.68rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:#e85d4a;margin-bottom:.4rem;
}
.how-card h4{font-size:1rem;font-weight:600;color:#0d1b2e;margin-bottom:.5rem}
.how-card p{font-size:.85rem;color:#6b7a8d;line-height:1.65}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

art          = load_model()
preprocessor = art["preprocessor"]
clf          = art["clf"]
scaler       = art["scaler"]
rf           = art["rf"]
lgb_reg      = art["lgb_reg"]
svr          = art["svr"]
ridge        = art["ridge"]
FEATURES     = art["FEATURES"]
metrics      = art["metrics"]

def build_row(portname, date, pc_cont, pc_bulk, pc_gen, pc_roro, pc_tank,
              im_cont, im_bulk, im_gen, im_roro, im_tank,
              ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank):
    total_pc = pc_cont+pc_bulk+pc_gen+pc_roro+pc_tank
    total_im = im_cont+im_bulk+im_gen+im_roro+im_tank
    total_ex = ex_cont+ex_bulk+ex_gen+ex_roro+ex_tank
    total_tp = total_im+total_ex
    row = dict(
        portname=portname,
        portcalls_container=pc_cont, portcalls_dry_bulk=pc_bulk,
        portcalls_general_cargo=pc_gen, portcalls_roro=pc_roro, portcalls_tanker=pc_tank,
        total_portcalls=total_pc,
        import_container=im_cont, import_dry_bulk=im_bulk, import_general_cargo=im_gen,
        import_roro=im_roro, import_tanker=im_tank,
        export_container=ex_cont, export_dry_bulk=ex_bulk, export_general_cargo=ex_gen,
        export_roro=ex_roro, export_tanker=ex_tank,
        tanker_call_ratio=pc_tank/total_pc if total_pc>0 else 0,
        container_call_ratio=pc_cont/total_pc if total_pc>0 else 0,
        bulk_call_ratio=pc_bulk/total_pc if total_pc>0 else 0,
        import_export_ratio=total_im/total_tp if total_tp>0 else 0.5,
        month=date.month, quarter=(date.month-1)//3+1,
        day_of_week=date.weekday(), is_weekend=int(date.weekday() in [5,6]),
    )
    return pd.DataFrame([row])[FEATURES]

def predict(row_df):
    enc   = preprocessor.transform(row_df)
    p_act = clf.predict_proba(enc)[0,1]
    active= clf.predict(enc)[0]
    if active==0:
        return 0.0, p_act, None, None, None
    enc_sc = scaler.transform(enc)
    p_rf   = float(rf.predict(enc)[0])
    p_lgb  = float(lgb_reg.predict(enc)[0])
    p_svr  = float(svr.predict(enc_sc)[0])
    meta   = np.array([[p_rf, p_lgb, p_svr, p_act]])
    final  = float(np.clip(ridge.predict(meta)[0], 0, None))
    return final, p_act, p_rf, p_lgb, p_svr

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-left">
    <div class="hero-eyebrow">⚓ Kenya Maritime Intelligence Platform</div>
    <div class="hero-title">Daily Port<br>Throughput<br><span>Forecaster</span></div>
    <div class="hero-desc">
      A two-stage zero-inflated stacking ensemble (Hybrid 1) trained on 5,048
      AIS-based daily observations from Mombasa and Lamu ports (2019–2025).
      Novel architecture combining LightGBM, Random Forest, and SVR under a
      Ridge meta-learner — achieving R² of {metrics['r2']:.4f} on held-out data.
    </div>
    <div class="hero-tags">
      <span class="hero-tag">Hybrid 1 Novel Algorithm</span>
      <span class="hero-tag">HDX PortWatch · AIS Data</span>
      <span class="hero-tag">Zero-Inflated Stacking</span>
      <span class="hero-tag">Strathmore University MSc DSA</span>
      <span class="hero-tag">Kiragu, A.W. · 2025</span>
    </div>
  </div>
  <div class="hero-right">
    <div class="stat-card">
      <div class="sc-icon">🎯</div>
      <div>
        <div class="sc-val coral">{metrics['r2']:.4f}</div>
        <div class="sc-lbl">R² Score · Test Set</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="sc-icon">📏</div>
      <div>
        <div class="sc-val gold">{metrics['mae']:,.0f} t</div>
        <div class="sc-lbl">Mean Absolute Error</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="sc-icon">📡</div>
      <div>
        <div class="sc-val sky">&gt; 0.99</div>
        <div class="sc-lbl">Stage 1 ROC-AUC</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="sc-icon">🗄</div>
      <div>
        <div class="sc-val lime">5,048</div>
        <div class="sc-lbl">Daily Observations · 2019–2025</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE DIAGRAM ──────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
  <div class="pipe-step">
    <div class="pipe-num">Input</div>
    <div class="pipe-name">AIS Port Activity</div>
    <div class="pipe-desc">22 engineered features</div>
  </div>
  <div class="pipe-arr">→</div>
  <div class="pipe-step">
    <div class="pipe-num">Stage 1</div>
    <div class="pipe-name">Zero Classifier</div>
    <div class="pipe-desc">LightGBM · P(active)</div>
  </div>
  <div class="pipe-arr">→</div>
  <div class="pipe-step">
    <div class="pipe-num">Stage 2a</div>
    <div class="pipe-name">Base Regressors</div>
    <div class="pipe-desc">RF · LightGBM · SVR</div>
  </div>
  <div class="pipe-arr">→</div>
  <div class="pipe-step">
    <div class="pipe-num">Stage 2b</div>
    <div class="pipe-name">Meta-Learner</div>
    <div class="pipe-desc">Ridge · OOF blending</div>
  </div>
  <div class="pipe-arr">→</div>
  <div class="pipe-step">
    <div class="pipe-num">Output</div>
    <div class="pipe-name">Throughput Forecast</div>
    <div class="pipe-desc">Tonnes per day</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">Configure Port Activity Inputs</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown('<div class="icard"><div class="icard-head"><span class="dot dot-coral"></span>Port & Vessel Calls</div>', unsafe_allow_html=True)
    port    = st.selectbox("Select port", ["Mombasa", "Lamu"])
    date    = st.date_input("Forecast date", value=datetime.date.today())
    st.markdown("<hr style='border:none;border-top:1px solid #ede8df;margin:.8rem 0'>", unsafe_allow_html=True)
    pc_cont = st.number_input("Container calls",     min_value=0, max_value=20, value=1, step=1)
    pc_tank = st.number_input("Tanker calls",        min_value=0, max_value=20, value=1, step=1)
    pc_bulk = st.number_input("Dry bulk calls",      min_value=0, max_value=20, value=0, step=1)
    pc_gen  = st.number_input("General cargo calls", min_value=0, max_value=20, value=0, step=1)
    pc_roro = st.number_input("Ro-Ro calls",         min_value=0, max_value=20, value=0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="icard"><div class="icard-head"><span class="dot dot-gold"></span>Import Volumes (tonnes)</div>', unsafe_allow_html=True)
    im_tank = st.number_input("Tanker imports",        min_value=0.0, value=50000.0, step=1000.0, format="%.0f")
    im_bulk = st.number_input("Dry bulk imports",      min_value=0.0, value=20000.0, step=1000.0, format="%.0f")
    im_cont = st.number_input("Container imports",     min_value=0.0, value=10000.0, step=1000.0, format="%.0f")
    im_gen  = st.number_input("General cargo imports", min_value=0.0, value=0.0,     step=1000.0, format="%.0f")
    im_roro = st.number_input("Ro-Ro imports",         min_value=0.0, value=0.0,     step=1000.0, format="%.0f")
    st.markdown("""
<div style="background:#fef8ec;border-left:3px solid #d4a853;border-radius:0 8px 8px 0;
            padding:.65rem .85rem;font-size:.82rem;color:#7a5c1e;margin-top:.8rem">
  <strong>Key insight:</strong> import_tanker is the strongest single predictor
  of throughput (Pearson r = 0.761). Enter it accurately.
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="icard"><div class="icard-head"><span class="dot dot-sky"></span>Export Volumes (tonnes)</div>', unsafe_allow_html=True)
    ex_tank = st.number_input("Tanker exports",        min_value=0.0, value=0.0,    step=1000.0, format="%.0f")
    ex_bulk = st.number_input("Dry bulk exports",      min_value=0.0, value=5000.0, step=1000.0, format="%.0f")
    ex_cont = st.number_input("Container exports",     min_value=0.0, value=3000.0, step=1000.0, format="%.0f")
    ex_gen  = st.number_input("General cargo exports", min_value=0.0, value=0.0,   step=1000.0, format="%.0f")
    ex_roro = st.number_input("Ro-Ro exports",         min_value=0.0, value=0.0,   step=1000.0, format="%.0f")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Mombasa historical reference**")
    st.markdown("""
<div class="bench-row">
  <div class="btile"><div class="bv">63,855 t</div><div class="bl">July — minimum</div></div>
  <div class="btile"><div class="bv">74,951 t</div><div class="bl">October — peak</div></div>
  <div class="btile"><div class="bv">69,012 t</div><div class="bl">2019 avg</div></div>
  <div class="btile"><div class="bv">78,092 t</div><div class="bl">2025 avg</div></div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_, bcol, _ = st.columns([1.8, 1, 1.8])
with bcol:
    run = st.button("⚓  Run Forecast", type="primary", use_container_width=True)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if run:
    row_df = build_row(
        port, date, pc_cont, pc_bulk, pc_gen, pc_roro, pc_tank,
        im_cont, im_bulk, im_gen, im_roro, im_tank,
        ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank,
    )
    final, p_act, p_rf, p_lgb, p_svr = predict(row_df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Forecast Results</div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1.15, 1, 1.1], gap="large")

    # ── Result + donut ────────────────────────────────────────────────────────
    with left:
        if p_act >= 0.5:
            pct = (final / 69000 - 1) * 100
            delta_html = f'<div class="result-delta">{"▲" if pct>=0 else "▼"} {abs(pct):.1f}% vs historical average</div>'
            st.markdown(f"""
<div class="result-card">
  <div class="result-eyebrow">Predicted Daily Throughput · {port} · {date.strftime('%d %b %Y')}</div>
  <div class="result-num">{final:,.0f}</div>
  <div class="result-unit">TONNES</div>
  <div class="result-sub">Ridge meta-learner output · Hybrid 1 pipeline</div>
  {delta_html}
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-zero-card">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.12em;
              text-transform:uppercase;color:#b0a898;margin-bottom:.6rem">
    Predicted Daily Throughput · {port} · {date.strftime('%d %b %Y')}
  </div>
  <div class="big">0</div>
  <div style="color:#b0a898;font-size:.95rem;margin:.4rem 0">tonnes / day</div>
  <div style="color:#c9c0b0;font-size:.82rem;margin-top:.8rem">
    Stage 1 classifier detected no active port day.<br>
    Stage 2 regression was not triggered.
  </div>
</div>""", unsafe_allow_html=True)

        # Donut — cream background to match page
        labels   = ["Container","Dry Bulk","General","Ro-Ro","Tanker"]
        im_vals  = [im_cont, im_bulk, im_gen, im_roro, im_tank]
        ex_vals  = [ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank]
        tot_vals = [i+e for i,e in zip(im_vals, ex_vals)]
        donut = go.Figure(go.Pie(
            labels=labels, values=tot_vals, hole=0.6,
            marker_colors=["#4a9eca","#d4a853","#7ec98e","#e85d4a","#9b8ecf"],
            textinfo="percent", showlegend=True,
            hovertemplate="%{label}: %{value:,.0f} t<extra></extra>",
        ))
        donut.update_layout(
            title=dict(text="Cargo composition (input)", font=dict(color="#6b7a8d", size=12, family="DM Sans")),
            height=270, margin=dict(l=0,r=0,t=36,b=0),
            paper_bgcolor="rgba(245,240,232,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6b7a8d", family="DM Sans"),
            legend=dict(font=dict(color="#6b7a8d", size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(donut, use_container_width=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    with mid:
        status_cls  = "stage-status-ok" if p_act >= 0.5 else "stage-status-skip"
        status_text = "Active ✓" if p_act >= 0.5 else "Inactive —"
        conf_pct    = round(p_act * 100, 1)
        bar_col     = "#2d7d46" if p_act >= 0.5 else "#b0a898"

        st.markdown(f"""
<div class="stage-panel">
  <div class="stage-head">
    <span class="stage-badge">Stage 1 — Classifier</span>
    <span class="{status_cls}">{status_text}</span>
  </div>
  <div style="font-size:.78rem;color:#6b7a8d;margin-bottom:.3rem">LightGBM · class_weight=balanced</div>
  <div class="cbar-wrap">
    <div class="cbar-labels">
      <span>P(active)</span>
      <span style="color:#0d1b2e;font-weight:700">{conf_pct}%</span>
    </div>
    <div class="cbar-track">
      <div class="cbar-fill" style="width:{conf_pct}%;background:{bar_col}"></div>
    </div>
  </div>
  <div style="font-size:.8rem;color:#6b7a8d;line-height:1.55">
    Detects whether the port day is operationally active or structurally
    inactive. Handles Lamu's 96.6% zero-throughput rate through class-balanced
    training. If P(active) &lt; 50%, forecast = 0 t and Stage 2 is skipped.
    <br><br>Test set ROC-AUC: <strong style="color:#0d1b2e">&gt; 0.99</strong>
  </div>
</div>""", unsafe_allow_html=True)

        # Gauge — light themed
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=conf_pct,
            gauge=dict(
                axis=dict(range=[0,100], ticksuffix="%",
                          tickcolor="#c9c0b0", tickfont=dict(color="#6b7a8d", family="DM Sans")),
                bar=dict(color=bar_col),
                bgcolor="#f5f0e8", bordercolor="#e2ddd4",
                steps=[
                    dict(range=[0,50],  color="#faf8f4"),
                    dict(range=[50,100],color="#edf6f0"),
                ],
                threshold=dict(line=dict(color="#0d1b2e", width=2), value=50),
            ),
            number=dict(suffix="%", font=dict(size=28, color="#0d1b2e", family="DM Sans")),
        ))
        fig_g.update_layout(
            height=210, margin=dict(l=20,r=20,t=10,b=10),
            paper_bgcolor="rgba(245,240,232,0)",
            font=dict(color="#6b7a8d", family="DM Sans"),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    with right:
        if p_rf is not None:
            st.markdown("""
<div class="stage-panel" style="margin-bottom:.8rem">
  <div class="stage-head">
    <span class="stage-badge">Stage 2 — Base Models</span>
    <span class="stage-status-ok">Active ✓</span>
  </div>""", unsafe_allow_html=True)

            bar_fig = go.Figure(go.Bar(
                x=[p_rf, p_lgb, p_svr],
                y=["Random Forest", "LightGBM", "SVR (RBF)"],
                orientation='h',
                marker_color=["#4a9eca","#d4a853","#e85d4a"],
                text=[f"{v:,.0f} t" for v in [p_rf, p_lgb, p_svr]],
                textposition="outside",
                textfont=dict(color="#6b7a8d", size=11, family="DM Sans"),
            ))
            bar_fig.update_layout(
                height=185, margin=dict(l=0,r=60,t=0,b=0),
                xaxis=dict(showgrid=False, showticklabels=False, color="#c9c0b0"),
                yaxis=dict(color="#6b7a8d", tickfont=dict(size=11, family="DM Sans")),
                paper_bgcolor="rgba(245,240,232,0)",
                plot_bgcolor="rgba(245,240,232,0)",
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
<div class="stage-panel">
  <div class="stage-head">
    <span class="stage-badge">Stage 2 — Meta-Learner</span>
    <span class="stage-status-ok">Ridge ✓</span>
  </div>""", unsafe_allow_html=True)

            w = ridge.coef_
            wfig = go.Figure(go.Bar(
                x=["RF", "LightGBM", "SVR", "P(active)"],
                y=w,
                marker_color=["#4a9eca","#d4a853","#e85d4a","#9b8ecf"],
                text=[f"{v:.3f}" for v in w],
                textposition="outside",
                textfont=dict(color="#6b7a8d", size=11, family="DM Sans"),
            ))
            wfig.update_layout(
                height=190, margin=dict(l=0,r=0,t=10,b=10),
                xaxis=dict(color="#6b7a8d", tickfont=dict(color="#6b7a8d", family="DM Sans")),
                yaxis=dict(showgrid=True, gridcolor="#ede8df", color="#6b7a8d",
                           zeroline=True, zerolinecolor="#c9c0b0"),
                paper_bgcolor="rgba(245,240,232,0)",
                plot_bgcolor="rgba(245,240,232,0)",
                showlegend=False,
            )
            st.plotly_chart(wfig, use_container_width=True)
            st.markdown('<div style="font-size:.78rem;color:#6b7a8d">RidgeCV · OOF 5-fold · α auto-selected</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="stage-panel" style="text-align:center;padding:2rem">
  <div style="font-size:2.5rem;margin-bottom:.6rem">⏭</div>
  <div style="font-weight:600;color:#0d1b2e;margin-bottom:.3rem">Stage 2 not triggered</div>
  <div style="font-size:.85rem;color:#6b7a8d">Port classified as inactive by Stage 1.<br>Final throughput forecast = 0 tonnes.</div>
</div>""", unsafe_allow_html=True)

# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">How the Hybrid 1 Model Works</div>', unsafe_allow_html=True)

h1, h2, h3 = st.columns(3, gap="large")
with h1:
    st.markdown("""<div class="how-card">
<div class="how-num">Stage 1 — Zero Classifier</div>
<h4>LightGBM Binary Classification</h4>
<p>A histogram-based gradient boosting classifier determines whether a given
port day will record any cargo activity. This directly resolves the structural
zero-inflation problem — Lamu records zero throughput on 96.6% of days,
making pooled single-stage regression statistically unsound.
<br><br>Achieves <strong>ROC-AUC &gt; 0.99</strong> with class_weight='balanced'.</p>
</div>""", unsafe_allow_html=True)
with h2:
    st.markdown("""<div class="how-card">
<div class="how-num">Stage 2 — Stacking Ensemble</div>
<h4>Heterogeneous Base Regressors + Ridge Meta-Learner</h4>
<p>Three models with fundamentally different inductive biases are trained
exclusively on active-day observations: <strong>Random Forest</strong> (bagged trees,
low variance), <strong>LightGBM Regressor</strong> (boosted, handles right-skew),
and <strong>SVR with RBF kernel</strong> (margin-based, corrects high-throughput extremes).
<br><br>A RidgeCV meta-learner blends out-of-fold predictions to prevent leakage.</p>
</div>""", unsafe_allow_html=True)
with h3:
    st.markdown("""<div class="how-card">
<div class="how-num">Feature Engineering — 22 Variables</div>
<h4>AIS-Derived Signals Beyond Vessel Call Counts</h4>
<p>The strongest predictor is <strong>import_tanker</strong> (Pearson r = 0.761) —
cargo volume estimates far outperform raw vessel counts. The 22-variable set
includes individual import/export tonnages, cargo composition ratios, and
temporal dummies encoding seasonality.
<br><br>Trained on 5,048 daily records from HDX PortWatch (OCHA / IMF), 2019–2025.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""<br>
<div style="text-align:center;color:#b0a898;font-size:.78rem;padding:1.5rem 0;
            border-top:1px solid #e2ddd4;margin-top:1rem">
  Kiragu, A.W. (2025) &nbsp;·&nbsp; Strathmore University MSc Data Science &amp; Analytics
  &nbsp;·&nbsp; HDX PortWatch AIS data (OCHA / IMF) &nbsp;·&nbsp; Mombasa &amp; Lamu Ports, Kenya
</div>""", unsafe_allow_html=True)
