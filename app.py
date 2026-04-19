import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Kenya Port Throughput Forecaster",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif;background:#0f1117;color:#e8eaf0}
[data-testid="stHeader"],[data-testid="stToolbar"],footer{display:none!important}
.hero{background:linear-gradient(135deg,#0d4f2e 0%,#157a46 55%,#1fa055 100%);border-radius:20px;padding:2.6rem 2.8rem 2.2rem;margin-bottom:1.8rem;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;right:-60px;top:-60px;width:340px;height:340px;border-radius:50%;background:rgba(255,255,255,0.06)}
.hero h1{color:#fff;font-size:2.1rem;font-weight:700;margin:0 0 .55rem;line-height:1.2}
.hero p{color:rgba(255,255,255,.82);font-size:.97rem;margin:0 0 1.2rem;max-width:600px;line-height:1.6}
.badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);color:#fff;border-radius:40px;padding:.28rem .9rem;font-size:.78rem;font-weight:500;margin:0 .35rem 0 0}
.metric-row{display:flex;gap:1rem;margin-bottom:1.6rem}
.mcard{flex:1;background:#1a1d27;border:1px solid #2a2d3a;border-radius:14px;padding:1.3rem 1.5rem}
.mcard .val{font-size:2rem;font-weight:700;color:#4ade80;line-height:1;margin-bottom:.3rem}
.mcard .val.amber{color:#fbbf24}
.mcard .lbl{font-size:.76rem;font-weight:500;letter-spacing:.08em;color:#6b7280;text-transform:uppercase}
.section-title{display:flex;align-items:center;gap:.55rem;font-size:1.15rem;font-weight:600;color:#e8eaf0;margin:0 0 1rem;padding-bottom:.5rem;border-bottom:1px solid #2a2d3a}
.input-panel{background:#1a1d27;border:1px solid #2a2d3a;border-radius:14px;padding:1.3rem 1.4rem .8rem;height:100%}
.input-panel .panel-title{font-size:.82rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#9ca3af;margin-bottom:.9rem;padding-bottom:.5rem;border-bottom:1px solid #2a2d3a}
[data-testid="stNumberInput"] label,[data-testid="stSelectbox"] label,[data-testid="stDateInput"] label{color:#9ca3af!important;font-size:.83rem!important;font-weight:500!important}
[data-testid="stNumberInput"] input,[data-testid="stSelectbox"]>div>div,[data-testid="stDateInput"] input{background:#0f1117!important;border:1px solid #2a2d3a!important;color:#e8eaf0!important;border-radius:8px!important}
[data-testid="stButton"]>button[kind="primary"]{background:linear-gradient(135deg,#157a46,#1fa055)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:1rem!important;padding:.7rem 1.5rem!important;width:100%;letter-spacing:.03em}
.result-hero{background:linear-gradient(135deg,#0d4f2e 0%,#157a46 100%);border-radius:18px;padding:2rem 2rem 1.6rem;text-align:center;margin-bottom:1rem}
.result-hero .rlabel{color:rgba(255,255,255,.75);font-size:.85rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem}
.result-hero .big-num{color:#fff;font-size:3.6rem;font-weight:800;line-height:1;margin-bottom:.3rem}
.result-hero .unit{color:rgba(255,255,255,.7);font-size:1rem}
.result-hero .rsub{color:rgba(255,255,255,.6);font-size:.82rem;margin-top:.6rem}
.result-zero{background:#1a1d27;border:2px solid #374151;border-radius:18px;padding:2rem 2rem 1.6rem;text-align:center;margin-bottom:1rem}
.result-zero .big-num{color:#6b7280;font-size:3.6rem;font-weight:800;line-height:1}
.result-zero .rlabel{color:#6b7280;font-size:.85rem;text-transform:uppercase;letter-spacing:.06em}
.stage-pill{display:inline-flex;align-items:center;gap:.35rem;background:rgba(21,122,70,.18);border:1px solid rgba(21,122,70,.4);color:#4ade80;border-radius:40px;padding:.22rem .85rem;font-size:.76rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.7rem}
.conf-wrap{margin:.8rem 0 1.2rem}
.conf-label{display:flex;justify-content:space-between;font-size:.83rem;color:#9ca3af;margin-bottom:.4rem}
.conf-track{background:#2a2d3a;border-radius:99px;height:10px;overflow:hidden}
.conf-fill{height:100%;border-radius:99px}
.bench-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:.8rem}
.bench-item{background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;padding:.6rem .8rem}
.bench-item .bval{font-size:1rem;font-weight:700;color:#e8eaf0}
.bench-item .blbl{font-size:.72rem;color:#6b7280;margin-top:.1rem}
.info-strip{background:rgba(21,122,70,.12);border:1px solid rgba(21,122,70,.3);border-radius:10px;padding:.7rem 1rem;font-size:.86rem;color:#4ade80;margin-top:.8rem}
.how-card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:14px;padding:1.3rem 1.4rem;height:100%}
.how-card .how-num{font-size:.72rem;font-weight:700;letter-spacing:.1em;color:#4ade80;text-transform:uppercase;margin-bottom:.4rem}
.how-card h4{margin:0 0 .5rem;font-size:1rem;color:#e8eaf0}
.how-card p{margin:0;font-size:.86rem;color:#9ca3af;line-height:1.6}
.skip-box{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:1.5rem;text-align:center;color:#6b7280;margin-top:1rem}
</style>
""", unsafe_allow_html=True)

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
        portcalls_container=pc_cont,portcalls_dry_bulk=pc_bulk,
        portcalls_general_cargo=pc_gen,portcalls_roro=pc_roro,portcalls_tanker=pc_tank,
        total_portcalls=total_pc,
        import_container=im_cont,import_dry_bulk=im_bulk,import_general_cargo=im_gen,
        import_roro=im_roro,import_tanker=im_tank,
        export_container=ex_cont,export_dry_bulk=ex_bulk,export_general_cargo=ex_gen,
        export_roro=ex_roro,export_tanker=ex_tank,
        tanker_call_ratio=pc_tank/total_pc if total_pc>0 else 0,
        container_call_ratio=pc_cont/total_pc if total_pc>0 else 0,
        bulk_call_ratio=pc_bulk/total_pc if total_pc>0 else 0,
        import_export_ratio=total_im/total_tp if total_tp>0 else 0.5,
        month=date.month,quarter=(date.month-1)//3+1,
        day_of_week=date.weekday(),is_weekend=int(date.weekday() in [5,6]),
    )
    return pd.DataFrame([row])[FEATURES]

def predict(row_df):
    enc   = preprocessor.transform(row_df)
    p_act = clf.predict_proba(enc)[0,1]
    active= clf.predict(enc)[0]
    if active==0:
        return 0.0,p_act,None,None,None
    enc_sc= scaler.transform(enc)
    p_rf  = float(rf.predict(enc)[0])
    p_lgb = float(lgb_reg.predict(enc)[0])
    p_svr = float(svr.predict(enc_sc)[0])
    meta  = np.array([[p_rf,p_lgb,p_svr,p_act]])
    final = float(np.clip(ridge.predict(meta)[0],0,None))
    return final,p_act,p_rf,p_lgb,p_svr

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚢 Kenya Port Throughput Forecaster</h1>
  <p>Predict daily cargo throughput for Mombasa and Lamu using the Hybrid 1
     Two-Stage Zero-Inflated Stacking Ensemble — trained on 5,048 AIS-based
     daily observations from January 2019 to November 2025.</p>
  <span class="badge">+ Hybrid 1 Novel Algorithm</span>
  <span class="badge">+ HDX PortWatch AIS Data</span>
  <span class="badge">+ Zero-Inflated Stacking</span>
  <span class="badge">+ Strathmore University MSc DSA</span>
</div>
""", unsafe_allow_html=True)

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="mcard"><div class="val">{metrics['r2']:.4f}</div><div class="lbl">R² Score (test set)</div></div>
  <div class="mcard"><div class="val">{metrics['mae']:,.0f} t</div><div class="lbl">MAE — Tonnes</div></div>
  <div class="mcard"><div class="val">{metrics['rmse']:,.0f} t</div><div class="lbl">RMSE — Tonnes</div></div>
  <div class="mcard"><div class="val amber">&gt; 0.99</div><div class="lbl">Stage 1 ROC-AUC</div></div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Enter Port Activity Details</div>', unsafe_allow_html=True)
st.caption("Fill in all three sections then click **Run Forecast**.")

col_a, col_b, col_c = st.columns(3, gap="medium")

with col_a:
    st.markdown('<div class="input-panel"><div class="panel-title">🚢 Port & Vessel Calls</div>', unsafe_allow_html=True)
    port    = st.selectbox("Port", ["Mombasa","Lamu"])
    date    = st.date_input("Forecast date", value=datetime.date.today())
    st.markdown("---")
    pc_cont = st.number_input("🟦 Container calls",     min_value=0,max_value=20,value=1,step=1)
    pc_tank = st.number_input("🟠 Tanker calls",        min_value=0,max_value=20,value=1,step=1)
    pc_bulk = st.number_input("🟡 Dry bulk calls",      min_value=0,max_value=20,value=0,step=1)
    pc_gen  = st.number_input("🟢 General cargo calls", min_value=0,max_value=20,value=0,step=1)
    pc_roro = st.number_input("🔴 Ro-Ro calls",         min_value=0,max_value=20,value=0,step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="input-panel"><div class="panel-title">📦 Import Volumes (tonnes)</div>', unsafe_allow_html=True)
    im_tank = st.number_input("🟠 Tanker imports",        min_value=0.0,value=50000.0,step=1000.0,format="%.0f")
    im_bulk = st.number_input("🟡 Dry bulk imports",      min_value=0.0,value=20000.0,step=1000.0,format="%.0f")
    im_cont = st.number_input("🟦 Container imports",     min_value=0.0,value=10000.0,step=1000.0,format="%.0f")
    im_gen  = st.number_input("🟢 General cargo imports", min_value=0.0,value=0.0,    step=1000.0,format="%.0f")
    im_roro = st.number_input("🔴 Ro-Ro imports",         min_value=0.0,value=0.0,    step=1000.0,format="%.0f")
    st.info("💡 **import_tanker** is the strongest predictor (r = 0.761) — set it accurately.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="input-panel"><div class="panel-title">📤 Export Volumes (tonnes)</div>', unsafe_allow_html=True)
    ex_tank = st.number_input("🟠 Tanker exports",        min_value=0.0,value=0.0,   step=1000.0,format="%.0f")
    ex_bulk = st.number_input("🟡 Dry bulk exports",      min_value=0.0,value=5000.0,step=1000.0,format="%.0f")
    ex_cont = st.number_input("🟦 Container exports",     min_value=0.0,value=3000.0,step=1000.0,format="%.0f")
    ex_gen  = st.number_input("🟢 General cargo exports", min_value=0.0,value=0.0,   step=1000.0,format="%.0f")
    ex_roro = st.number_input("🔴 Ro-Ro exports",         min_value=0.0,value=0.0,   step=1000.0,format="%.0f")
    st.markdown("**Historical benchmarks — Mombasa**")
    st.markdown("""
<div class="bench-grid">
  <div class="bench-item"><div class="bval">63,855 t</div><div class="blbl">July — lowest month</div></div>
  <div class="bench-item"><div class="bval">74,951 t</div><div class="blbl">October — peak month</div></div>
  <div class="bench-item"><div class="bval">69,012 t</div><div class="blbl">2019 annual average</div></div>
  <div class="bench-item"><div class="bval">78,092 t</div><div class="blbl">2025 annual average</div></div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2,1,2])
with btn_col:
    run = st.button("🔮 Run Forecast", type="primary", use_container_width=True)

# ── RESULTS ──────────────────────────────────────────────────────────────────
if run:
    row_df = build_row(port,date,pc_cont,pc_bulk,pc_gen,pc_roro,pc_tank,
                       im_cont,im_bulk,im_gen,im_roro,im_tank,
                       ex_cont,ex_bulk,ex_gen,ex_roro,ex_tank)
    final,p_act,p_rf,p_lgb,p_svr = predict(row_df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Forecast Results</div>', unsafe_allow_html=True)

    res_col, s1_col, s2_col = st.columns([1.1,1,1.2], gap="medium")

    with res_col:
        if p_act >= 0.5:
            pct = (final/69000-1)*100
            st.markdown(f"""
<div class="result-hero">
  <div class="rlabel">Predicted Total Throughput</div>
  <div class="big-num">{final:,.0f}</div>
  <div class="unit">tonnes / day</div>
  <div class="rsub">{port} &nbsp;·&nbsp; {date.strftime('%d %b %Y')}</div>
</div>
<div class="info-strip">📈 {abs(pct):.1f}% {"above" if pct>=0 else "below"} the Mombasa historical average (69,000 t)</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-zero">
  <div class="rlabel">Predicted Total Throughput</div>
  <div class="big-num">0</div>
  <div style="color:#6b7280;font-size:.95rem;margin-top:.3rem">tonnes / day</div>
  <div style="color:#9ca3af;font-size:.82rem;margin-top:.8rem">{port} · {date.strftime('%d %b %Y')}<br>No active port day detected by Stage 1.</div>
</div>""", unsafe_allow_html=True)

        labels   = ["Container","Dry Bulk","General","Ro-Ro","Tanker"]
        im_vals  = [im_cont,im_bulk,im_gen,im_roro,im_tank]
        ex_vals  = [ex_cont,ex_bulk,ex_gen,ex_roro,ex_tank]
        tot_vals = [i+e for i,e in zip(im_vals,ex_vals)]
        donut = go.Figure(go.Pie(
            labels=labels,values=tot_vals,hole=0.58,
            marker_colors=["#3b82f6","#f59e0b","#10b981","#f43f5e","#8b5cf6"],
            textinfo="percent",showlegend=True,
            hovertemplate="%{label}: %{value:,.0f} t<extra></extra>",
        ))
        donut.update_layout(
            title=dict(text="Cargo mix (input volumes)",font=dict(color="#9ca3af",size=13)),
            height=260,margin=dict(l=0,r=0,t=36,b=0),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            legend=dict(font=dict(color="#9ca3af",size=11),bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(donut, use_container_width=True)

    with s1_col:
        st.markdown('<div class="stage-pill">● Stage 1 — Zero Classifier</div>', unsafe_allow_html=True)
        conf_pct  = round(p_act*100,1)
        bar_color = "#4ade80" if p_act>=0.5 else "#6b7280"
        status    = "🟢 Active" if p_act>=0.5 else "🔴 Inactive"
        st.markdown(f"""
<div class="conf-wrap">
  <div class="conf-label">
    <span>Activity probability</span>
    <span style="color:#e8eaf0;font-weight:600">{conf_pct}%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"**Port status:** {status}")
        st.caption("Threshold: 50% · LightGBM · class_weight='balanced'")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",value=conf_pct,
            gauge=dict(
                axis=dict(range=[0,100],tickcolor="#4b5563",tickfont=dict(color="#6b7280"),ticksuffix="%"),
                bar=dict(color=bar_color),
                bgcolor="#0f1117",bordercolor="#2a2d3a",
                steps=[dict(range=[0,50],color="#1a1d27"),dict(range=[50,100],color="#0d2e1a")],
                threshold=dict(line=dict(color="#4ade80",width=2),value=50),
            ),
            number=dict(suffix="%",font=dict(size=30,color="#e8eaf0")),
        ))
        fig_g.update_layout(height=220,margin=dict(l=20,r=20,t=20,b=10),
                            paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#9ca3af"))
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("Handles Lamu's 96.6% zero-throughput rate. If inactive → forecast = 0 t and Stage 2 is skipped.")

    with s2_col:
        st.markdown('<div class="stage-pill">● Stage 2 — Stacking Ensemble</div>', unsafe_allow_html=True)
        if p_rf is not None:
            bar_fig = go.Figure(go.Bar(
                x=[p_rf,p_lgb,p_svr],y=["Random Forest","LightGBM","SVR (RBF)"],
                orientation='h',
                marker_color=["#3b82f6","#10b981","#f59e0b"],
                text=[f"{v:,.0f} t" for v in [p_rf,p_lgb,p_svr]],
                textposition="outside",textfont=dict(color="#9ca3af",size=12),
            ))
            bar_fig.update_layout(
                title=dict(text="Base model predictions",font=dict(color="#9ca3af",size=13)),
                height=210,margin=dict(l=0,r=70,t=36,b=10),
                xaxis=dict(showgrid=False,showticklabels=False,color="#6b7280"),
                yaxis=dict(color="#9ca3af",tickfont=dict(size=12)),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            w = ridge.coef_
            wfig = go.Figure(go.Bar(
                x=["RF","LightGBM","SVR","P(active)"],y=w,
                marker_color=["#3b82f6","#10b981","#f59e0b","#a78bfa"],
                text=[f"{v:.3f}" for v in w],textposition="outside",
                textfont=dict(color="#9ca3af",size=11),
            ))
            wfig.update_layout(
                title=dict(text="Ridge meta-learner weights",font=dict(color="#9ca3af",size=13)),
                height=220,margin=dict(l=0,r=20,t=36,b=10),
                xaxis=dict(color="#6b7280",tickfont=dict(color="#9ca3af")),
                yaxis=dict(showgrid=True,gridcolor="#2a2d3a",color="#6b7280",zeroline=True,zerolinecolor="#374151"),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,
            )
            st.plotly_chart(wfig, use_container_width=True)
            st.caption("RidgeCV meta-learner trained on leak-free out-of-fold predictions (5-fold CV).")
        else:
            st.markdown("""
<div class="skip-box">
  <div style="font-size:2rem;margin-bottom:.5rem">⏭</div>
  <div style="font-weight:600;margin-bottom:.3rem">Stage 2 skipped</div>
  <div style="font-size:.85rem">Port classified as inactive.<br>Final forecast = 0 tonnes.</div>
</div>""", unsafe_allow_html=True)

# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 How it works</div>', unsafe_allow_html=True)
hw1,hw2,hw3 = st.columns(3,gap="medium")
with hw1:
    st.markdown("""<div class="how-card">
<div class="how-num">Stage 1</div><h4>Zero-Activity Classifier</h4>
<p>A LightGBM binary classifier predicts whether a port day will record any activity.
Directly addresses Lamu's 96.6% zero-throughput rate through class-balanced training.
Achieves ROC-AUC > 0.99.</p></div>""", unsafe_allow_html=True)
with hw2:
    st.markdown("""<div class="how-card">
<div class="how-num">Stage 2</div><h4>Heterogeneous Stacking Ensemble</h4>
<p>Random Forest · LightGBM · SVR (RBF kernel) are trained exclusively on active days.
A Ridge meta-learner blends their out-of-fold predictions, combining complementary error profiles.</p></div>""", unsafe_allow_html=True)
with hw3:
    st.markdown("""<div class="how-card">
<div class="how-num">Features</div><h4>22-Variable AIS Feature Set</h4>
<p>Vessel call counts · Individual cargo import/export volumes (import_tanker r = 0.761)
· Cargo composition ratios · Temporal dummies. Trained on 5,048 records from HDX PortWatch (2019–2025).</p></div>""", unsafe_allow_html=True)

st.markdown("""<br><div style="text-align:center;color:#4b5563;font-size:.8rem;padding:1rem 0">
Kiragu, A.W. (2025) · Strathmore University MSc Data Science &amp; Analytics · HDX PortWatch data (OCHA / IMF)
</div>""", unsafe_allow_html=True)
