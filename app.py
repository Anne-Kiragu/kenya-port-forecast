import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Port Throughput Forecaster",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f8f9fc; }
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1a3c6e; }
.metric-label { font-size: 0.85rem; color: #6b7280; margin-top: 0.2rem; }
.prediction-box {
    background: linear-gradient(135deg, #1a3c6e 0%, #2563eb 100%);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.prediction-box h1 { color: white; font-size: 3.5rem; margin: 0; }
.prediction-box p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1.1rem; }
.zero-box {
    background: linear-gradient(135deg, #374151 0%, #6b7280 100%);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.zero-box h1 { color: white; font-size: 3.5rem; margin: 0; }
.zero-box p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1.1rem; }
.stage-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.warning-box {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    color: #92400e;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

artefacts = load_model()
preprocessor  = artefacts["preprocessor"]
clf           = artefacts["clf"]
scaler        = artefacts["scaler"]
rf            = artefacts["rf"]
lgb_reg       = artefacts["lgb_reg"]
svr           = artefacts["svr"]
ridge         = artefacts["ridge"]
FEATURES      = artefacts["FEATURES"]
metrics       = artefacts["metrics"]

# ── Feature builder ──────────────────────────────────────────────────────
def build_single_row(portname, date, pc_cont, pc_bulk, pc_gen, pc_roro, pc_tank,
                     im_cont, im_bulk, im_gen, im_roro, im_tank,
                     ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank):
    total_pc = pc_cont + pc_bulk + pc_gen + pc_roro + pc_tank
    total_im = im_cont + im_bulk + im_gen + im_roro + im_tank
    total_ex = ex_cont + ex_bulk + ex_gen + ex_roro + ex_tank
    total_tp = total_im + total_ex

    row = {
        "portname": portname,
        "portcalls_container":     pc_cont,
        "portcalls_dry_bulk":      pc_bulk,
        "portcalls_general_cargo": pc_gen,
        "portcalls_roro":          pc_roro,
        "portcalls_tanker":        pc_tank,
        "total_portcalls":         total_pc,
        "import_container":        im_cont,
        "import_dry_bulk":         im_bulk,
        "import_general_cargo":    im_gen,
        "import_roro":             im_roro,
        "import_tanker":           im_tank,
        "export_container":        ex_cont,
        "export_dry_bulk":         ex_bulk,
        "export_general_cargo":    ex_gen,
        "export_roro":             ex_roro,
        "export_tanker":           ex_tank,
        "tanker_call_ratio":    pc_tank/total_pc if total_pc>0 else 0,
        "container_call_ratio": pc_cont/total_pc if total_pc>0 else 0,
        "bulk_call_ratio":      pc_bulk/total_pc if total_pc>0 else 0,
        "import_export_ratio":  total_im/total_tp if total_tp>0 else 0.5,
        "month":       date.month,
        "quarter":     (date.month-1)//3+1,
        "day_of_week": date.weekday(),
        "is_weekend":  int(date.weekday() in [5,6]),
    }
    return pd.DataFrame([row])[FEATURES]

def predict(row_df):
    enc     = preprocessor.transform(row_df)
    p_act   = clf.predict_proba(enc)[0,1]
    pred_cls= clf.predict(enc)[0]

    if pred_cls == 0:
        return 0.0, p_act, None, None, None, None

    enc_sc  = scaler.transform(enc)
    p_rf    = rf.predict(enc)[0]
    p_lgb   = lgb_reg.predict(enc)[0]
    p_svr   = svr.predict(enc_sc)[0]
    meta    = np.array([[p_rf, p_lgb, p_svr, p_act]])
    final   = float(np.clip(ridge.predict(meta)[0], 0, None))
    return final, p_act, p_rf, p_lgb, p_svr, meta

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png", width=60)
    st.title("🚢 Port Forecaster")
    st.caption("Hybrid 1 — Two-Stage Zero-Inflated Stacking Ensemble")
    st.markdown("---")

    port = st.selectbox("**Port**", ["Mombasa", "Lamu"], index=0)
    date = st.date_input("**Forecast date**", value=datetime.date.today())

    st.markdown("---")
    st.markdown("### 🚢 Vessel call counts")
    pc_cont = st.number_input("Container calls",  min_value=0, max_value=20, value=1, step=1)
    pc_tank = st.number_input("Tanker calls",     min_value=0, max_value=20, value=1, step=1)
    pc_bulk = st.number_input("Dry bulk calls",   min_value=0, max_value=20, value=0, step=1)
    pc_gen  = st.number_input("General cargo calls", min_value=0, max_value=20, value=0, step=1)
    pc_roro = st.number_input("Ro-Ro calls",      min_value=0, max_value=20, value=0, step=1)

    st.markdown("---")
    st.markdown("### 📦 Import volumes (tonnes)")
    im_tank = st.number_input("Tanker imports",       min_value=0.0, value=50000.0, step=1000.0, format="%.0f")
    im_bulk = st.number_input("Dry bulk imports",     min_value=0.0, value=20000.0, step=1000.0, format="%.0f")
    im_cont = st.number_input("Container imports",    min_value=0.0, value=10000.0, step=1000.0, format="%.0f")
    im_gen  = st.number_input("General cargo imports",min_value=0.0, value=0.0,     step=1000.0, format="%.0f")
    im_roro = st.number_input("Ro-Ro imports",        min_value=0.0, value=0.0,     step=1000.0, format="%.0f")

    st.markdown("---")
    st.markdown("### 📤 Export volumes (tonnes)")
    ex_tank = st.number_input("Tanker exports",       min_value=0.0, value=0.0,    step=1000.0, format="%.0f")
    ex_bulk = st.number_input("Dry bulk exports",     min_value=0.0, value=5000.0, step=1000.0, format="%.0f")
    ex_cont = st.number_input("Container exports",    min_value=0.0, value=3000.0, step=1000.0, format="%.0f")
    ex_gen  = st.number_input("General cargo exports",min_value=0.0, value=0.0,   step=1000.0, format="%.0f")
    ex_roro = st.number_input("Ro-Ro exports",        min_value=0.0, value=0.0,   step=1000.0, format="%.0f")

    st.markdown("---")
    forecast_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

# ── Main content ──────────────────────────────────────────────────────────
st.title("Kenya Port Throughput Forecaster")
st.caption("Machine Learning-Based Daily Port Cargo Throughput Prediction · Mombasa & Lamu · Hybrid 1 Architecture")

# ── Model performance banner ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['r2']:.4f}</div>
        <div class="metric-label">Model R² (test set)</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['mae']:,.0f}</div>
        <div class="metric-label">MAE — tonnes</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['rmse']:,.0f}</div>
        <div class="metric-label">RMSE — tonnes</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">>0.99</div>
        <div class="metric-label">Stage 1 ROC-AUC</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Prediction output ────────────────────────────────────────────────────
if forecast_btn:
    row_df = build_single_row(
        port, date,
        pc_cont, pc_bulk, pc_gen, pc_roro, pc_tank,
        im_cont, im_bulk, im_gen, im_roro, im_tank,
        ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank
    )

    final, p_act, p_rf, p_lgb, p_svr, meta = predict(row_df)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        # Stage 1 result
        st.markdown('<span class="stage-badge">Stage 1 — Zero-Activity Classifier</span>', unsafe_allow_html=True)
        gauge_col, text_col = st.columns([1, 1])
        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(p_act*100, 1),
                title={"text": "P(Active)", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": "#2563eb" if p_act >= 0.5 else "#6b7280"},
                    "steps": [
                        {"range": [0, 50],  "color": "#f3f4f6"},
                        {"range": [50, 100],"color": "#dbeafe"}
                    ],
                    "threshold": {"line": {"color": "#1d4ed8", "width": 3}, "value": 50}
                },
                number={"suffix": "%", "font": {"size": 28}}
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with text_col:
            status = "🟢 Active" if p_act >= 0.5 else "🔴 Inactive"
            st.markdown(f"**Port status:** {status}")
            st.markdown(f"**Confidence:** {p_act*100:.1f}%")
            if p_act >= 0.5:
                st.markdown("Active-day regime → Stage 2 regression active")
            else:
                st.markdown("Zero-throughput predicted — no cargo movement expected")

        # Stage 2 + final prediction
        st.markdown("---")
        if p_act >= 0.5 and p_rf is not None:
            st.markdown('<span class="stage-badge">Stage 2 — Stacking Ensemble</span>', unsafe_allow_html=True)

            # Base model breakdown
            base_fig = go.Figure(go.Bar(
                x=["Random Forest", "LightGBM", "SVR"],
                y=[p_rf, p_lgb, p_svr],
                marker_color=["#3b82f6","#10b981","#f59e0b"],
                text=[f"{v:,.0f}" for v in [p_rf, p_lgb, p_svr]],
                textposition="outside"
            ))
            base_fig.update_layout(
                title="Base model predictions (tonnes)",
                height=280,
                margin=dict(l=10,r=10,t=40,b=10),
                yaxis_title="Tonnes",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            base_fig.update_yaxis(gridcolor="#f3f4f6")
            st.plotly_chart(base_fig, use_container_width=True)

            # Ridge weights
            w = ridge.coef_
            st.markdown(f"""**Ridge meta-learner weights:** RF `{w[0]:.3f}` · LGB `{w[1]:.3f}` · SVR `{w[2]:.3f}` · P(active) `{w[3]:.5f}`""")

        # Final forecast box
        st.markdown("---")
        if p_act >= 0.5:
            st.markdown(f"""<div class="prediction-box">
                <p>Predicted total throughput — {port} · {date.strftime('%d %b %Y')}</p>
                <h1>{final:,.0f}</h1>
                <p>tonnes</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="zero-box">
                <p>Predicted total throughput — {port} · {date.strftime('%d %b %Y')}</p>
                <h1>0</h1>
                <p>tonnes — no active port day detected</p>
            </div>""", unsafe_allow_html=True)

    with right_col:
        # Cargo breakdown donut
        labels = ["Container","Dry Bulk","General Cargo","Ro-Ro","Tanker"]
        im_vals = [im_cont, im_bulk, im_gen, im_roro, im_tank]
        ex_vals = [ex_cont, ex_bulk, ex_gen, ex_roro, ex_tank]
        total_vals = [i+e for i,e in zip(im_vals, ex_vals)]

        donut = go.Figure(go.Pie(
            labels=labels, values=total_vals,
            hole=0.55,
            marker_colors=["#3b82f6","#f59e0b","#10b981","#f43f5e","#8b5cf6"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,.0f} tonnes<extra></extra>"
        ))
        donut.update_layout(
            title="Cargo mix (input volumes)",
            height=280, margin=dict(l=10,r=10,t=40,b=10),
            showlegend=False, paper_bgcolor="white"
        )
        st.plotly_chart(donut, use_container_width=True)

        # Input summary table
        st.markdown("**Input summary**")
        summary = pd.DataFrame({
            "Cargo type": labels,
            "Portcalls": [pc_cont, pc_bulk, pc_gen, pc_roro, pc_tank],
            "Imports (t)": [f"{v:,.0f}" for v in im_vals],
            "Exports (t)": [f"{v:,.0f}" for v in ex_vals],
        })
        st.dataframe(summary.set_index("Cargo type"), use_container_width=True)

        # Benchmarks
        st.markdown("---")
        st.markdown("**Mombasa benchmarks (historical)**")
        bench_data = {
            "Period": ["July (lowest)", "Oct (peak)", "2019 avg", "2025 avg"],
            "Mean throughput": ["63,855 t", "74,951 t", "69,012 t", "78,092 t"]
        }
        st.table(pd.DataFrame(bench_data).set_index("Period"))

        if p_act >= 0.5 and final > 0:
            pct_vs_avg = (final / 69000 - 1) * 100
            direction  = "above" if pct_vs_avg >= 0 else "below"
            st.info(f"Forecast is **{abs(pct_vs_avg):.1f}%** {direction} the historical Mombasa average (69,000 t).")

else:
    # Landing state
    st.info("👈 Configure vessel activity inputs in the sidebar and click **Generate Forecast** to run the model.")

    st.markdown("---")
    st.markdown("### How it works")
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.markdown("""**Stage 1 — Zero Classifier**
LightGBM binary classifier predicts whether the port will have any activity.
Handles Lamu's 96.6% zero-throughput rate through class-balanced training.
Output: P(active) probability.""")
    with arch_col2:
        st.markdown("""**Stage 2 — Stacking Ensemble**
Three diverse base regressors trained on active days only:
Random Forest · LightGBM · SVR (RBF kernel).
Ridge meta-learner blends outputs using out-of-fold predictions.""")
    with arch_col3:
        st.markdown("""**22-Variable Feature Set**
Vessel call counts · Individual cargo import/export volumes · Cargo composition ratios · Temporal features.
import_tanker is the strongest predictor (r = 0.761).""")

    st.markdown("---")
    st.markdown("### About the data")
    st.markdown("""
| | |
|---|---|
| **Source** | HDX PortWatch (OCHA / IMF AIS platform) |
| **Ports** | Mombasa (KEMBA) · Lamu (KELAU) |
| **Period** | January 2019 – November 2025 |
| **Observations** | 5,048 daily records |
| **Target** | Total daily cargo throughput (tonnes) |
| **Reference** | Kiragu, A.W. — Strathmore University MSc Data Science |
""")
