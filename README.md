# Kenya Port Throughput Forecaster

A publicly deployable Streamlit web application implementing the **Two-Stage Zero-Inflated Stacking Ensemble (Hybrid 1)** for daily cargo throughput forecasting at Mombasa and Lamu ports.

## Architecture

- **Stage 1:** LightGBM binary classifier — predicts whether a port day is active (ROC-AUC > 0.99)
- **Stage 2:** Random Forest + LightGBM + SVR base models blended by Ridge meta-learner
- **Test set performance:** R² = 0.9939, MAE = 1,777 tonnes, RMSE = 4,145 tonnes

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `model.pkl` | Trained Hybrid 1 pipeline (all models + preprocessor) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | App theme and server config |

## Deploy on Streamlit Community Cloud (free, public URL)

1. Push this folder to a **GitHub repository** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — your app gets a public URL like `https://yourname-portforecast.streamlit.app`

> **Important:** `model.pkl` must be in the repo root alongside `app.py`. GitHub has a 100 MB file limit. If `model.pkl` exceeds this, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> ```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Data source

HDX PortWatch (OCHA / IMF) — daily AIS-based port activity data for Mombasa and Lamu, Kenya.  
Period: January 2019 – November 2025 · 5,048 observations.

## Reference

Kiragu, A.W. (2025). *Machine Learning-Based Forecasting and Optimization of Port Throughput Using Daily Maritime Activity from Kenyan Ports*. Strathmore University MSc Data Science.
