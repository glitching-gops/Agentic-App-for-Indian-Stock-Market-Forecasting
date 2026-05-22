# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZeRO is an agentic AI system for 30-day price target prediction on 53 NSE India stocks. It runs a four-node LangGraph pipeline (Trading Data → External Data → Forecasting → Critic) with an XGBoost + LSTM ensemble and a Ridge regression meta-learner, exposed via FastAPI with a Streamlit frontend.

## Development Commands

All commands are run from within `stock_forecast/`.

```bash
# Activate virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run full local app (DB init → pipeline → scheduler → Streamlit)
python main.py

# API server only (after pipeline has run once)
uvicorn api.main:app --reload --port 8000

# Streamlit dashboard only
streamlit run app/main.py

# Run pipeline stages individually
python -c "from pipeline.fetch import fetch_and_store; fetch_and_store()"
python -c "from pipeline.signals import compute_and_store; compute_and_store()"
python -c "from pipeline.sentiment import fetch_and_score; fetch_and_score()"
python -c "from pipeline.macro import fetch_and_store; fetch_and_store()"
python -c "from pipeline.model import train_and_forecast; train_and_forecast()"

# Run agent graph for a single ticker
python -c "from agents.graph import run_graph; run_graph('RELIANCE.NS')"
```

## Required Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL URL for prod, defaults to local SQLite |
| `GROQ_API_KEY` | LLM inference via Groq |
| `GROQ_MODEL` | Defaults to `openai/gpt-oss-20b` |
| `ADMIN_API_KEY` | Protects `/api/admin` endpoints |
| `GITHUB_TOKEN` | Render only — downloads trained models from `model-store` branch |
| `API_BASE_URL` | Streamlit → FastAPI base URL, defaults to `http://localhost:8000` |

## Architecture

### Data flow

```
yfinance / Google News RSS
        │
        ▼
LangGraph Pipeline (agents/graph.py)
  ├─ trading_data_node  — OHLCV + 20 technical signals (parallel)
  ├─ external_data_node — FinBERT sentiment + macro (parallel)
  ├─ forecasting_node   — XGBoost + LSTM + Ridge ensemble + Groq narrative
  └─ critic_node        — LLM quality gate → APPROVED/FLAGGED/REJECTED
        │
        ▼
Database (SQLite dev / Supabase PostgreSQL prod)
        │
        ├─► FastAPI backend  (api/)      ← Render
        └─► Streamlit dashboard (app/)   ← Streamlit Community Cloud
```

### Key modules

- **`agents/state.py`** — `AgentState` TypedDict; the single shared state struct passed through the LangGraph nodes.
- **`agents/graph.py`** — `build_graph()` compiles the StateGraph; `run_graph(ticker)` executes it and calls `save_forecast_to_db()`; `compute_composite_score()` calculates the 100-pt leaderboard score.
- **`pipeline/model.py`** — XGBoost training with Optuna-tuned hyperparameters (50 trials, expanding-window CV).
- **`pipeline/lstm_model.py`** — Single-layer PyTorch LSTM.
- **`pipeline/meta_learner.py`** — Ridge regression that combines XGBoost + LSTM predictions.
- **`pipeline/tuning.py`** — Optuna hyperparameter search; saves best params to `tuned_params/<ticker>.json`.
- **`data/db.py`** — Single SQLAlchemy engine (singleton); `init_db()` creates all tables with safe `ALTER TABLE` migrations for new columns. Normalises `postgres://` → `postgresql+psycopg2://` automatically.
- **`data/tickers.py`** — 53-stock TICKERS dict; `get_company()` / `get_sector()` helpers.
- **`scheduler.py`** — APScheduler: daily pipeline at 18:30 IST, weekly Optuna retune on Sunday 02:00 IST.
- **`api/main.py`** — FastAPI app with lifespan; starts scheduler on startup.
- **`app/main.py`** — Streamlit multi-page entry point.

### Database tables

`ohlcv` · `signals` · `sentiment` · `macro` · `forecasts` · `leaderboard` · `model_metadata`

Primary keys are `(date, ticker)` on time-series tables; `ticker` on `leaderboard` and `model_metadata`. The leaderboard upsert uses `ON CONFLICT (ticker) DO UPDATE` (PostgreSQL) with an `INSERT OR REPLACE` fallback for SQLite.

### Trained model artifacts

Stored under `models/` (gitignored, ~49 MB total):
- `models/joblib/<ticker>.joblib` — XGBoost models
- `models/lstm/<ticker>_model.pt` + `<ticker>_scaler.pkl` — LSTM weights and scalers
- `models/meta/<ticker>_meta.pkl` — Ridge meta-learner

On Render, `render_start.sh` downloads `models_store.zip` from the `model-store` branch using `GITHUB_TOKEN` if `models/joblib/RELIANCE.NS.joblib` is absent.

### Composite leaderboard score (0–100)

| Component | Max pts | Notes |
|---|---|---|
| Directional accuracy | 30 | Linear 0–100% → 0–30 pts |
| Critic verdict | 30 | APPROVED=30, FLAGGED=12, REJECTED=0 |
| Forecast upside | 25 | `upside_pct × 1.5`, capped at 25 |
| Model confidence | 15 | High=15, Medium=7, Low=0 |

## Deployment

- **Backend:** Render — entry point is `render_start.sh` (downloads models, inits DB, starts uvicorn).
- **Frontend:** Streamlit Community Cloud — entry point is `app/main.py`; reads from `API_BASE_URL`.

## Maintenance Scripts (`tools/`)

Run manually as needed, not part of the application:
- `migrate_to_supabase.py` — one-time SQLite → Supabase migration
- `data_check.py` — sanity-check DB contents
- `verify_endpoints.py` — health-check FastAPI endpoints
