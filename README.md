# ZeRO Agentic Stock Forecast

> An end-to-end agentic AI system for 30-day price target prediction on the National Stock Exchange of India — powered by a four-agent LangGraph pipeline, XGBoost + LSTM ensemble forecasting, and LLM-based forecast critique.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://glitching-gops-zer0.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=flat&logo=render)](https://agentic-stock-forecast.onrender.com/api/health)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](https://python.org)

---

## What It Does

ZeRO forecasts the 30-day price target for 53 diversified Nifty stocks
across 10 sectors. For each stock it:

- Fetches 2 years of daily OHLCV data and computes 20+ technical signals
- Scores recent news headlines using FinBERT sentiment analysis
- Pulls macroeconomic indicators: USD/INR, India VIX, Nifty momentum,
  sector relative strength, and earnings surprise
- Trains a per-stock ensemble of XGBoost (Optuna-tuned) and a PyTorch LSTM,
  combined by a Ridge regression meta-learner
- Generates a plain-English signal narrative via the Groq LLM
- Critically reviews each forecast using a Critic Agent that assigns a
  structured APPROVED / FLAGGED / REJECTED verdict
- Ranks all stocks on a composite leaderboard and refreshes daily

---

## Live Demo

| | |
|---|---|
| **Streamlit Dashboard** | https://agentic-stock-forecast.streamlit.app/ |
| **FastAPI Backend** | https://agentic-stock-forecast.onrender.com/docs |

---

## System Architecture
Data Sources (yfinance, Google News RSS)
│
▼
┌───────────────────────────────────────────────┐
│              LangGraph Agent Pipeline          │
│                                               │
│  ┌─────────────────┐  ┌────────────────────┐  │
│  │ Trading Data    │  │ External Data      │  │
│  │ Agent           │  │ Agent              │  │
│  │                 │  │                    │  │
│  │ OHLCV + 20      │  │ FinBERT sentiment  │  │
│  │ technical       │  │ + macro signals    │  │
│  │ signals         │  │ + earnings         │  │
│  └────────┬────────┘  └─────────┬──────────┘  │
│           │                     │             │
│           └──────────┬──────────┘             │
│                      ▼                        │
│           ┌─────────────────────┐             │
│           │  Forecasting Agent  │             │
│           │                     │             │
│           │  XGBoost (Optuna)   │             │
│           │  + LSTM (PyTorch)   │             │
│           │  + Ridge ensemble   │             │
│           │  + Groq narrative   │             │
│           └──────────┬──────────┘             │
│                      ▼                        │
│           ┌─────────────────────┐             │
│           │    Critic Agent     │             │
│           │                     │             │
│           │  LLM review +       │             │
│           │  APPROVED/FLAGGED/  │             │
│           │  REJECTED verdict   │             │
│           └──────────┬──────────┘             │
└──────────────────────┼───────────────────────┘
▼
Supabase PostgreSQL
│
▼
FastAPI Backend (Render) ──► Streamlit Dashboard

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| LLM | Groq API — `openai/gpt-oss-20b` |
| Sentiment | ProsusAI/FinBERT |
| ML Models | XGBoost (Optuna-tuned), PyTorch LSTM, Ridge meta-learner |
| Signal Engineering | `ta` library, pandas, scikit-learn |
| Hyperparameter Tuning | Optuna (50 trials/stock, expanding window CV) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Database | Supabase PostgreSQL |
| Scheduler | APScheduler (daily 18:30 IST) |
| Data Sources | yfinance, feedparser (Google News RSS) |
| Deployment | Render (backend), Streamlit Community Cloud (frontend) |

---

## Model Performance

Evaluated across 53 Nifty stocks using 5-fold expanding window
cross-validation — a realistic methodology that prevents data leakage
by never training on future data.

| Metric | Value |
|---|---|
| Mean MAPE | ~4.3% |
| Mean Directional Accuracy | ~85% |
| Forecast Horizon | 30 days |
| Stocks Covered | 53 |
| Sectors | 10 |

---

## Signal Library

**Technical (20):** RSI-14, MACD histogram, Bollinger Band width/upper/lower,
OBV, SMA-20, EMA-9/21/50, ATR-14, Stochastic %K, Williams %R, ROC-10,
VROC-10, 52-week proximity, Lag-1/5 returns, SMA-50 deviation,
Hurst exponent (regime detection)

**Sentiment:** FinBERT news sentiment score (Google News RSS, daily)

**Macro:** USD/INR rate, India VIX, Nifty 5d/20d returns

**Sector:** Relative momentum vs sector index over 5d, 10d, 20d windows

**Earnings:** Quarterly EPS surprise (forward-filled between reports)

---

## Project Structure
├── agents/          # LangGraph agents (Trading Data, External Data,
│                    #   Forecasting, Critic) and shared state
├── api/             # FastAPI backend — routers, schemas, dependencies
├── app/             # Streamlit multi-page frontend
│   ├── pages/       #   Leaderboard, Stock Detail, Portfolio, About
│   ├── components/  #   Reusable UI components
│   └── styles/      #   Dark navy CSS theme
├── data/            # Database setup, ticker universe, sector config
├── pipeline/        # Data fetch, signal computation, model training
│   ├── model.py     #   XGBoost with Optuna + walk-forward CV
│   ├── lstm_model.py#   PyTorch single-layer LSTM
│   ├── meta_learner.py # Ridge regression ensemble
│   └── tuning.py    #   Optuna hyperparameter search
├── tools/           # Maintenance scripts (run manually as needed)
├── models/          # Trained models — gitignored, 49MB total
│   ├── joblib/      #   XGBoost models per stock
│   ├── lstm/        #   LSTM models + scalers per stock
│   └── meta/        #   Meta-learner models per stock
├── tuned_params/    # Optuna best params per stock — gitignored
├── main.py          # FastAPI application entry point
├── scheduler.py     # Daily pipeline + weekly Optuna retune
└── requirements.txt

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/glitching-gops/Agentic-Stock-Forecast.git
cd Agentic-Stock-Forecast

# 2. Create virtual environment (Python 3.12 required)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in: DATABASE_URL, GROQ_API_KEY, HF_TOKEN, ADMIN_API_KEY

# 5. Initialise database and run pipeline
python main.py

# 6. Start the API server
uvicorn api.main:app --reload --port 8000

# 7. Launch the dashboard
streamlit run app/main.py
```

---

## How the Composite Score Works

Each stock is ranked by a weighted composite score out of 100:

| Component | Weight | Description |
|---|---|---|
| Directional Accuracy | 30 pts | % of correct up/down predictions |
| Critic Verdict | 30 pts | APPROVED=30, FLAGGED=12, REJECTED=0 |
| Forecast Upside | 25 pts | Predicted % gain (capped at 25pts) |
| Model Confidence | 15 pts | High=15, Medium=7, Low=0 |

---

## Author

**Venu Gopal Battula**
[github.com/glitching-gops](https://github.com/glitching-gops)

---

## Acknowledgements

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — financial sentiment
- [Groq](https://groq.com) — LLM inference
- [Supabase](https://supabase.com) — PostgreSQL hosting
- [Render](https://render.com) — backend hosting
- [Streamlit](https://streamlit.io) — frontend framework
