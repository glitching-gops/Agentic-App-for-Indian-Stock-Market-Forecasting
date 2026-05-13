# Agentic Stock Forecast Prototype

A production-ready stock market forecasting system using a LangGraph multi-agent architecture, combining XGBoost and LSTM models via a Ridge meta-learner.

## Project Structure

```
stock_forecast/
├── agents/         # LangGraph multi-agent pipeline
├── api/            # FastAPI backend with routers and schemas
├── app/            # Streamlit multi-page frontend
│   ├── pages/      # Leaderboard, Stock Detail, Portfolio, About
│   ├── components/ # Reusable UI components
│   └── styles/     # Custom CSS theme
├── data/           # Database setup and ticker configuration
├── pipeline/       # Data fetch, signal computation, model training
│   ├── lstm_model.py    # PyTorch LSTM model
│   ├── meta_learner.py  # Ridge regression ensemble
│   ├── model.py         # XGBoost training and inference
│   └── tuning.py        # Optuna hyperparameter tuning
├── tools/          # Maintenance scripts (run manually as needed)
├── models/         # Trained models — gitignored, generated locally
│   ├── joblib/     # XGBoost models per stock
│   ├── lstm/       # LSTM models and scalers per stock
│   └── meta/       # Meta-learner models per stock
├── tuned_params/   # Optuna best params per stock — gitignored
├── main.py         # FastAPI entry point
├── scheduler.py    # APScheduler daily pipeline job
└── requirements.txt
```

## Features
- **Multi-Agent Architecture**: LangGraph-powered workflow with External Data, Trading Data, Forecasting, and Critic agents.
- **Ensemble Modeling**: Combines XGBoost (tabular) and LSTM (sequential) using a meta-learner for robust price predictions.
- **Automated Critic**: A dedicated Critic Agent uses LLM-based reasoning (via Groq/GPT-OSS-20B) to validate model outputs against fundamental/momentum flags.
- **Production Scheduler**: Daily automated pipeline runs at 18:30 IST using APScheduler.
- **FastAPI Backend**: Structured API for frontend data access and administrative controls.
- **Professional Dashboard**: Multi-page Streamlit UI with a comprehensive Leaderboard, Stock Details, and Portfolio view.

## User Manual

### Prerequisites
1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your `.env` file with `GROQ_API_KEY` and other necessary credentials.

### Running the App
The system requires both the FastAPI backend and the Streamlit frontend to be running.

**1. Start the FastAPI Backend (including Scheduler):**
```bash
python main.py
```
Or directly via uvicorn:
```bash
uvicorn api.main:app --port 8000
```

**2. Start the Streamlit Dashboard:**
```bash
streamlit run app/main.py
```

### Maintenance
Diagnostic and maintenance scripts are located in the `tools/` directory. These can be used for database migrations, manual signal recomputation, or hyperparameter retuning.
