"""
app/pages/about.py

About page describing the system architecture and methodology.
"""

import streamlit as st


def main():
    st.markdown("## ℹ️ About ZeRO Stock Forecast")
    st.divider()

    st.markdown("""
    **ZeRO Agentic Stock Forecast** is a multi-agent AI system for
    30-day price target prediction on the National Stock Exchange of India.
    Built as a B.Tech final project at Rajiv Gandhi Institute of Petroleum
    Technology under the guidance of Dr. Roopa Manjunatha.
    """)

    st.markdown("### System Architecture")
    st.markdown("""
    The system uses four specialised agents orchestrated by **LangGraph**:

    1. **Trading Data Agent** — fetches OHLCV data and computes 20 technical
       signals including RSI, MACD, Bollinger Bands, ATR, OBV, and Hurst
       exponent for regime detection.

    2. **External Data Agent** — fetches news headlines and scores them
       using **ProsusAI/FinBERT**, a finance-domain sentiment classifier.
       Also fetches macro indicators: USD/INR, India VIX, Nifty momentum,
       sector relative strength, and earnings surprise.

    3. **Forecasting Agent** — trains an **XGBoost + LSTM ensemble** with a
       Ridge regression meta-learner. The meta-learner uses the Hurst exponent
       to dynamically weight the two models based on market regime. Also
       generates a plain-English signal narrative via the **Groq API**.

    4. **Critic Agent** — reviews each forecast using an LLM and assigns a
       structured verdict of APPROVED, FLAGGED, or REJECTED based on signal
       consistency, model reliability, and forecast plausibility.
    """)

    st.markdown("### Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Covered", "100")
    col2.metric("Mean MAPE", "4.34%")
    col3.metric("Mean Dir Accuracy", "85.13%")
    col4.metric("Forecast Horizon", "30 days")

    st.markdown("### Tech Stack")
    st.markdown("""
    | Layer | Technology |
    |---|---|
    | Agents & LLMs | LangGraph, LangChain, Groq (openai/gpt-oss-20b) |
    | Sentiment | ProsusAI/FinBERT |
    | ML Models | XGBoost (Optuna-tuned), PyTorch LSTM, Ridge meta-learner |
    | Backend | FastAPI + Uvicorn |
    | Frontend | Streamlit + Plotly |
    | Database | Supabase PostgreSQL |
    | Data | yfinance, feedparser (Google News RSS) |
    | Scheduler | APScheduler (daily 18:30 IST, weekly retune Sunday 02:00 IST) |
    """)

    st.divider()
    st.caption(
        "Built by Venu Gopal Battula · Roll No. 22CS2014 · "
        "Rajiv Gandhi Institute of Petroleum Technology · 2025-26"
    )


main()
