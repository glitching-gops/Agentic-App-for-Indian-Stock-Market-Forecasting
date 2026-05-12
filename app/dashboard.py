# app/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def fetch_forecast(ticker: str) -> dict:
    """Fetches the latest forecast for a ticker from the FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}/api/forecasts/{ticker}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()

def fetch_signals(ticker: str) -> dict:
    """Fetches historical signals from the FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}/api/signals/{ticker}")
    if response.status_code == 404:
        return {"signals_df": [], "latest_signals": {}}
    response.raise_for_status()
    return response.json()

def fetch_leaderboard(sector=None, verdict=None, confidence=None, sort_by="composite_score") -> dict:
    """Fetches the leaderboard from the FastAPI backend with optional filters."""
    params = {"sort_by": sort_by, "limit": 100}
    if sector:     params["sector"]     = sector
    if verdict:    params["verdict"]    = verdict
    if confidence: params["confidence"] = confidence
    response = requests.get(f"{API_BASE_URL}/api/leaderboard", params=params)
    response.raise_for_status()
    return response.json()

# Removed TICKERS import
from app.style import inject_custom_css
from app.components.chart import render_price_chart
from app.components.signals_view import render_signals_view
from app.components.sentiment_view import render_sentiment_view
from app.components.critic_view import render_critic_view

st.set_page_config(
    page_title="Stock Forecast v3",
    page_icon="🤖",
    layout="wide"
)

inject_custom_css()

# --- Sidebar ---
st.sidebar.markdown("<h2 style='color:#00B4D8; margin-bottom:0;'>ZeRO</h2><p style='color:#475569; margin-top:0;'>Agentic Stock Forecast v3</p>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_stocks() -> list[dict]:
    """
    Fetches the full list of available stocks from the FastAPI backend.
    Returns a list of dicts with ticker, company, and sector keys.
    Falls back to an empty list if the API is unreachable.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/stocks",
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("stocks", [])
    except requests.exceptions.ConnectionError:
        st.warning(
            "⚠️ Backend API is not reachable. "
            "Stock list unavailable. Start the FastAPI server with: "
            "`uvicorn api.main:app --reload --port 8000`"
        )
        return []
    except Exception as e:
        st.error(f"⚠️ Error fetching stock list: {e}")
        return []

stocks = fetch_stocks()

if not stocks:
    st.sidebar.error("No stocks available. Backend may be offline.")
    st.stop()

# Build display name → ticker mapping
company_to_ticker = {s["company"]: s["ticker"] for s in stocks}
ticker_to_sector  = {s["ticker"]: s["sector"]  for s in stocks}

selected_company = st.sidebar.selectbox(
    "Select Stock",
    options=sorted(company_to_ticker.keys())
)
selected_ticker = company_to_ticker[selected_company]
selected_sector = ticker_to_sector[selected_ticker]

# Stock info card
st.sidebar.markdown(f"""
<div class="stCard">
    <div style="font-size: 0.85rem; color: #94A3B8;">Sector</div>
    <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">{selected_sector}</div>
    <div style="font-size: 0.85rem; color: #94A3B8;">Market Cap</div>
    <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">Large Cap</div>
    <div style="font-size: 0.85rem; color: #94A3B8;">Exchange</div>
    <div style="font-weight: 600; color: #FFFFFF;">NSE</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Refresh Data"):
    with st.spinner(f"Triggering pipeline for {selected_company}..."):
        try:
            requests.post(f"{API_BASE_URL}/api/admin/run/{selected_ticker}", headers={"x-api-key": os.getenv("ADMIN_API_KEY", "")})
        except:
            pass
        st.cache_data.clear()
        st.rerun()

# --- Data Loading ---
@st.cache_data(ttl=3600, show_spinner="Fetching forecast from backend...")
def get_agent_state(ticker):
    forecast = fetch_forecast(ticker)
    if not forecast:
        return None
    signals_data = fetch_signals(ticker)
    # Merge signals data into the state dict for the components
    forecast["signals_df"] = signals_data.get("signals_df", [])
    forecast["latest_signals"] = signals_data.get("latest_signals", {})
    return forecast

try:
    state = get_agent_state(selected_ticker)
except Exception as e:
    st.error(f"Error fetching data from API: {e}")
    st.stop()

if not state:
    st.warning(f"No forecast available for {selected_company}. Trigger the pipeline from the backend.")
    st.stop()

mape = state.get('mape', 0)
dir_acc = state.get('directional_accuracy', 0)
if mape > 10 or dir_acc < 55:
    st.sidebar.warning(
        f"⚠️ Model quality for {selected_company} is below threshold "
        f"(MAPE: {mape:.1f}%, Dir Acc: {dir_acc:.1f}%). "
        f"Forecasts for this stock should be treated with caution. "
        f"Review the Agent Analysis tab for Critic feedback."
    )

# df could be empty list of dicts, let's load it to DataFrame
signals_raw = state.get("signals_df", [])
if isinstance(signals_raw, pd.DataFrame):
    df = signals_raw
else:
    df = pd.DataFrame(signals_raw)

# Agent Status
last_updated = "N/A"
is_today = False
if not df.empty and "date" in df.columns:
    last_updated = df["date"].iloc[-1]
    is_today = last_updated == datetime.today().strftime('%Y-%m-%d')
    
status_color = "#06D6A0" if is_today else "#FFB703"

st.sidebar.markdown(f"""
<div class="stCard" style="padding: 1rem;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {status_color};"></div>
        <span style="font-size: 0.9rem; font-weight: 600; color:#FFFFFF;">Data Sync Status</span>
    </div>
    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px; padding-left: 18px;">Last updated: {state.get('last_updated', 'Unknown')}</div>
</div>
""", unsafe_allow_html=True)

st.title(f"{selected_company} ({selected_ticker})")

verdict = state.get("critic_verdict", "FLAGGED")
if verdict == "REJECTED":
    st.error("⚠️ **This forecast has been rejected by the Critic Agent.** Review the Agent Analysis tab before acting on this data.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Signals", "Sentiment", "Agent Analysis"])

# ==========================================
# TAB 1: Overview
# ==========================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = state.get('current_price', 0)
    forecast_price = state.get('forecast_price', 0)
    forecast_change_pct = state.get('change_pct', 0)
    direction = state.get("direction", "UNKNOWN")
    mape = state.get('mape', 0)
    
    with col1:
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Current Price</div>
            <div style="color: #FFB703; font-size: 2rem; font-weight: 700;">₹{current_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        dir_color = "#06D6A0" if direction == "UP" else "#EF476F" if direction == "DOWN" else "#94A3B8"
        dir_symbol = "▲" if direction == "UP" else "▼" if direction == "DOWN" else "-"
        v_class = f"badge-{verdict.lower()}"
        v_icon = "✓" if verdict == "APPROVED" else "⚠" if verdict == "FLAGGED" else "✕"
        
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">30-Day Target</div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                <div style="color: #FFB703; font-size: 2rem; font-weight: 700;">₹{forecast_price:,.2f}</div>
                <div style="background-color: rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 6px; color: {dir_color}; font-weight: 600;">{dir_symbol} {forecast_change_pct}%</div>
            </div>
            <div class="{v_class}" style="font-size: 0.8rem; padding: 2px 8px;">{v_icon} CRITIC {verdict}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Direction</div>
            <div style="color: {dir_color}; font-size: 2rem; font-weight: 700;">{direction}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        mape_color = "#EF476F" if mape > 8.0 else "#FFFFFF"
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Model Error (MAPE)</div>
            <div style="color: {mape_color}; font-size: 2rem; font-weight: 700;">{mape}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    if not df.empty:
        render_price_chart(df)

# ==========================================
# TAB 2: Signals
# ==========================================
with tab2:
    latest = state.get("latest_signals", {})
    render_signals_view(df, latest)

# ==========================================
# TAB 3: Sentiment
# ==========================================
with tab3:
    agg_score = state.get("sentiment_score", 0.0)
    render_sentiment_view(selected_ticker, agg_score)


# ==========================================
# TAB 4: Agent Analysis
# ==========================================
with tab4:
    st.markdown("""
    <div class="stInfoCard">
        <h4 style="color: #00B4D8; display: flex; align-items: center; gap: 8px; margin-top: 0;"><span style="font-size: 1.5rem;">🤖</span> Forecasting Agent — Signal Narrative</h4>
        <p style="color: #FFFFFF; font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">""" + state.get("signal_narrative", "No narrative generated.") + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; color:#FFFFFF;'>Critic Agent Review</h3>", unsafe_allow_html=True)
    
    c_verdict = state.get("critic_verdict", "FLAGGED")
    v_class = f"badge-{c_verdict.lower()}"
    v_icon = "✓" if c_verdict == "APPROVED" else "⚠" if c_verdict == "FLAGGED" else "✕"
    
    st.markdown(f"""
<div style="margin-left: 20px; border-left: 2px solid #1E3A6E; padding-left: 20px;">
<div style="position: relative;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #00B4D8;"></div>
</div>
<div class="{v_class}" style="margin-bottom: 1rem;">{v_icon} VERDICT: {c_verdict}</div>
</div>
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #1E3A6E;"></div>
</div>
<div class="stCard" style="margin-bottom: 1rem;">
<div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Reasoning</div>
<div style="color: #FFFFFF; line-height: 1.6;">{state.get('critic_reasoning', '')}</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    flags = state.get("critic_flags", [])
    if flags:
        st.markdown(f"""
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #EF476F;"></div>
</div>
<div style="margin-bottom: 1rem;">
<div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Flags Raised</div>
""", unsafe_allow_html=True)
        
        for flag in flags:
            st.markdown(f"""
<div class="stCard" style="border-left: 4px solid #EF476F !important; padding: 1rem !important; margin-bottom: 0.5rem !important;">
<span style="color: #EF476F; font-weight: 600; margin-right: 8px;">⚠</span> <span style="color: #FFFFFF;">{flag}</span>
</div>
""", unsafe_allow_html=True)
            
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    st.markdown(f"""
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #FFB703;"></div>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-right: 1rem;">Confidence Adjustment</span>
<span style="background-color: rgba(255, 183, 3, 0.2); color: #FFB703; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.9rem;">{state.get('critic_confidence_adjustment', 'MAINTAINED')}</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Raw Shared LangGraph State (JSON)"):
        safe_state = {}
        for k, v in state.items():
            if k in ["signals_df", "macro_df"] and isinstance(v, list):
                safe_state[k] = f"List of {len(v)} records"
            else:
                safe_state[k] = v
                
        import numpy as np
        def default_serializer(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        st.code(json.dumps(safe_state, indent=2, default=default_serializer), language="json")
