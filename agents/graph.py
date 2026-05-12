"""
graph.py
LangGraph orchestration for the Stock Forecast v3 pipeline.
"""
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.trading_data_agent import trading_data_node
from agents.external_data_agent import external_data_node
from agents.forecasting_agent import forecasting_node
from agents.critic_agent import critic_node
from data.tickers import TICKERS
import pandas as pd

def compute_composite_score(
    upside_pct: float,
    verdict: str,
    confidence: str,
    directional_accuracy: float,
) -> float:
    """
    Computes the composite leaderboard score out of 100.

    Components:
      - Directional accuracy (30 pts): most actionable signal
      - Forecast upside potential (25 pts): expected return
      - Critic verdict (30 pts): quality gate
      - Model confidence (15 pts): ensemble reliability

    Args:
        upside_pct:           forecast change % (e.g. 5.2 for 5.2%)
        verdict:              APPROVED / FLAGGED / REJECTED
        confidence:           High / Medium / Low
        directional_accuracy: model directional accuracy 0-100

    Returns:
        composite score 0-100 (float, rounded to 2dp)
    """
    # Directional accuracy — linear 0 to 30
    dir_score = (min(max(directional_accuracy, 0.0), 100.0) / 100.0) * 30.0

    # Upside score — capped at 25 points
    # multiplier 1.5 means 16.7% upside = max score
    upside_score = min(max(upside_pct * 1.5, 0.0), 25.0)

    # Verdict score
    verdict_score = {"APPROVED": 30.0, "FLAGGED": 12.0, "REJECTED": 0.0}.get(
        verdict, 12.0
    )

    # Confidence score
    conf_score = {"High": 15.0, "Medium": 7.0, "Low": 0.0}.get(
        confidence, 0.0
    )

    return round(dir_score + upside_score + verdict_score + conf_score, 2)

def save_forecast_to_db(state: dict):
    """
    Saves the completed agent pipeline state to the forecasts
    and leaderboard tables in the database.
    Called automatically at the end of run_graph().
    """
    import json
    from data.db import get_engine
    from data.tickers import get_company, get_sector
    from sqlalchemy import text
    from datetime import datetime

    engine = get_engine()
    ticker = state.get("ticker", "")

    # Compute composite score
    upside_pct = state.get("forecast_change_pct", 0)
    verdict    = state.get("critic_verdict", "FLAGGED")
    confidence = state.get("forecast_confidence", "Low")

    composite = compute_composite_score(
        upside_pct           = upside_pct,
        verdict              = verdict,
        confidence           = confidence,
        directional_accuracy = state.get("model_directional_accuracy", 0.0),
    )

    now = datetime.utcnow()

    with engine.connect() as conn:
        # Insert into forecasts table (keep full history)
        conn.execute(text("""
            INSERT INTO forecasts (
                ticker, company, sector, current_price, forecast_price,
                direction, change_pct, mape, directional_accuracy,
                forecast_confidence, signal_narrative, critic_verdict,
                critic_reasoning, critic_flags, critic_confidence_adjustment,
                last_updated
            ) VALUES (
                :ticker, :company, :sector, :current_price, :forecast_price,
                :direction, :change_pct, :mape, :directional_accuracy,
                :forecast_confidence, :signal_narrative, :critic_verdict,
                :critic_reasoning, :critic_flags, :critic_confidence_adjustment,
                :last_updated
            )
        """), {
            "ticker":                       ticker,
            "company":                      get_company(ticker),
            "sector":                       get_sector(ticker),
            "current_price":                state.get("current_price"),
            "forecast_price":               state.get("forecast_price"),
            "direction":                    state.get("forecast_direction"),
            "change_pct":                   state.get("forecast_change_pct"),
            "mape":                         state.get("model_mape"),
            "directional_accuracy":         state.get("model_directional_accuracy"),
            "forecast_confidence":          state.get("forecast_confidence"),
            "signal_narrative":             state.get("signal_narrative"),
            "critic_verdict":               verdict,
            "critic_reasoning":             state.get("critic_reasoning"),
            "critic_flags":                 json.dumps(state.get("critic_flags", [])),
            "critic_confidence_adjustment": state.get("critic_confidence_adjustment"),
            "last_updated":                 now,
        })

        # Upsert into leaderboard table (one row per ticker, always current)
        try:
            conn.execute(text("""
                INSERT INTO leaderboard (
                    ticker, company, sector, current_price, forecast_price,
                    upside_pct, composite_score, critic_verdict,
                    forecast_confidence, mape, directional_accuracy, last_updated
                ) VALUES (
                    :ticker, :company, :sector, :current_price, :forecast_price,
                    :upside_pct, :composite_score, :critic_verdict,
                    :forecast_confidence, :mape, :directional_accuracy, :last_updated
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    company              = EXCLUDED.company,
                    sector               = EXCLUDED.sector,
                    current_price        = EXCLUDED.current_price,
                    forecast_price       = EXCLUDED.forecast_price,
                    upside_pct           = EXCLUDED.upside_pct,
                    composite_score      = EXCLUDED.composite_score,
                    critic_verdict       = EXCLUDED.critic_verdict,
                    forecast_confidence  = EXCLUDED.forecast_confidence,
                    mape                 = EXCLUDED.mape,
                    directional_accuracy = EXCLUDED.directional_accuracy,
                    last_updated         = EXCLUDED.last_updated
            """), {
                "ticker":               ticker,
                "company":              get_company(ticker),
                "sector":               get_sector(ticker),
                "current_price":        state.get("current_price"),
                "forecast_price":       state.get("forecast_price"),
                "upside_pct":           upside_pct,
                "composite_score":      composite,
                "critic_verdict":       verdict,
                "forecast_confidence":  confidence,
                "mape":                 state.get("model_mape"),
                "directional_accuracy": state.get("model_directional_accuracy"),
                "last_updated":         now,
            })
        except Exception:
            # Fallback for local SQLite development
            conn.execute(text("""
                INSERT OR REPLACE INTO leaderboard (
                    ticker, company, sector, current_price, forecast_price,
                    upside_pct, composite_score, critic_verdict,
                    forecast_confidence, mape, directional_accuracy, last_updated
                ) VALUES (
                    :ticker, :company, :sector, :current_price, :forecast_price,
                    :upside_pct, :composite_score, :critic_verdict,
                    :forecast_confidence, :mape, :directional_accuracy, :last_updated
                )
            """), {
                "ticker":               ticker,
                "company":              get_company(ticker),
                "sector":               get_sector(ticker),
                "current_price":        state.get("current_price"),
                "forecast_price":       state.get("forecast_price"),
                "upside_pct":           upside_pct,
                "composite_score":      composite,
                "critic_verdict":       verdict,
                "forecast_confidence":  confidence,
                "mape":                 state.get("model_mape"),
                "directional_accuracy": state.get("model_directional_accuracy"),
                "last_updated":         now,
            })

        conn.commit()

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("trading_data", trading_data_node)
    workflow.add_node("external_data", external_data_node)
    workflow.add_node("forecasting", forecasting_node)
    workflow.add_node("critic", critic_node)
    
    # Add edges
    workflow.add_edge(START, "trading_data")
    workflow.add_edge(START, "external_data")
    
    # Forecasting waits for both parallel data nodes
    workflow.add_edge(["trading_data", "external_data"], "forecasting")
    
    workflow.add_edge("forecasting", "critic")
    workflow.add_edge("critic", END)
    
    return workflow.compile()

graph = build_graph()

def run_graph(ticker: str) -> dict:
    print(f"\n--- Running Agent Graph for {ticker} ---")
    
    try:
        # Initialize state
        initial_state = AgentState(
            ticker=ticker,
            company_name=TICKERS.get(ticker, {}).get("company", ticker),
            current_price=0.0,
            signals_df=[],
            latest_signals={},
            sentiment_score=0.0,
            macro_df=[],
            forecast_price=0.0,
            forecast_direction="UNKNOWN",
            forecast_change_pct=0.0,
            forecast_confidence="Low",
            model_mape=0.0,
            model_directional_accuracy=0.0,
            feature_importances={},
            signal_narrative="",
            critic_verdict="FLAGGED",
            critic_reasoning="",
            critic_flags=[],
            critic_confidence_adjustment="MAINTAINED"
        )
        
        final_state = graph.invoke(initial_state)
        save_forecast_to_db(final_state)
        print(f"--- Completed Agent Graph for {ticker} ---\n")
        return final_state
        
    except Exception as e:
        safe_err = str(e).encode("ascii", "backslashreplace").decode("ascii")
        print(f"--- FAILED Agent Graph for {ticker}: {safe_err} ---\n")
        import traceback
        traceback.print_exc()
        return {
            "ticker": ticker,
            "company_name": TICKERS.get(ticker, {}).get("company", ticker),
            "current_price": 0.0,
            "forecast_price": 0.0,
            "forecast_direction": "ERROR",
            "forecast_change_pct": 0.0,
            "forecast_confidence": "Low",
            "model_mape": 100.0,
            "model_directional_accuracy": 0.0,
            "critic_verdict": "REJECTED",
            "critic_reasoning": f"Catastrophic Graph Failure: {safe_err}",
            "critic_flags": ["Pipeline Error"],
            "critic_confidence_adjustment": "DOWNGRADED"
        }
