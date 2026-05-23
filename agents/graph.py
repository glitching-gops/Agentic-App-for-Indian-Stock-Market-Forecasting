"""
agents/graph.py

LangGraph orchestration for the Stock Forecast v2 pipeline.

Graph structure:
  START ──► trading_data  ─────────────────────────────►
                                                         forecasting ──► critic
  START ──► external_data ─────────────────────────────►
                                                              │
                    ┌─── REJECTED + reflection_count < 2 ◄───┘
                    │         (loop back to forecasting
                    │          with critic_feedback injected)
                    └─── APPROVED / FLAGGED / exhausted ──► END

The reflection loop gives the Critic Agent real agency: a REJECTED forecast
triggers a retry where the forecasting node receives the critic's flags as
additional context and can adjust its narrative.  Maximum 2 retries to
prevent infinite loops.
"""

from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.trading_data_agent import trading_data_node
from agents.external_data_agent import external_data_node
from agents.forecasting_agent import forecasting_node
from agents.critic_agent import critic_node
from data.tickers import TICKERS

MAX_REFLECTIONS = 2


def _should_reflect(state: AgentState) -> str:
    """
    Routing function called after the critic node.

    Returns "reflect" if the forecast was REJECTED and we haven't hit the
    reflection limit yet, otherwise "end".
    """
    verdict = state.get("critic_verdict", "FLAGGED")
    count   = state.get("reflection_count", 0)

    if verdict == "REJECTED" and count < MAX_REFLECTIONS:
        return "reflect"
    return "end"


def _reflection_node(state: AgentState) -> dict:
    """
    Prepares the state for a forecasting retry by:
      1. Incrementing reflection_count
      2. Copying critic_reasoning into critic_feedback so the
         forecasting node can see why the previous attempt was rejected
      3. Clearing the stale critic fields so the next critic pass is fresh
    """
    return {
        "reflection_count": state.get("reflection_count", 0) + 1,
        "critic_feedback":  state.get("critic_reasoning", ""),
        "critic_verdict":   "FLAGGED",   # reset so routing doesn't loop on stale value
        "critic_flags":     [],
    }


def compute_composite_score(
    upside_pct: float,
    verdict: str,
    confidence: str,
    directional_accuracy: float,
) -> float:
    """
    Computes the composite leaderboard score out of 100.

    Components:
      - Directional accuracy (30 pts) — primary signal quality measure
      - Critic verdict       (30 pts) — quality gate
      - Forecast upside      (25 pts) — expected return potential
      - Model confidence     (15 pts) — ensemble reliability
    """
    dir_score     = (min(max(directional_accuracy, 0.0), 100.0) / 100.0) * 30.0
    upside_score  = min(max(upside_pct * 1.5, 0.0), 25.0)
    verdict_score = {"APPROVED": 30.0, "FLAGGED": 12.0, "REJECTED": 0.0}.get(verdict, 12.0)
    conf_score    = {"High": 15.0, "Medium": 7.0, "Low": 0.0}.get(confidence, 0.0)
    return round(dir_score + upside_score + verdict_score + conf_score, 2)


def save_forecast_to_db(state: dict):
    """
    Saves the completed agent pipeline state to the forecasts
    and leaderboard tables.  Called at the end of run_graph().
    """
    import json
    from data.db import get_engine
    from data.tickers import get_company, get_sector
    from sqlalchemy import text
    from datetime import datetime

    engine = get_engine()
    ticker = state.get("ticker", "")

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

    workflow.add_node("trading_data",  trading_data_node)
    workflow.add_node("external_data", external_data_node)
    workflow.add_node("forecasting",   forecasting_node)
    workflow.add_node("critic",        critic_node)
    workflow.add_node("reflection",    _reflection_node)

    # Parallel data fetch
    workflow.add_edge(START, "trading_data")
    workflow.add_edge(START, "external_data")

    # Both data nodes must complete before forecasting
    workflow.add_edge(["trading_data", "external_data"], "forecasting")

    workflow.add_edge("forecasting", "critic")

    # Conditional: reflect (loop) or end
    workflow.add_conditional_edges(
        "critic",
        _should_reflect,
        {"reflect": "reflection", "end": END},
    )

    # After reflection, go back to forecasting with updated state
    workflow.add_edge("reflection", "forecasting")

    return workflow.compile()


graph = build_graph()


def run_graph(ticker: str) -> dict:
    print(f"\n--- Running Agent Graph for {ticker} ---")

    try:
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
            forecast_price_q10=None,
            forecast_price_q90=None,
            xgb_forecast_price=None,
            tft_forecast_price=None,
            tfm_forecast_price=None,
            critic_verdict="FLAGGED",
            critic_reasoning="",
            critic_flags=[],
            critic_confidence_adjustment="MAINTAINED",
            reflection_count=0,
            critic_feedback="",
        )

        final_state = graph.invoke(initial_state)
        save_forecast_to_db(final_state)

        reflections = final_state.get("reflection_count", 0)
        if reflections > 0:
            print(f"--- {ticker}: {reflections} reflection(s) performed ---")
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
            "critic_confidence_adjustment": "DOWNGRADED",
            "reflection_count": 0,
        }
