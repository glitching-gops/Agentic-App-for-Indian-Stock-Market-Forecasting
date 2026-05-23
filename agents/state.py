from typing import TypedDict, Any, List, Optional


class AgentState(TypedDict):
    ticker: str
    company_name: str
    current_price: float

    # Trading Data Agent
    signals_df: Any       # list of dicts (JSON-serialisable)
    latest_signals: dict

    # External Data Agent
    sentiment_score: float
    macro_df: Any         # list of dicts

    # Forecasting Agent
    forecast_price: float
    forecast_direction: str          # "UP" or "DOWN"
    forecast_change_pct: float
    forecast_confidence: str         # "High", "Medium", "Low"
    model_mape: float
    model_directional_accuracy: float
    feature_importances: dict
    signal_narrative: str

    # Prediction intervals from TFT quantiles (log-returns, back-transformed to prices)
    forecast_price_q10: Optional[float]   # 10th percentile price
    forecast_price_q90: Optional[float]   # 90th percentile price

    # Individual model forecasts — useful for critic context
    xgb_forecast_price: Optional[float]
    tft_forecast_price: Optional[float]
    tfm_forecast_price: Optional[float]

    # Critic Agent
    critic_verdict: str              # "APPROVED", "FLAGGED", "REJECTED"
    critic_reasoning: str
    critic_flags: List[str]
    critic_confidence_adjustment: str   # "UPGRADED", "MAINTAINED", "DOWNGRADED"

    # Reflection loop — set by graph.py, read by forecasting_node on retry
    reflection_count: int            # 0 on first pass; incremented on each retry
    critic_feedback: str             # populated with critic_reasoning on REJECTED
