from typing import TypedDict, Any, List

class AgentState(TypedDict):
    ticker: str
    company_name: str
    current_price: float
    
    # Trading Data Agent
    signals_df: Any  # pandas DataFrame
    latest_signals: dict
    
    # External Data Agent
    sentiment_score: float
    macro_df: Any    # pandas DataFrame
    
    # Forecasting Agent
    forecast_price: float
    forecast_direction: str         # "UP" or "DOWN"
    forecast_change_pct: float
    forecast_confidence: str        # "High", "Medium", or "Low"
    model_mape: float
    model_directional_accuracy: float
    feature_importances: dict
    signal_narrative: str
    
    # Critic Agent
    critic_verdict: str             # "APPROVED", "FLAGGED", or "REJECTED"
    critic_reasoning: str
    critic_flags: List[str]
    critic_confidence_adjustment: str  # "UPGRADED", "MAINTAINED", "DOWNGRADED"
