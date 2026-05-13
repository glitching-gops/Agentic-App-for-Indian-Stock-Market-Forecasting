from pydantic import BaseModel
from typing import Optional

class ForecastResponse(BaseModel):
    ticker: str
    company: str
    sector: str
    current_price: float
    forecast_price: float
    direction: str
    change_pct: float
    mape: float
    directional_accuracy: float
    forecast_confidence: str
    signal_narrative: Optional[str]
    critic_verdict: str
    critic_reasoning: Optional[str]
    critic_flags: list[str]
    critic_confidence_adjustment: str
    last_updated: str
