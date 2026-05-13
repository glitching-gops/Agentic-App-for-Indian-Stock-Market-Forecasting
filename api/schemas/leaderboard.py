from pydantic import BaseModel

class LeaderboardEntry(BaseModel):
    rank: int
    ticker: str
    company: str
    sector: str
    current_price: float
    forecast_price: float
    upside_pct: float
    composite_score: float
    critic_verdict: str
    forecast_confidence: str
    mape: float
    directional_accuracy: float

class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    total: int
    last_updated: str
    filters_applied: dict
