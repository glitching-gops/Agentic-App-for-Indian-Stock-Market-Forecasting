"""
GET /api/forecasts/{ticker} — returns the latest forecast for a single stock.
Reads from the database rather than re-running the pipeline on every request.
"""
from fastapi import APIRouter, HTTPException
from api.schemas.forecast import ForecastResponse
from data.db import get_engine
from sqlalchemy import text
import json

router = APIRouter()

@router.get("/{ticker}", response_model=ForecastResponse)
def get_forecast(ticker: str):
    """Returns the latest stored forecast for a given ticker symbol."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM forecasts WHERE ticker = :ticker ORDER BY created_at DESC LIMIT 1"),
                {"ticker": ticker.upper()}
            )
            row = result.mappings().first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for {ticker}. Run the pipeline first."
        )

    data = dict(row)

    # critic_flags is stored as a JSON string — parse it back to a list
    raw_flags = data.get("critic_flags", "[]")
    if isinstance(raw_flags, str):
        try:
            data["critic_flags"] = json.loads(raw_flags)
        except (json.JSONDecodeError, TypeError):
            data["critic_flags"] = []
    elif raw_flags is None:
        data["critic_flags"] = []

    # Ensure last_updated is a string for the schema
    if data.get("last_updated") is not None:
        data["last_updated"] = str(data["last_updated"])

    return ForecastResponse(**data)
