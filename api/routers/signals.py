"""
GET /api/signals/{ticker} — returns the historical signals DataFrame and
the most recent row as latest_signals, used by the dashboard chart and
signals view components. Reads directly from the signals table in the DB.
"""
from fastapi import APIRouter, HTTPException
from data.db import get_engine
import pandas as pd

router = APIRouter()

@router.get("/{ticker}")
def get_signals(ticker: str, days: int = 200):
    """
    Returns up to `days` rows of signals data for the given ticker,
    sorted ascending by date, plus the most recent row as `latest_signals`.
    """
    engine = get_engine()
    try:
        df = pd.read_sql(
            "SELECT * FROM signals WHERE ticker = :ticker ORDER BY date DESC LIMIT :days",
            con=engine,
            params={"ticker": ticker.upper(), "days": days}
        )
    except Exception as e:
        # Fallback: try without named params for SQLite
        try:
            df = pd.read_sql(
                f"SELECT * FROM signals WHERE ticker = '{ticker.upper()}' ORDER BY date DESC LIMIT {days}",
                con=engine,
            )
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No signal data found for {ticker}. Run the pipeline first."
        )

    # Return in ascending order for charting
    df = df.sort_values("date", ascending=True)

    # Replace NaN/inf with None so JSON serialisation works
    df = df.where(df.notna(), other=None)

    # Latest signals dict from the most recent row
    latest_row = df.iloc[-1].to_dict()
    latest_signals = {
        k: (float(v) if isinstance(v, float) and pd.notna(v) else (None if pd.isna(v) else v))
        for k, v in latest_row.items()
        if k not in ("date", "ticker", "target")
    }

    return {
        "ticker": ticker.upper(),
        "signals_df": df.to_dict(orient="records"),
        "latest_signals": latest_signals,
        "rows": len(df),
    }
