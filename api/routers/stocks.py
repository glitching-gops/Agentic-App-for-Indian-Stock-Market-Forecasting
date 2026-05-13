"""
GET /api/stocks — returns the full list of tickers with company name and sector.
"""
from fastapi import APIRouter
from api.schemas.stock import StockList, StockInfo
from data.tickers import TICKERS

router = APIRouter()

@router.get("", response_model=StockList)
def get_stocks():
    stocks = [
        StockInfo(ticker=ticker, company=info["company"], sector=info["sector"])
        for ticker, info in TICKERS.items()
    ]
    return StockList(stocks=stocks, total=len(stocks))

@router.get("/{ticker}/signals")
def get_signals(ticker: str, days: int = 30):
    """
    Returns the last N days of signal data for a ticker.
    Only queries columns that exist in the signals table.
    """
    from data.db import get_engine
    from fastapi import HTTPException
    import pandas as pd
    engine = get_engine()
    try:
        df = pd.read_sql(
            """
            SELECT date, close, rsi, macd_hist, bb_width, obv,
                   sma_20, ema_9, ema_21, ema_50, atr_14,
                   stoch_k, williams_r, roc_10, vroc_10,
                   prox_52w, lag1_ret, lag5_ret, dev_sma50,
                   bb_upper, bb_lower, hurst,
                   sector_rel_5d, sector_rel_10d, sector_rel_20d,
                   earnings_surprise
            FROM signals
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :days
            """,
            con=engine,
            params={"ticker": ticker.upper(), "days": days}
        )
    except Exception:
        # Fallback for SQLite (no named params)
        try:
            df = pd.read_sql(
                f"""
                SELECT date, close, rsi, macd_hist, bb_width, obv,
                       sma_20, ema_9, ema_21, ema_50, atr_14,
                       stoch_k, williams_r, roc_10, vroc_10,
                       prox_52w, lag1_ret, lag5_ret, dev_sma50,
                       bb_upper, bb_lower, hurst,
                       sector_rel_5d, sector_rel_10d, sector_rel_20d,
                       earnings_surprise
                FROM signals
                WHERE ticker = '{ticker.upper()}'
                ORDER BY date DESC
                LIMIT {days}
                """,
                con=engine
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch signals for {ticker}: {str(e)}")

    df = df.sort_values("date", ascending=True)
    # Replace NaN/inf with None so JSON serialisation works
    df = df.where(df.notna(), other=None)
    return df.to_dict(orient="records")
