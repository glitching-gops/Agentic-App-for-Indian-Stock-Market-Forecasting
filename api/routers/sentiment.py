from fastapi import APIRouter, HTTPException
from data.db import get_engine
from sqlalchemy import text

router = APIRouter()

@router.get("/{ticker}/headlines")
def get_headlines(ticker: str):
    """
    Returns the 5 most recent headlines for a given ticker from the sentiment table.
    """
    engine = get_engine()
    try:
        query = text("""
            SELECT headline, sentiment_label, sentiment_score, date 
            FROM sentiment 
            WHERE ticker = :ticker 
            ORDER BY date DESC 
            LIMIT 5
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker.upper()})
            rows = result.mappings().all()
            
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
