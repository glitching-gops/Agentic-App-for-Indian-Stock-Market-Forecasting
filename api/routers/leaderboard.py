"""
GET /api/leaderboard — returns ranked stocks by composite score.
Supports filtering by sector, verdict, confidence, and sorting.
"""
from fastapi import APIRouter, Query
from api.schemas.leaderboard import LeaderboardResponse, LeaderboardEntry
from data.db import get_engine
import pandas as pd
from datetime import datetime
from typing import Optional

router = APIRouter()

@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    sector:     Optional[str] = Query(None),
    verdict:    Optional[str] = Query(None),
    confidence: Optional[str] = Query(None),
    sort_by:    str = Query("composite_score"),
    limit:      int = Query(20)
):
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM leaderboard ORDER BY composite_score DESC", con=engine)

    filters_applied = {}

    if sector:
        df = df[df["sector"] == sector]
        filters_applied["sector"] = sector
    if verdict:
        verdict_upper = verdict.upper()
        if verdict_upper == "APPROVED_OR_FLAGGED":
            df = df[df["critic_verdict"].isin(["APPROVED", "FLAGGED"])]
            filters_applied["verdict"] = "APPROVED_OR_FLAGGED"
        elif verdict_upper in ["APPROVED", "FLAGGED", "REJECTED"]:
            df = df[df["critic_verdict"] == verdict_upper]
            filters_applied["verdict"] = verdict_upper
        # Silently ignore invalid verdict values
    if confidence:
        df = df[df["forecast_confidence"] == confidence.capitalize()]
        filters_applied["confidence"] = confidence

    valid_sort = ["composite_score", "upside_pct", "mape", "directional_accuracy"]
    if sort_by in valid_sort:
        ascending = sort_by == "mape"
        df = df.sort_values(sort_by, ascending=ascending)

    df = df.head(limit)
    df["rank"] = range(1, len(df) + 1)

    entries = [LeaderboardEntry(**row.to_dict()) for _, row in df.iterrows()]
    last_updated = df["last_updated"].max() if "last_updated" in df.columns else datetime.now().isoformat()

    return LeaderboardResponse(
        entries=entries,
        total=len(entries),
        last_updated=str(last_updated),
        filters_applied=filters_applied
    )
