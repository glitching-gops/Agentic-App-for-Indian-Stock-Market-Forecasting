from data.db import get_engine
import pandas as pd

engine = get_engine()

print("=== Forecasts Table ===")
forecasts = pd.read_sql("""
    SELECT ticker, direction, mape, directional_accuracy,
           forecast_confidence, critic_verdict, last_updated
    FROM forecasts
    ORDER BY last_updated DESC
""", con=engine)
print(forecasts.to_string())
print(f"\nTotal forecast records: {len(forecasts)}")
print(f"Unique tickers with forecasts: {forecasts['ticker'].nunique()}")

print("\n=== Leaderboard Table ===")
leaderboard = pd.read_sql("""
    SELECT rank() OVER (ORDER BY composite_score DESC) as rank,
           ticker, company, sector, upside_pct,
           composite_score, critic_verdict, forecast_confidence,
           mape, directional_accuracy
    FROM leaderboard
    ORDER BY composite_score DESC
""", con=engine)
print(leaderboard.to_string())
print(f"\nTotal stocks on leaderboard: {len(leaderboard)}")
print(f"Sectors represented: {leaderboard['sector'].nunique()}")

print("\n=== Stocks missing from leaderboard ===")
from data.tickers import TICKERS
missing = [t for t in TICKERS.keys() if t not in leaderboard['ticker'].values]
print(missing if missing else "None — all stocks present")
