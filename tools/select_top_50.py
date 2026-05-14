import sys
import os
sys.path.append(os.getcwd())

from data.db import get_engine
from data.tickers import TICKERS
import pandas as pd

engine = get_engine()

# Pull latest metrics from leaderboard table
lb = pd.read_sql("""
    SELECT ticker, company, sector, mape, directional_accuracy,
           composite_score, critic_verdict, forecast_confidence
    FROM leaderboard
    ORDER BY composite_score DESC
""", con=engine)

# Merge with TICKERS to ensure sector is populated correctly
ticker_df = pd.DataFrame([
    {"ticker": t, "company": v["company"], "sector": v["sector"]}
    for t, v in TICKERS.items()
])
lb = lb.merge(ticker_df[["ticker", "sector"]], on="ticker", suffixes=("_old", ""))
lb["sector"] = lb["sector"].fillna(lb["sector_old"])
lb = lb.drop(columns=["sector_old"], errors="ignore")

# Rank within each sector
# Primary sort: composite_score DESC
# Secondary sort: mape ASC (lower is better)
# Tertiary sort: directional_accuracy DESC
lb["rank_in_sector"] = lb.groupby("sector")["composite_score"].rank(
    method="first", ascending=False
)

# Select top 5 per sector
selected = lb[lb["rank_in_sector"] <= 5].copy()
selected = selected.sort_values(["sector", "rank_in_sector"])

print("=== Selected 50 Stocks (Top 5 per Sector) ===\n")
for sector in sorted(selected["sector"].unique()):
    sector_stocks = selected[selected["sector"] == sector]
    print(f"\n{sector} ({len(sector_stocks)} stocks):")
    for _, row in sector_stocks.iterrows():
        print(
            f"  {int(row['rank_in_sector'])}. {row['ticker']:25s} "
            f"MAPE: {row['mape']:.2f}%  "
            f"Dir: {row['directional_accuracy']:.1f}%  "
            f"Score: {row['composite_score']:.1f}  "
            f"Verdict: {row['critic_verdict']}"
        )

print(f"\n=== Summary ===")
print(f"Total selected:      {len(selected)}")
print(f"Sectors covered:     {selected['sector'].nunique()}")
print(f"Mean MAPE:           {selected['mape'].mean():.2f}%")
print(f"Mean Dir Accuracy:   {selected['directional_accuracy'].mean():.2f}%")
print(f"APPROVED verdicts:   {(selected['critic_verdict'] == 'APPROVED').sum()}")
print(f"FLAGGED verdicts:    {(selected['critic_verdict'] == 'FLAGGED').sum()}")
print(f"REJECTED verdicts:   {(selected['critic_verdict'] == 'REJECTED').sum()}")

# Save the selected tickers for the next step
selected_tickers = selected["ticker"].tolist()
print(f"\nSelected tickers:")
print(selected_tickers)
