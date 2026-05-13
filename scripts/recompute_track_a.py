import sys
import os
sys.path.append(os.getcwd())

# Step 5a — Run migrations
from data.db import init_db
init_db()
print("Schema migrations complete")

# Step 5b — Recompute signals for all 100 stocks
from pipeline.signals import compute_and_store
from data.tickers import TICKERS

errors = []
for i, ticker in enumerate(TICKERS.keys()):
    try:
        compute_and_store(ticker)
        print(f"[{i+1}/100] {ticker}: OK")
    except Exception as e:
        errors.append((ticker, str(e)))
        print(f"[{i+1}/100] {ticker}: ERROR — {e}")

print(f"\nCompleted. Errors: {len(errors)}")
for t, e in errors:
    print(f"  {t}: {e}")

# Health Check
from data.db import get_engine
import pandas as pd

engine = get_engine()

print("\n=== Track A Signal Health Check ===")
df = pd.read_sql("""
    SELECT
        COUNT(*)                                                         AS total_rows,
        COUNT(DISTINCT ticker)                                           AS tickers,
        SUM(CASE WHEN sector_rel_5d    IS NULL THEN 1 ELSE 0 END)       AS sector_5d_null,
        SUM(CASE WHEN sector_rel_10d   IS NULL THEN 1 ELSE 0 END)       AS sector_10d_null,
        SUM(CASE WHEN sector_rel_20d   IS NULL THEN 1 ELSE 0 END)       AS sector_20d_null,
        SUM(CASE WHEN earnings_surprise IS NULL THEN 1 ELSE 0 END)      AS earnings_null,
        SUM(CASE WHEN sector_rel_5d = 0.0 THEN 1 ELSE 0 END) * 100.0
            / COUNT(*)                                                   AS sector_zero_pct,
        SUM(CASE WHEN earnings_surprise = 0.0 THEN 1 ELSE 0 END) * 100.0
            / COUNT(*)                                                   AS earnings_zero_pct
    FROM signals
""", con=engine)
print(df.to_string())

print("\n=== Per-sector sector_rel_5d coverage ===")
coverage = pd.read_sql("""
    SELECT
        s.ticker,
        AVG(CASE WHEN s.sector_rel_5d != 0 THEN 1.0 ELSE 0.0 END) * 100
            AS non_zero_pct
    FROM signals s
    GROUP BY s.ticker
    ORDER BY non_zero_pct ASC
    LIMIT 10
""", con=engine)
print(coverage.to_string())
