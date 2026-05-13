from data.db import get_engine
import pandas as pd
import numpy as np

engine = get_engine()

print("=== Database Integrity Report ===\n")

tables = ["ohlcv", "signals", "sentiment", "macro", "forecasts", "leaderboard", "model_metadata"]
for table in tables:
    try:
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", con=engine).iloc[0]["cnt"]
        print(f"{table:20s}: {count:>8,} rows")
    except Exception as e:
        print(f"{table:20s}: ERROR - {e}")

print()

lb = pd.read_sql("SELECT * FROM leaderboard", con=engine)
from data.tickers import TICKERS
all_tickers = set(TICKERS.keys())
lb_tickers  = set(lb["ticker"].tolist())
missing     = all_tickers - lb_tickers
extra       = lb_tickers - all_tickers

print(f"Leaderboard coverage:  {len(lb_tickers)}/100 stocks")
print(f"Missing from leaderboard: {sorted(missing) if missing else 'None'}")
print(f"Extra in leaderboard:     {sorted(extra) if extra else 'None'}")

print()

verdicts = lb["critic_verdict"].value_counts()
print("Verdict distribution:")
for verdict, count in verdicts.items():
    print(f"  {verdict:12s}: {count}")

print()

print("Composite score stats:")
print(f"  Mean:   {lb['composite_score'].mean():.2f}")
print(f"  Median: {lb['composite_score'].median():.2f}")
print(f"  Min:    {lb['composite_score'].min():.2f}")
print(f"  Max:    {lb['composite_score'].max():.2f}")

print()

try:
    signals = pd.read_sql("SELECT * FROM signals LIMIT 5000", con=engine)
    inf_count = np.isinf(signals.select_dtypes(include="number")).sum().sum()
    nan_count = signals.select_dtypes(include="number").isna().sum().sum()
    print(f"Signals (sample 5000) - infinite values: {inf_count}")
    print(f"Signals (sample 5000) - NaN values:      {nan_count}")
except Exception as e:
    print(f"Signals check error: {e}")

print()

try:
    latest = pd.read_sql("""
        SELECT ticker, MAX(last_updated) as latest
        FROM forecasts
        GROUP BY ticker
        ORDER BY latest ASC
        LIMIT 5
    """, con=engine)
    print("Oldest forecast records (potential staleness):")
    print(latest.to_string())
except Exception as e:
    print(f"Forecasts staleness check error: {e}")

print()

# Check APPROVED_OR_FLAGGED filter
try:
    approved = lb[lb["critic_verdict"] == "APPROVED"]
    flagged = lb[lb["critic_verdict"] == "FLAGGED"]
    print(f"APPROVED stocks count: {len(approved)}")
    print(f"FLAGGED stocks count: {len(flagged)}")
    print(f"REJECTED stocks count: {len(lb[lb['critic_verdict'] == 'REJECTED'])}")
except Exception as e:
    print(f"Verdict count error: {e}")
