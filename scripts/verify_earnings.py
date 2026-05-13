import sys
import os
sys.path.append(os.getcwd())

from pipeline.signals import compute_and_store
from data.db import get_engine, init_db
import pandas as pd
from sqlalchemy import text

init_db()

engine = get_engine()
with engine.begin() as conn:
    conn.execute(text("DELETE FROM signals WHERE ticker = 'INFY.NS'"))

compute_and_store("INFY.NS")

df = pd.read_sql("""
    SELECT date, close, earnings_surprise
    FROM signals
    WHERE ticker = 'INFY.NS'
    ORDER BY date DESC
    LIMIT 20
""", con=engine)

print("=== Earnings Surprise — INFY.NS (last 20 rows) ===")
print(df.to_string())

non_zero = (df["earnings_surprise"] != 0.0).sum()
print(f"\nNon-zero rows: {non_zero}/20")
print(f"Value range: {df['earnings_surprise'].min():.4f} to {df['earnings_surprise'].max():.4f}")
