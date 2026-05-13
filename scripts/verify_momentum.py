import sys
import os
sys.path.append(os.getcwd())

from data.db import init_db
init_db()

from pipeline.signals import compute_and_store
from data.db import get_engine
import pandas as pd
from sqlalchemy import text

engine = get_engine()
with engine.begin() as conn:
    conn.execute(text("DELETE FROM signals WHERE ticker = 'RELIANCE.NS'"))

compute_and_store("RELIANCE.NS")


engine = get_engine()
df = pd.read_sql("""
    SELECT date, close, sector_rel_5d, sector_rel_10d, sector_rel_20d
    FROM signals
    WHERE ticker = 'RELIANCE.NS'
    ORDER BY date DESC
    LIMIT 10
""", con=engine)

print("=== Sector Momentum — RELIANCE.NS (last 10 rows) ===")
print(df.to_string())
print(f"\nNon-zero sector_rel_5d rows: {(df['sector_rel_5d'] != 0).sum()}")
print(f"Value range: {df['sector_rel_5d'].min():.6f} to {df['sector_rel_5d'].max():.6f}")
