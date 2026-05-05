import sys
import os
sys.path.append(os.getcwd())

from data.db import get_engine
from pipeline.signals import compute_and_store
from sqlalchemy import text
import pandas as pd

engine = get_engine()
with engine.connect() as conn:
    conn.execute(text("DELETE FROM signals"))
    conn.commit()
print("Signals table cleared.")

compute_and_store()

print("\n=== Hurst Exponent Verification ===")
df = pd.read_sql("""
    SELECT ticker, AVG(hurst) as avg_hurst,
           MIN(hurst) as min_hurst,
           MAX(hurst) as max_hurst,
           SUM(CASE WHEN hurst IS NULL THEN 1 ELSE 0 END) as null_count
    FROM signals
    GROUP BY ticker
    ORDER BY avg_hurst DESC
    LIMIT 10
""", con=engine)
print(df.to_string())
