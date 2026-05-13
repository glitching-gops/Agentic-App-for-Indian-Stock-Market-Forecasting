import sys
import os
sys.path.append(os.getcwd())

import yfinance as yf
from data.tickers import SECTOR_INDICES
import pandas as pd

print("=== Sector Index Ticker Verification ===\n")
failed = []

for sector, index_ticker in SECTOR_INDICES.items():
    try:
        df = yf.download(
            index_ticker, period="5d", interval="1d",
            auto_adjust=True, progress=False
        )
        rows = len(df)
        if rows > 0:
            latest_close = df["Close"].iloc[-1] if "Close" in df.columns else "N/A"
            if isinstance(latest_close, pd.Series):
                latest_close = latest_close.iloc[0]
            if not isinstance(latest_close, str):
                latest_close = float(latest_close)
            print(f"  OK      {index_ticker:20s} ({sector}): {rows} rows, latest close: {latest_close:.2f}")
        else:
            print(f"  NO DATA {index_ticker:20s} ({sector}): 0 rows returned")
            failed.append((sector, index_ticker))
    except Exception as e:
        print(f"  ERROR   {index_ticker:20s} ({sector}): {e}")
        failed.append((sector, index_ticker))

print(f"\nPassed: {len(SECTOR_INDICES) - len(failed)}/{len(SECTOR_INDICES)}")
if failed:
    print(f"Failed tickers that need replacement:")
    for sector, ticker in failed:
        print(f"  {sector}: {ticker}")
