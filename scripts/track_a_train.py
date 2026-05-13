import sys
import os
sys.path.append(os.getcwd())

import glob

# Remove existing models
for f in glob.glob("models/*.joblib"):
    os.remove(f)

from data.tickers import TICKERS
from pipeline.model import train_and_forecast
import pandas as pd

results = []
for i, (ticker, info) in enumerate(TICKERS.items()):
    try:
        r_dict = train_and_forecast(ticker)
        r = r_dict.get(ticker, {})
        results.append({
            "ticker":  ticker,
            "sector":  info["sector"],
            "mape":    round(r.get("mape", float('nan')), 2),
            "dir_acc": round(r.get("directional_accuracy", float('nan')), 2),
            "conf":    r.get("forecast_confidence", "Unknown"),
        })
        print(
            f"[{i+1}/100] {ticker}: "
            f"MAPE {r.get('mape', float('nan')):.2f}% | "
            f"Dir {r.get('directional_accuracy', float('nan')):.2f}% | "
            f"{r.get('forecast_confidence', 'Unknown')}"
        )
    except Exception as e:
        results.append({
            "ticker":  ticker,
            "sector":  info["sector"],
            "error":   str(e)
        })
        print(f"[{i+1}/100] {ticker}: ERROR — {e}")

df = pd.DataFrame(results)
valid = df[df["mape"].notna()]

print(f"\n=== Track A Results ===")
print(f"Mean MAPE:                  {valid['mape'].mean():.2f}%")
print(f"Median MAPE:                {valid['mape'].median():.2f}%")
print(f"Mean Directional Accuracy:  {valid['dir_acc'].mean():.2f}%")
print(f"Stocks with MAPE < 8%:      {(valid['mape'] < 8).sum()}")
print(f"Stocks with Dir Acc > 65%:  {(valid['dir_acc'] > 65).sum()}")
print(f"High confidence models:     {(valid['conf'] == 'High').sum()}")
print(f"Medium confidence models:   {(valid['conf'] == 'Medium').sum()}")
print(f"Low confidence models:      {(valid['conf'] == 'Low').sum()}")
print(f"Errors:                     {df['error'].notna().sum() if 'error' in df.columns else 0}")

print(f"\n=== Comparison Table ===")
print(f"{'Metric':<30} {'Pre-Track-C':>12} {'Track C':>12} {'Track A':>12}")
print(f"{'Mean MAPE':<30} {'~11.57%':>12} {'9.44%':>12} {valid['mape'].mean():.2f}%{'>':>5}")
print(f"{'Mean Dir Accuracy':<30} {'~59.76%':>12} {'59.26%':>12} {valid['dir_acc'].mean():.2f}%{'>':>5}")
print(f"{'High Confidence Models':<30} {'N/A':>12} {'17':>12} {(valid['conf'] == 'High').sum()!s:>12}")
