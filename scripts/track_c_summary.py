import sys
import os
sys.path.append(os.getcwd())

from data.tickers import TICKERS
from pipeline.model import train_and_forecast
import pandas as pd

results = []
print("Retraining all models and generating summary...")
for ticker, info in TICKERS.items():
    try:
        r_dict = train_and_forecast(ticker)
        if ticker in r_dict:
            r = r_dict[ticker]
            results.append({
                "ticker":   ticker,
                "company":  info["company"],
                "sector":   info["sector"],
                "mape":     round(r["mape"], 2),
                "dir_acc":  round(r["directional_accuracy"], 2),
                "conf":     r["forecast_confidence"],
            })
        else:
            results.append({
                "ticker":  ticker,
                "company": info["company"],
                "error":   "Not enough data"
            })
    except Exception as e:
        results.append({
            "ticker":  ticker,
            "company": info["company"],
            "error":   str(e)
        })

df = pd.DataFrame(results)
print(df.to_string())

if "mape" in df.columns:
    valid = df[df["mape"].notna()]
    print(f"\n=== Aggregate Stats (Track C — XGBoost with Optuna + Expanding CV) ===")
    print(f"Mean MAPE:                {valid['mape'].mean():.2f}%")
    print(f"Median MAPE:              {valid['mape'].median():.2f}%")
    print(f"Mean Dir Accuracy:        {valid['dir_acc'].mean():.2f}%")
    print(f"Stocks MAPE < 8%:         {(valid['mape'] < 8).sum()}")
    print(f"Stocks Dir Acc > 65%:     {(valid['dir_acc'] > 65).sum()}")
    print(f"High confidence:          {(valid['conf'] == 'High').sum()}")
else:
    print("No valid results found.")
