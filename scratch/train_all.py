from data.tickers import TICKERS
from pipeline.model import train_and_forecast
import pandas as pd

results = []
errors  = []

for i, (ticker, info) in enumerate(TICKERS.items()):
    try:
        r_all = train_and_forecast(ticker)
        if ticker not in r_all:
            raise ValueError("No result returned for ticker")
        r = r_all[ticker]
        results.append({
            "ticker":  ticker,
            "sector":  info["sector"],
            "mape":    round(r["mape"], 2),
            "dir_acc": round(r["directional_accuracy"], 2),
            "conf":    r["forecast_confidence"],
            "device":  r.get("device", "unknown"),
        })
        print(
            f"[{i+1}/100] {ticker}: "
            f"MAPE {r['mape']:.2f}% | "
            f"Dir {r['directional_accuracy']:.2f}% | "
            f"{r['forecast_confidence']} | "
            f"{r.get('device', '?')}"
        )
    except Exception as e:
        errors.append((ticker, str(e)))
        print(f"[{i+1}/100] {ticker}: ERROR — {e}")

df = pd.DataFrame(results)
valid = df[df["mape"].notna()]

print(f"\n=== Track B Results (XGBoost + LSTM Ensemble) ===")
print(f"Mean MAPE:                  {valid['mape'].mean():.2f}%")
print(f"Median MAPE:                {valid['mape'].median():.2f}%")
print(f"Mean Directional Accuracy:  {valid['dir_acc'].mean():.2f}%")
print(f"Stocks with MAPE < 8%:      {(valid['mape'] < 8).sum()}")
print(f"Stocks with Dir Acc > 65%:  {(valid['dir_acc'] > 65).sum()}")
print(f"High confidence models:     {(valid['conf'] == 'High').sum()}")
print(f"CUDA used:                  {(valid['device'] == 'cuda').sum()}/{len(valid)}")
print(f"Errors:                     {len(errors)}")

print(f"\n=== Full Progression Table ===")
print(f"{'Metric':<30} {'Pre-C':>10} {'Track C':>10} {'Track A':>10} {'Track B':>10}")
print(f"{'Mean MAPE':<30} {'~11.57%':>10} {'9.44%':>10} {'9.48%':>10} {valid['mape'].mean():.2f}%{'>':>3}")
print(f"{'Mean Dir Accuracy':<30} {'~59.76%':>10} {'59.26%':>10} {'59.62%':>10} {valid['dir_acc'].mean():.2f}%{'>':>3}")
print(f"{'High Confidence':<30} {'N/A':>10} {'17':>10} {'16':>10} {(valid['conf'] == 'High').sum()!s:>10}")
