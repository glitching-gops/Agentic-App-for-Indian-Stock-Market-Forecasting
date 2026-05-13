import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from data.tickers import TICKERS
from pipeline.model import train_and_forecast
from data.db import init_db

import warnings
warnings.filterwarnings('ignore')

def run_track_b():
    # Ensure DB schema is up to date
    init_db()
    
    results = []
    start_time = time.time()
    
    print(f"=== Starting Track B Training & Evaluation ({len(TICKERS)} stocks) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, (ticker, info) in enumerate(TICKERS.items()):
        try:
            print(f"[{i+1}/{len(TICKERS)}] Processing {ticker}...", end=" ", flush=True)
            # Use force=False for LSTM to speed up if already trained, but force meta-learner?
            # Actually, train_and_forecast will train meta-learner anyway if needed.
            r_dict = train_and_forecast(ticker)
            
            if ticker in r_dict:
                r = r_dict[ticker]
                res = {
                    "ticker": ticker,
                    "xgb_mape": r.get("xgb_mape"),
                    "xgb_dir_acc": r.get("xgb_dir_acc"),
                    "ensemble_mape": r.get("mape"),
                    "ensemble_dir_acc": r.get("directional_accuracy"),
                    "conf": r.get("forecast_confidence")
                }
                results.append(res)
                print(f"MAPE: {r.get('mape'):.2f}% | Dir: {r.get('directional_accuracy'):.2f}%")
                
                # Incremental save
                pd.DataFrame(results).to_csv("track_b_results.csv", index=False)
            else:
                print(f"Error: {ticker} not in results")
                
        except Exception as e:
            print(f"  FAILED {ticker}: {e}")
            
    # Save results to CSV for record keeping
    df = pd.DataFrame(results)
    df.to_csv("track_b_results.csv", index=False)
    
    # Report Aggregate Stats
    print("\n" + "="*50)
    print("=== TRACK B FINAL RESULTS (ENSEMBLE) ===")
    print("="*50)
    
    if not df.empty:
        valid = df[df["ensemble_mape"].notna()]
        
        xgb_mean_mape = valid["xgb_mape"].mean()
        ens_mean_mape = valid["ensemble_mape"].mean()
        
        xgb_mean_dir = valid["xgb_dir_acc"].mean()
        ens_mean_dir = valid["ensemble_dir_acc"].mean()
        
        print(f"Mean XGB MAPE:      {xgb_mean_mape:.2f}%")
        print(f"Mean Ensemble MAPE: {ens_mean_mape:.2f}% (Delta: {ens_mean_mape - xgb_mean_mape:.2f}%)")
        print("-" * 30)
        print(f"Mean XGB Dir Acc:   {xgb_mean_dir:.2f}%")
        print(f"Mean Ensemble Dir:  {ens_mean_dir:.2f}% (Delta: {ens_mean_dir - xgb_mean_dir:.2f}%)")
        print("-" * 30)
        print(f"High Conf Models:   {(valid['conf'] == 'High').sum()}")
        print(f"Total Stocks:       {len(valid)}")
        
        # Progression Table Comparison (Track A values from summary)
        print("\n=== PROGRESSION TABLE ===")
        print(f"{'Metric':<20} | {'Track A':<10} | {'Track B':<10}")
        print("-" * 45)
        print(f"{'Mean MAPE':<20} | {'9.48%':<10} | {ens_mean_mape:>7.2f}%")
        print(f"{'Mean Dir Acc':<20} | {'59.62%':<10} | {ens_mean_dir:>7.2f}%")
        print(f"{'High Conf Models':<20} | {'16':<10} | {str((valid['conf'] == 'High').sum()):>7}")
    else:
        print("No valid results to report.")
        
    end_time = time.time()
    print(f"\nTotal Time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    run_track_b()
