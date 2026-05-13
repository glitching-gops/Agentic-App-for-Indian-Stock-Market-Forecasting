import pandas as pd
import os

if os.path.exists('track_b_results.csv'):
    df = pd.read_csv('track_b_results.csv')
    print(f"Count: {len(df)}")
    if not df.empty:
        print(f"Mean XGB MAPE: {df['xgb_mape'].mean():.2f}%")
        print(f"Mean Ens MAPE: {df['ensemble_mape'].mean():.2f}%")
        print(f"Mean XGB Dir: {df['xgb_dir_acc'].mean():.2f}%")
        print(f"Mean Ens Dir: {df['ensemble_dir_acc'].mean():.2f}%")
else:
    print("CSV not found.")
