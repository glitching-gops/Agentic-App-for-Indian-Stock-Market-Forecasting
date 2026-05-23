# main.py — Entry point for the Stock Forecast v1 application
# Runs: DB init → Initial full pipeline run (if empty) → Starts scheduler → Launches Streamlit

import subprocess
import sys
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from data.db import init_db, get_engine
from pipeline.fetch import fetch_and_store
from pipeline.signals import compute_and_store
from pipeline.sentiment import fetch_and_score
from pipeline.macro import fetch_and_store as fetch_macro
from pipeline.model import train_and_forecast
from scheduler import start_scheduler

def is_db_empty():
    engine = get_engine()
    try:
        count = pd.read_sql("SELECT COUNT(*) as c FROM ohlcv", con=engine).iloc[0]["c"]
        return count == 0
    except Exception:
        return True

if __name__ == "__main__":
    print("=== Stock Forecast v1 ===\n")

    # Step 1: Initialise database
    print("[1/4] Initialising database...")
    init_db()

    # Step 2: Initial data fetch if empty
    if is_db_empty():
        print("[2/4] Database is empty. Running initial full pipeline (this will take a while)...")
        fetch_and_store()
        compute_and_store()
        fetch_and_score()
        fetch_macro()
        train_and_forecast()

        # Train TFT jointly on all tickers after initial XGBoost run
        try:
            from pipeline.tft_model import train_tft
            from pipeline.model import load_features_for_ticker
            from data.tickers import TICKERS
            engine = get_engine()
            print("[2/4] Training TFT model on all tickers (this may take a while)...")
            all_data = {}
            for t in TICKERS:
                X, y = load_features_for_ticker(t, engine)
                if not X.empty and len(X) >= 100:
                    df_t = X.copy()
                    df_t["target"] = y
                    sig = pd.read_sql(
                        f"SELECT date, close FROM signals WHERE ticker = '{t}'", con=engine
                    )
                    sig.set_index("date", inplace=True)
                    df_t["close"] = sig.loc[df_t.index, "close"]
                    all_data[t] = df_t
            if all_data:
                train_tft(all_data)
                print("[2/4] TFT training complete.")
        except Exception as e:
            print(f"Warning: TFT training failed — {e}")
    else:
        print("[2/4] Database already contains data. Skipping initial fetch.")

    # Step 2.5: Initialise LangGraph State for all stocks
    from agents.graph import run_graph

    print("[2.5/4] Pre-warming LangGraph states...")
    # We can pre-run the graph for RELIANCE.NS just to make sure it compiles
    try:
        run_graph("RELIANCE.NS")
    except Exception as e:
        print(f"Warning: Failed to pre-warm graph: {e}")

    # Step 3: Start Scheduler
    print("[3/4] Starting background scheduler...")
    start_scheduler()

    # Step 4: Launch Streamlit dashboard
    print("[4/4] Launching dashboard...\n")
    dashboard_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except KeyboardInterrupt:
        print("\nShutting down...")
