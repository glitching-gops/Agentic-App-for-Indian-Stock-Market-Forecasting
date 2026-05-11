# scheduler.py — Runs the pipeline daily at 6:30 PM IST

import time
import logging
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

from pipeline.fetch import fetch_and_store
from pipeline.signals import compute_and_store
from pipeline.sentiment import fetch_and_score
from pipeline.macro import fetch_and_store as fetch_macro
from pipeline.model import train_and_forecast, load_features_for_ticker
from pipeline.tuning import tune_hyperparameters

# Setup logging to file
os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "scheduler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LOCK_FILE = os.path.join(os.path.dirname(__file__), "scheduler.lock")

def run_pipeline_job():
    logger.info("Starting scheduled pipeline run...")
    try:
        # Step 1: Fetch OHLCV
        logger.info("[1/5] Fetching OHLCV data...")
        fetch_and_store()
        
        # Step 2: Compute signals
        logger.info("[2/5] Computing signals...")
        compute_and_store()
        
        # Step 3: Fetch sentiment
        logger.info("[3/5] Fetching news sentiment...")
        fetch_and_score()
        
        # Step 4: Fetch macro data
        logger.info("[4/5] Fetching macro data...")
        fetch_macro()
        
        # Step 5: Train models
        logger.info("[5/5] Training models...")
        train_and_forecast()
        
        logger.info("Scheduled pipeline run completed successfully.")
    except Exception as e:
        logger.error(f"Error during scheduled pipeline run: {e}")

def weekly_retune_all():
    """
    Runs full Optuna retuning (force=True) for all 100 stocks.
    Scheduled weekly on Sunday at 02:00 IST.
    Saves best params to tuned_params/ for use in daily fast retrains.
    """
    from data.tickers import TICKERS
    from data.db import get_engine
    import pandas as pd

    engine = get_engine()
    logger.info(f"[Scheduler] Weekly retune started for {len(TICKERS)} stocks")

    for ticker in TICKERS.keys():
        try:
            # Load feature matrix for this ticker
            X, y = load_features_for_ticker(ticker, engine)
            if len(X) < 100:
                logger.info(f"[Scheduler] {ticker}: insufficient data, skipping")
                continue
            tune_hyperparameters(ticker, X, y, force=True)
            
            from pipeline.lstm_model import train_lstm
            from pipeline.meta_learner import train_meta_learner

            try:
                df_full = X.copy()
                df_full['target'] = y
                signals_df = pd.read_sql(f"SELECT date, close FROM signals WHERE ticker = '{ticker}'", con=engine)
                signals_df.set_index("date", inplace=True)
                df_full['close'] = signals_df.loc[df_full.index, 'close']
                train_lstm(ticker, df_full.copy(), force=True)
                logger.info(f"[Scheduler] {ticker}: LSTM retrained")
            except Exception as e:
                logger.error(f"[Scheduler] {ticker}: LSTM retrain failed — {e}")

        except Exception as e:
            logger.error(f"[Scheduler] {ticker}: retuning failed — {e}")

    logger.info("[Scheduler] Weekly retune complete")

def start_scheduler():
    # Simple duplicate prevention
    if os.path.exists(LOCK_FILE):
        logger.warning("Scheduler lock file exists. Assuming scheduler is already running in another process.")
        return None
        
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
        
    import atexit
    def remove_lock():
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    atexit.register(remove_lock)

    ist_tz = pytz.timezone('Asia/Kolkata')
    scheduler = BackgroundScheduler(timezone=ist_tz)
    
    # Run at 18:30 IST daily
    scheduler.add_job(run_pipeline_job, 'cron', hour=18, minute=30, id='pipeline_job', replace_existing=True)
    
    # Run weekly full Optuna retune on Sunday at 02:00 IST
    scheduler.add_job(
        weekly_retune_all,
        trigger="cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        id="weekly_retune",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Scheduler started. Pipeline will run daily at 18:30 IST.")
    return scheduler

if __name__ == "__main__":
    scheduler = start_scheduler()
    if scheduler:
        try:
            while True:
                time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            pass
