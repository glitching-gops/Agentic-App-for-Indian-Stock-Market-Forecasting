# model.py — Trains an XGBoost regressor on the computed signals, sentiment, and macro data
# Uses Optuna for hyperparameter tuning and expanding window cross-validation

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from data.db import get_engine
from data.tickers import TICKERS
from pipeline.tuning import tune_hyperparameters, expanding_window_cv

FEATURES = [
    "rsi", "macd_hist", "bb_width", "obv", "sma_20",
    "ema_9", "ema_21", "ema_50", "atr_14", "stoch_k",
    "williams_r", "roc_10", "vroc_10", "prox_52w",
    "lag1_ret", "lag5_ret", "dev_sma50", "bb_upper",
    "bb_lower", "hurst",
    # External signals joined from other tables
    "sentiment_score", "usdinr", "india_vix",
    "nifty_5d_return", "nifty_20d_return",
]
TARGET   = "target"

def load_features_for_ticker(ticker: str, engine):
    """
    Reads signals, sentiment, and macro data from the database for a given ticker.
    Returns (X, y) as a tuple of DataFrames.
    """
    # Load signals
    signals_df = pd.read_sql(f"SELECT * FROM signals WHERE ticker = '{ticker}' ORDER BY date ASC", con=engine)
    if signals_df.empty:
        return pd.DataFrame(), pd.Series()
        
    signals_df.set_index("date", inplace=True)
    
    # Load macro data
    macro_df = pd.read_sql("SELECT * FROM macro ORDER BY date ASC", con=engine)
    macro_df.set_index("date", inplace=True)
    
    # Load sentiment data
    sentiment_df = pd.read_sql(f"SELECT date, sentiment_label, sentiment_score FROM sentiment WHERE ticker = '{ticker}'", con=engine)
    
    # Aggregate sentiment by date
    daily_sentiment = {}
    for _, row in sentiment_df.iterrows():
        d = row["date"]
        # Score = score if positive, -score if negative, 0 if neutral
        score = row["sentiment_score"] if row["sentiment_label"] == "positive" else (-row["sentiment_score"] if row["sentiment_label"] == "negative" else 0)
        if d not in daily_sentiment:
            daily_sentiment[d] = []
        daily_sentiment[d].append(score)
        
    daily_sentiment_avg = {d: np.mean(scores) for d, scores in daily_sentiment.items()}
    
    # Merge signals and macro
    df = signals_df.join(macro_df, how="inner")
    
    # Map sentiment (fallback to 0.0 for historical dates without sentiment data)
    df["sentiment_score"] = df.index.map(lambda d: daily_sentiment_avg.get(d, 0.0))
    
    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with any NaN features
    df.dropna(subset=FEATURES, inplace=True)
    
    if len(df) < 50:
        return pd.DataFrame(), pd.Series()
        
    X = df[FEATURES]
    y = df[TARGET]
    
    return X, y

def train_and_forecast(single_ticker=None):
    engine = get_engine()
    tickers_to_process = [single_ticker] if single_ticker else list(TICKERS.keys())
    results = {}
    
    # Create models dir if not exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    
    for ticker in tickers_to_process:
        print(f"Training model for {ticker}...")
        
        X, y = load_features_for_ticker(ticker, engine)
        
        if X.empty or len(X) < 100:
            print(f"Not enough complete data for {ticker}")
            continue

        # Split into instances where target is known (for training) vs unknown (for prediction)
        train_mask = y.notna()
        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        
        # Test evaluation split (last 15% for directional accuracy check)
        n = len(X_train_full)
        split_idx = int(n * 0.85)
        X_train, y_train = X_train_full.iloc[:split_idx], y_train_full.iloc[:split_idx]
        X_test, y_test   = X_train_full.iloc[split_idx:], y_train_full.iloc[split_idx:]

        # Get tuned parameters — uses saved params if available, runs Optuna if not
        best_params = tune_hyperparameters(ticker, X_train_full, y_train_full)

        # Train final model on full training set using tuned parameters
        model = XGBRegressor(**best_params, random_state=42, verbosity=0)
        model.fit(X_train_full, y_train_full)

        # Evaluate using expanding window CV for a more realistic MAPE estimate
        cv_mape = expanding_window_cv(X_train_full, y_train_full, best_params) * 100

        # Also compute test set directional accuracy on the held-out 15%
        # We need to retrain on the 85% part to avoid data leakage for the test set accuracy
        test_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
        test_model.fit(X_train, y_train)
        y_pred_test = test_model.predict(X_test)
        
        # Directional accuracy: compare actual direction vs predicted direction
        # Get previous close for the test set period
        # We need the original dataframe close prices
        signals_df = pd.read_sql(f"SELECT date, close FROM signals WHERE ticker = '{ticker}'", con=engine)
        signals_df.set_index("date", inplace=True)
        test_prev_close = signals_df.loc[y_test.index, "close"].values
        
        actual_dir = np.where(y_test.values > test_prev_close, 1, 0)
        pred_dir   = np.where(y_pred_test > test_prev_close, 1, 0)
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

        mape = cv_mape  # use CV MAPE as the reported metric

        print(f"{ticker} -> CV MAPE: {mape:.2f}%, Dir Acc: {directional_accuracy:.2f}%")
        
        # Save model
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"{ticker}.joblib")
        joblib.dump(model, model_path)

        # Forecast: use the most recent row (where target is NaN)
        pred_mask = y.isna()
        if pred_mask.any():
            latest_row = X[pred_mask].iloc[[-1]]
            current_price = float(signals_df.loc[latest_row.index[0], "close"])
        else:
            # Fallback if no target is NaN (shouldn't happen with 30-day shift)
            latest_row = X.iloc[[-1]]
            current_price = float(signals_df.iloc[-1]["close"])
            
        forecast_price = float(model.predict(latest_row)[0])
        forecast_date = latest_row.index[0]

        # Determine confidence
        if mape < 8.0 and directional_accuracy > 65.0:
            confidence = "High"
        elif mape <= 12.0 or directional_accuracy >= 55.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        results[ticker] = {
            "current_price":  current_price,
            "forecast_price": forecast_price,
            "mape":           round(mape, 2),
            "directional_accuracy":  round(directional_accuracy, 2),
            "forecast_date":  forecast_date,
            "direction":      "UP" if forecast_price > current_price else "DOWN",
            "change_pct":     round(((forecast_price - current_price) / current_price) * 100, 2),
            "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
            "forecast_confidence": confidence
        }

    return results
