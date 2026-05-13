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
from pipeline.lstm_model import train_lstm, predict_lstm
from pipeline.meta_learner import train_meta_learner, predict_ensemble

FEATURES = [
    # Technical signals (20)
    "rsi", "macd_hist", "bb_width", "obv", "sma_20",
    "ema_9", "ema_21", "ema_50", "atr_14", "stoch_k",
    "williams_r", "roc_10", "vroc_10", "prox_52w",
    "lag1_ret", "lag5_ret", "dev_sma50", "bb_upper",
    "bb_lower", "hurst",
    # Sentiment
    "sentiment_score",
    # Macro signals
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
    # New signals — Track A
    "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
    "earnings_surprise",
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
    
    # Defensive feature filling
    for col in FEATURES:
        if col not in df.columns:
            print(f"[Model] {ticker}: feature '{col}' not found — filling with 0.0")
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
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
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models", "joblib"), exist_ok=True)
    
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
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "joblib", f"{ticker}.joblib")
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

        df_full = X.copy()
        df_full['target'] = y
        df_full['close'] = signals_df.loc[df_full.index, 'close']

        # ── LSTM training ────────────────────────────────────────────────────────────
        print(f"[Model] {ticker}: training LSTM...")
        try:
            lstm_result = train_lstm(ticker, df_full.copy(), force=False)
            lstm_price = predict_lstm(ticker, df_full.copy())
            lstm_device = lstm_result.get("device")
        except Exception as e:
            print(f"[Model] {ticker}: LSTM failed — {e}")
            lstm_price = None
            lstm_device = "cpu"
            lstm_result = {}

        # ── Meta-learner training ────────────────────────────────────────────────────
        try:
            val_df = df_full.iloc[split_idx:n].copy()
            lstm_val_preds = []

            for i in range(len(val_df)):
                context_start = max(0, split_idx + i - 30)
                context_df    = df_full.iloc[context_start:split_idx + i].copy()
                pred = predict_lstm(ticker, context_df)
                lstm_val_preds.append(pred if pred is not None else y_pred_test[i])

            lstm_val_preds = np.array(lstm_val_preds)
            hurst_val      = X_test["hurst"].values if "hurst" in X_test.columns \
                             else np.full(len(y_test), 0.5)

            meta = train_meta_learner(
                ticker,
                xgb_val_preds  = y_pred_test,
                lstm_val_preds = lstm_val_preds,
                hurst_val      = hurst_val,
                y_val          = y_test.values,
                force          = False
            )
        except Exception as e:
            print(f"[Model] {ticker}: meta-learner training failed — {e}")
            meta = None
            ensemble_mape = mape
            ensemble_dir_acc = directional_accuracy

        if meta:
            try:
                # Calculate ensemble MAPE on validation set
                # Re-align lengths as in train_meta_learner
                min_len = min(len(y_pred_test), len(lstm_val_preds), len(hurst_val), len(y_test))
                X_meta_test = np.column_stack([
                    y_pred_test[-min_len:],
                    lstm_val_preds[-min_len:],
                    hurst_val[-min_len:]
                ])
                y_meta_test = y_test.values[-min_len:]
                
                ensemble_val_preds = meta.predict(X_meta_test)
                ensemble_mape = float(mean_absolute_percentage_error(y_meta_test, ensemble_val_preds) * 100)
                
                # Calculate ensemble directional accuracy
                # Need prev_close for y_meta_test
                test_prev_close_meta = test_prev_close[-min_len:]
                actual_dir_meta = np.where(y_meta_test > test_prev_close_meta, 1, 0)
                pred_dir_meta   = np.where(ensemble_val_preds > test_prev_close_meta, 1, 0)
                ensemble_dir_acc = float(np.mean(actual_dir_meta == pred_dir_meta) * 100)
            except Exception as e:
                print(f"[Model] {ticker}: ensemble evaluation failed — {e}")
                ensemble_mape = mape
                ensemble_dir_acc = directional_accuracy
        else:
            ensemble_mape = mape
            ensemble_dir_acc = directional_accuracy

        # ── Ensemble final forecast ──────────────────────────────────────────────────
        current_hurst  = float(df_full["hurst"].iloc[-1]) \
                         if "hurst" in df_full.columns else 0.5

        ensemble_price = predict_ensemble(
            ticker        = ticker,
            xgb_price     = forecast_price,
            lstm_price    = lstm_price,
            current_hurst = current_hurst,
            current_price = current_price,
        )

        results[ticker] = {
            "current_price":  current_price,
            "xgb_forecast_price": forecast_price,
            "lstm_forecast_price": lstm_price,
            "forecast_price": ensemble_price,
            "xgb_mape":       round(mape, 2),
            "xgb_dir_acc":    round(directional_accuracy, 2),
            "mape":           round(ensemble_mape, 2),
            "directional_accuracy":  round(ensemble_dir_acc, 2),
            "forecast_date":  forecast_date,
            "direction":      "UP" if ensemble_price > current_price else "DOWN",
            "change_pct":     round(((ensemble_price - current_price) / current_price) * 100, 2),
            "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
            "forecast_confidence": confidence,
            "device": lstm_device
        }

        # Step 5 - write model metadata to database
        from sqlalchemy import text
        from datetime import datetime
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_metadata (
                    ticker, xgb_mape, xgb_dir_acc, lstm_val_mape,
                    ensemble_mape, ensemble_dir_acc,
                    lstm_epochs_trained, meta_xgb_coef, meta_lstm_coef,
                    meta_hurst_coef, ensemble_in_use, last_trained
                ) VALUES (
                    :ticker, :xgb_mape, :xgb_dir_acc, :lstm_val_mape,
                    :ensemble_mape, :ensemble_dir_acc,
                    :lstm_epochs, :meta_xgb, :meta_lstm, :meta_hurst,
                    :in_use, :trained
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    xgb_mape            = EXCLUDED.xgb_mape,
                    xgb_dir_acc         = EXCLUDED.xgb_dir_acc,
                    lstm_val_mape       = EXCLUDED.lstm_val_mape,
                    ensemble_mape       = EXCLUDED.ensemble_mape,
                    ensemble_dir_acc    = EXCLUDED.ensemble_dir_acc,
                    lstm_epochs_trained = EXCLUDED.lstm_epochs_trained,
                    meta_xgb_coef       = EXCLUDED.meta_xgb_coef,
                    meta_lstm_coef      = EXCLUDED.meta_lstm_coef,
                    meta_hurst_coef     = EXCLUDED.meta_hurst_coef,
                    ensemble_in_use     = EXCLUDED.ensemble_in_use,
                    last_trained        = EXCLUDED.last_trained
            """), {
                "ticker":    ticker,
                "xgb_mape":  mape,
                "xgb_dir_acc": directional_accuracy,
                "lstm_val_mape": lstm_result.get("val_mape"),
                "ensemble_mape": ensemble_mape,
                "ensemble_dir_acc": ensemble_dir_acc,
                "lstm_epochs":   lstm_result.get("epochs_trained"),
                "meta_xgb":  float(meta.coef_[0]) if meta is not None else 0.5,
                "meta_lstm": float(meta.coef_[1]) if meta is not None else 0.5,
                "meta_hurst":float(meta.coef_[2]) if meta is not None else 0.0,
                "in_use":    1,
                "trained":   datetime.utcnow(),
            })
            conn.commit()

    return results
