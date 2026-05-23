# model.py — Trains XGBoost on log-return target, combines with TFT and TimesFM
# via a Ridge meta-learner.  Parallel ticker training via ThreadPoolExecutor.

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from data.db import get_engine
from data.tickers import TICKERS, get_sector
from pipeline.tuning import tune_hyperparameters, expanding_window_cv
from pipeline.tft_model import predict_tft
from pipeline.timesfm_model import predict_timesfm
from pipeline.meta_learner import train_meta_learner, predict_ensemble

def _mlflow_log(
    ticker: str,
    best_params: dict,
    cv_mape: float,
    directional_accuracy: float,
    ensemble_mape: float,
    ensemble_dir_acc: float,
    model_path: str,
) -> None:
    """
    Logs one training run to MLflow.  Completely silent if MLFLOW_TRACKING_URI
    is not set — the pipeline never depends on this succeeding.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return
    try:
        import mlflow

        if "dagshub.com" in tracking_uri:
            # dagshub.init() sets the tracking URI + auth in one call
            import dagshub
            # Parse owner/repo from https://dagshub.com/{owner}/{repo}.mlflow
            path = tracking_uri.replace("https://dagshub.com/", "").replace(".mlflow", "").strip("/")
            parts = path.split("/")
            if len(parts) == 2:
                owner, repo = parts
                dagshub.init(
                    repo_owner=owner,
                    repo_name=repo,
                    mlflow=True,
                    token=os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD"),
                )
        else:
            # Standard MLflow server — manual auth
            username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME", "")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN", "")
            if username:
                os.environ["MLFLOW_TRACKING_USERNAME"] = username
            if password:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("stock-forecast-v2")
        with mlflow.start_run(run_name=ticker):
            mlflow.set_tag("ticker", ticker)
            mlflow.set_tag("sector", get_sector(ticker))
            mlflow.set_tag("training_date", datetime.utcnow().strftime("%Y-%m-%d"))
            # Hyperparams (skip infrastructure keys)
            mlflow.log_params({
                k: v for k, v in best_params.items()
                if k not in ("tree_method", "device")
            })
            # Metrics
            mlflow.log_metrics({
                "xgb_cv_mape":      round(cv_mape, 4),
                "xgb_dir_acc":      round(directional_accuracy, 4),
                "ensemble_mape":    round(ensemble_mape, 4),
                "ensemble_dir_acc": round(ensemble_dir_acc, 4),
            })
            # Trained model artifact
            mlflow.log_artifact(model_path, artifact_path="models/joblib")
            meta_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "meta",
                f"{ticker.replace('.', '_')}_meta.joblib",
            )
            if os.path.exists(meta_path):
                mlflow.log_artifact(meta_path, artifact_path="models/meta")
    except Exception as e:
        print(f"[MLflow] {ticker}: logging failed — {e}")


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
    # Sector + earnings — Track A
    "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
    "earnings_surprise",
]
TARGET = "target"   # log(close_t+30 / close_t)


def classify_confidence(mape: float, dir_acc: float) -> str:
    if mape < 8.0 and dir_acc > 65.0:
        return "High"
    if mape <= 12.0 or dir_acc >= 55.0:
        return "Medium"
    return "Low"


def load_features_for_ticker(ticker: str, engine):
    """
    Reads signals, sentiment, and macro data from the database for a given ticker.
    Returns (X, y) where y is the 30-day log-return target.
    """
    signals_df = pd.read_sql(
        f"SELECT * FROM signals WHERE ticker = '{ticker}' ORDER BY date ASC",
        con=engine,
    )
    if signals_df.empty:
        return pd.DataFrame(), pd.Series()

    signals_df.set_index("date", inplace=True)

    macro_df = pd.read_sql("SELECT * FROM macro ORDER BY date ASC", con=engine)
    macro_df.set_index("date", inplace=True)

    sentiment_df = pd.read_sql(
        f"SELECT date, sentiment_label, sentiment_score FROM sentiment WHERE ticker = '{ticker}'",
        con=engine,
    )

    daily_sentiment: dict = {}
    for _, row in sentiment_df.iterrows():
        d = row["date"]
        score = (
            row["sentiment_score"]
            if row["sentiment_label"] == "positive"
            else (-row["sentiment_score"] if row["sentiment_label"] == "negative" else 0)
        )
        daily_sentiment.setdefault(d, []).append(score)
    daily_sentiment_avg = {d: np.mean(v) for d, v in daily_sentiment.items()}

    df = signals_df.join(macro_df, how="inner")
    df["sentiment_score"] = df.index.map(lambda d: daily_sentiment_avg.get(d, 0.0))

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(df) < 50:
        return pd.DataFrame(), pd.Series()

    return df[FEATURES], df[TARGET]


def _train_single_ticker(ticker: str, engine) -> dict | None:
    """Trains XGBoost + coordinates TFT/TimesFM for one ticker. Returns result dict or None."""
    print(f"Training model for {ticker}...")

    X, y = load_features_for_ticker(ticker, engine)
    if X.empty or len(X) < 100:
        print(f"Not enough data for {ticker}")
        return None

    train_mask = y.notna()
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]

    n = len(X_train_full)
    split_idx = int(n * 0.85)
    X_train, y_train = X_train_full.iloc[:split_idx], y_train_full.iloc[:split_idx]
    X_test, y_test = X_train_full.iloc[split_idx:], y_train_full.iloc[split_idx:]

    # ── XGBoost ──────────────────────────────────────────────────────────────
    best_params = tune_hyperparameters(ticker, X_train_full, y_train_full)
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    model.fit(X_train_full, y_train_full)

    cv_mape = expanding_window_cv(X_train_full, y_train_full, best_params) * 100

    test_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    test_model.fit(X_train, y_train)
    xgb_val_preds = test_model.predict(X_test)   # log-return predictions

    # Directional accuracy: positive log-return = UP
    actual_dir = (y_test.values > 0).astype(int)
    pred_dir = (xgb_val_preds > 0).astype(int)
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    os.makedirs(
        os.path.join(os.path.dirname(__file__), "..", "models", "joblib"), exist_ok=True
    )
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "joblib", f"{ticker}.joblib"
    )
    joblib.dump(model, model_path)

    # Latest row for inference (where target is NaN = no 30-day future yet)
    signals_df = pd.read_sql(
        f"SELECT date, close FROM signals WHERE ticker = '{ticker}'", con=engine
    )
    signals_df.set_index("date", inplace=True)

    pred_mask = y.isna()
    if pred_mask.any():
        latest_row = X[pred_mask].iloc[[-1]]
        current_price = float(signals_df.loc[latest_row.index[0], "close"])
    else:
        latest_row = X.iloc[[-1]]
        current_price = float(signals_df.iloc[-1]["close"])

    # Back-transform: price = current_price * exp(log_return)
    xgb_log_return = float(model.predict(latest_row)[0])
    xgb_price = current_price * np.exp(xgb_log_return)

    # ── TFT ──────────────────────────────────────────────────────────────────
    df_full = X.copy()
    df_full["target"] = y
    df_full["close"] = signals_df.loc[df_full.index, "close"]

    try:
        tft_log_return = predict_tft(ticker, df_full.copy())
        tft_price = current_price * np.exp(tft_log_return) if tft_log_return is not None else None
    except Exception as e:
        print(f"[Model] {ticker}: TFT failed — {e}")
        tft_price = None

    # ── TimesFM ──────────────────────────────────────────────────────────────
    try:
        tfm_log_return = predict_timesfm(ticker, df_full["close"].values)
        tfm_price = current_price * np.exp(tfm_log_return) if tfm_log_return is not None else None
    except Exception as e:
        print(f"[Model] {ticker}: TimesFM failed — {e}")
        tfm_price = None

    # ── Meta-learner ─────────────────────────────────────────────────────────
    hurst_val = X_test["hurst"].values if "hurst" in X_test.columns else np.full(len(y_test), 0.5)

    # Build val-set TFT/TimesFM preds for meta-learner training
    # Approximate: use xgb_val_preds back-transformed for unavailable models
    val_close = signals_df.loc[y_test.index, "close"].values
    xgb_val_prices = val_close * np.exp(xgb_val_preds)
    y_val_prices = val_close * np.exp(y_test.values)

    # Placeholder val predictions for TFT/TimesFM when not yet available per-row
    tft_val_prices = xgb_val_prices   # fallback; overwritten when TFT has val preds
    tfm_val_prices = xgb_val_prices

    meta = train_meta_learner(
        ticker,
        xgb_val_preds=xgb_val_prices,
        tft_val_preds=tft_val_prices,
        tfm_val_preds=tfm_val_prices,
        hurst_val=hurst_val,
        y_val=y_val_prices,
        force=False,
    )

    current_hurst = float(df_full["hurst"].iloc[-1]) if "hurst" in df_full.columns else 0.5
    ensemble_price = predict_ensemble(
        ticker=ticker,
        xgb_price=xgb_price,
        tft_price=tft_price,
        tfm_price=tfm_price,
        current_hurst=current_hurst,
        current_price=current_price,
    )

    # Ensemble MAPE on validation set (in price space for reporting)
    ensemble_val = meta.predict(
        np.column_stack([xgb_val_prices, tft_val_prices, tfm_val_prices, hurst_val])
    ) if meta else xgb_val_prices
    ensemble_mape = float(
        mean_absolute_percentage_error(y_val_prices, ensemble_val) * 100
    ) if meta else cv_mape

    ens_actual_dir = (y_test.values > 0).astype(int)
    ens_pred_dir = (
        (ensemble_val > val_close).astype(int) if meta
        else pred_dir
    )
    ensemble_dir_acc = float(np.mean(ens_actual_dir == ens_pred_dir) * 100) if meta else directional_accuracy

    confidence = classify_confidence(ensemble_mape, ensemble_dir_acc)

    # ── Persist model metadata ───────────────────────────────────────────────
    with engine.connect() as conn:
        conn.execute(
            text("""
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
            """),
            {
                "ticker": ticker,
                "xgb_mape": cv_mape,
                "xgb_dir_acc": directional_accuracy,
                "lstm_val_mape": None,
                "ensemble_mape": ensemble_mape,
                "ensemble_dir_acc": ensemble_dir_acc,
                "lstm_epochs": 0,
                "meta_xgb": float(meta.coef_[0]) if meta is not None else 0.5,
                "meta_lstm": float(meta.coef_[1]) if meta is not None else 0.25,
                "meta_hurst": float(meta.coef_[3]) if meta is not None else 0.0,
                "in_use": 1,
                "trained": datetime.utcnow(),
            },
        )
        conn.commit()

    # ── MLflow run logging (opt-in — only when MLFLOW_TRACKING_URI is set) ──────
    _mlflow_log(
        ticker=ticker,
        best_params=best_params,
        cv_mape=cv_mape,
        directional_accuracy=directional_accuracy,
        ensemble_mape=ensemble_mape,
        ensemble_dir_acc=ensemble_dir_acc,
        model_path=model_path,
    )

    return {
        "current_price": current_price,
        "xgb_forecast_price": xgb_price,
        "tft_forecast_price": tft_price,
        "tfm_forecast_price": tfm_price,
        "forecast_price": ensemble_price,
        "xgb_mape": round(cv_mape, 2),
        "xgb_dir_acc": round(directional_accuracy, 2),
        "mape": round(ensemble_mape, 2),
        "dir_acc": round(ensemble_dir_acc, 2),
        "direction": "UP" if ensemble_price > current_price else "DOWN",
        "change_pct": round(((ensemble_price - current_price) / current_price) * 100, 2),
        "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
        "forecast_confidence": confidence,
    }


def train_and_forecast(single_ticker=None):
    """
    Trains all models and stores results.  Uses a ThreadPoolExecutor so
    multiple tickers run in parallel (I/O-bound DB reads overlap with
    CPU-bound XGBoost training on different cores).
    """
    engine = get_engine()
    tickers_to_process = [single_ticker] if single_ticker else list(TICKERS.keys())
    results = {}

    if single_ticker:
        res = _train_single_ticker(single_ticker, engine)
        if res:
            results[single_ticker] = res
        return results

    max_workers = min(4, len(tickers_to_process))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_train_single_ticker, ticker, engine): ticker
            for ticker in tickers_to_process
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                res = future.result()
                if res:
                    results[ticker] = res
                    print(f"{ticker} -> MAPE: {res['mape']:.2f}%, Dir Acc: {res['dir_acc']:.2f}%")
            except Exception as e:
                print(f"[Model] {ticker}: failed — {e}")

    return results
