# signals.py — Computes technical indicators from OHLCV data
# Uses the `ta` library and pandas to compute 15 signals
# Target: closing price 30 days forward

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from data.db import get_engine
from data.tickers import TICKERS

def compute_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Computes the Hurst exponent for a price series using the
    rescaled range (R/S) method.
    Returns a float between 0 and 1:
      > 0.5 = trending (persistent)
      < 0.5 = mean-reverting (anti-persistent)
      ~ 0.5 = random walk
    Falls back to 0.5 if computation fails.
    """
    try:
        lags = range(2, max_lag)
        tau = [
            np.std(np.subtract(series[lag:].values, series[:-lag].values))
            for lag in lags
        ]
        # Fit a line to log(lag) vs log(tau)
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return round(float(poly[0]), 4)
    except Exception:
        return 0.5

def compute_and_store(single_ticker=None):
    engine = get_engine()
    tickers_to_fetch = [single_ticker] if single_ticker else list(TICKERS.keys())
    
    total_new_rows = 0
    for ticker in tickers_to_fetch:
        print(f"Computing signals for {ticker}...")
        
        # Load raw OHLCV for this ticker
        df = pd.read_sql(f"SELECT * FROM ohlcv WHERE ticker = '{ticker}' ORDER BY date ASC", con=engine)
        if df.empty or len(df) < 50:
            print(f"Not enough data to compute signals for {ticker}")
            continue
            
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # ── 1. RSI (14-period) ──────────────────────────────────────────────────
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

        # ── 2. MACD Histogram ───────────────────────────────────────────────────
        df["macd_hist"] = MACD(close=df["close"]).macd_diff()

        # ── 3. Bollinger Band Width ─────────────────────────────────────────────
        df["bb_width"] = BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_wband()

        # ── 4. On-Balance Volume ────────────────────────────────────────────────
        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

        # ── 5. 20-day SMA ───────────────────────────────────────────────────────
        df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
        
        # ── 5a. 50-day EMA ──────────────────────────────────────────────────────
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        
        # ── 5b. Bollinger Upper & Lower ─────────────────────────────────────────
        df["bb_upper"] = BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_hband()
        df["bb_lower"] = BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_lband()
        
        # ── 6. 9-day EMA ────────────────────────────────────────────────────────
        df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
        
        # ── 7. 21-day EMA ───────────────────────────────────────────────────────
        df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
        
        # ── 8. ATR (14-period) ──────────────────────────────────────────────────
        df["atr_14"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        
        # ── 9. Stochastic %K (14-period) ────────────────────────────────────────
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["stoch_k"] = stoch.stoch()
        
        # ── 10. Williams %R (14-period) ─────────────────────────────────────────
        df["williams_r"] = WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14).williams_r()
        
        # ── 11. Price Rate of Change (10-period) ────────────────────────────────
        df["roc_10"] = ROCIndicator(close=df["close"], window=10).roc()
        
        # ── 12. Volume Rate of Change (10-period) ───────────────────────────────
        # Volume Rate of Change — safe version
        vroc_raw = df["volume"].pct_change(10)
        df["vroc_10"] = vroc_raw.replace([float("inf"), float("-inf")], float("nan"))
        
        # ── 13. 52-week High Proximity ──────────────────────────────────────────
        # ~252 trading days in a year
        rolling_high = df["high"].rolling(window=252, min_periods=50).max()
        rolling_low = df["low"].rolling(window=252, min_periods=50).min()
        range_52w = rolling_high - rolling_low
        df["prox_52w"] = np.where(range_52w == 0, 0.5, (df["close"] - rolling_low) / range_52w)
        
        # ── 14. Lag-1 Daily Return ──────────────────────────────────────────────
        df["lag1_ret"] = df["close"].pct_change(1)
        
        # ── 15. Lag-5 Daily Return ──────────────────────────────────────────────
        df["lag5_ret"] = df["close"].pct_change(5)
        
        # ── 16. Price deviation from SMA-50 ─────────────────────────────────────
        sma_50 = SMAIndicator(close=df["close"], window=50).sma_indicator()
        df["dev_sma50"] = (df["close"] - sma_50) / sma_50 * 100

        # ── 17. Hurst Exponent (60-day rolling) ─────────────────────────────────
        df["hurst"] = df["close"].rolling(window=60).apply(
            lambda x: compute_hurst_exponent(pd.Series(x)), raw=False
        )

        # ── Target: closing price 30 days forward ───────────────────────────────
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["target"] = df["close"].shift(-30)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["target"], inplace=True)
        # We don't drop rows with NaN target here because we want to predict for today!
        # The model training step will filter out NaN targets.
        feature_cols = ["rsi", "macd_hist", "bb_width", "obv", "sma_20", "ema_50", "bb_upper", "bb_lower", "ema_9", "ema_21", 
                       "atr_14", "stoch_k", "williams_r", "roc_10", "vroc_10", "prox_52w", 
                       "lag1_ret", "lag5_ret", "dev_sma50", "hurst"]
                       
        # Replace any remaining inf values across all signal columns
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                       
        df.dropna(subset=feature_cols, inplace=True)

        signal_cols = ["date", "ticker", "close"] + feature_cols + ["target"]
        
        # Check existing records to avoid duplicates
        existing_dates = pd.read_sql(
            f"SELECT date FROM signals WHERE ticker = '{ticker}'", con=engine
        )["date"].tolist()
        
        # Filter new rows
        new_rows = df[~df["date"].isin(existing_dates)][signal_cols]
        
        if not new_rows.empty:
            new_rows.to_sql("signals", con=engine, if_exists="append", index=False)
            total_new_rows += len(new_rows)
            print(f"Stored {len(new_rows)} new signal rows for {ticker}.")
        else:
            print(f"No new signal rows for {ticker}.")

    print(f"Signals computation complete. Total new rows: {total_new_rows}")
    return total_new_rows
