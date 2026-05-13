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

def compute_sector_momentum(
    ticker: str,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes sector relative momentum for a stock over 5, 10, and 20-day windows.

    sector_rel_Xd = stock_pct_return_Xd - sector_index_pct_return_Xd

    Positive values indicate the stock is outperforming its sector index
    over that window (relative strength). Negative values indicate
    underperformance (relative weakness).

    Adds three columns: sector_rel_5d, sector_rel_10d, sector_rel_20d
    Falls back to 0.0 for all three if sector index data cannot be fetched
    or if the stock's sector is not in SECTOR_INDICES.
    """
    import yfinance as yf
    from data.tickers import TICKERS, SECTOR_INDICES

    # Identify this stock's sector and its corresponding index ticker
    sector       = TICKERS.get(ticker, {}).get("sector", "")
    index_ticker = SECTOR_INDICES.get(sector)

    fallback_cols = ["sector_rel_5d", "sector_rel_10d", "sector_rel_20d"]

    if not index_ticker:
        for col in fallback_cols:
            df[col] = 0.0
        return df

    try:
        sector_data = yf.download(
            index_ticker, period="2y", interval="1d",
            auto_adjust=True, progress=False
        )

        sector_data.reset_index(inplace=True)

        # Flatten MultiIndex columns if present
        if isinstance(sector_data.columns, pd.MultiIndex):
            sector_data.columns = [str(c[0]).lower() for c in sector_data.columns]
        else:
            sector_data.columns = [str(c).lower() for c in sector_data.columns]

        sector_data["date"] = sector_data["date"].astype(str).str[:10]
        sector_data = sector_data[["date", "close"]].rename(
            columns={"close": "sector_close"}
        )

        if sector_data.empty:
            raise ValueError(f"No sector data returned for {index_ticker}")

        # Merge sector close prices into the stock dataframe by date
        df = df.merge(sector_data, on="date", how="left")

        # Forward fill any missing sector dates (weekends, holidays)
        df["sector_close"] = df["sector_close"].ffill()

        # Compute relative momentum for each window
        for window in [5, 10, 20]:
            stock_ret  = df["close"].pct_change(window)
            sector_ret = df["sector_close"].pct_change(window)
            rel_mom    = stock_ret - sector_ret
            df[f"sector_rel_{window}d"] = (
                rel_mom
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

        df.drop(columns=["sector_close"], inplace=True)

    except Exception as e:
        print(f"[Signals] Sector momentum failed for {ticker}: {e}")
        for col in fallback_cols:
            df[col] = 0.0

    return df

def compute_earnings_surprise(
    ticker: str,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes earnings surprise as a persistent quarterly signal.

    earnings_surprise = (actual_EPS - estimated_EPS) / abs(estimated_EPS)

    Clipped to [-2.0, 2.0] to prevent extreme outliers from dominating
    the feature space. Forward-filled between quarterly earnings dates
    so the model has a non-zero signal between reports.

    A value of +0.10 means the company beat EPS estimates by 10%.
    A value of -0.15 means it missed by 15%.

    Adds column: earnings_surprise
    Falls back to 0.0 for all rows if earnings data is unavailable
    or if yfinance does not provide estimates for this ticker.
    """
    import yfinance as yf

    try:
        ticker_obj = yf.Ticker(ticker)
        earnings   = ticker_obj.earnings_dates

        if earnings is None or len(earnings) == 0:
            df["earnings_surprise"] = 0.0
            return df

        earnings = earnings.reset_index()
        earnings.columns = [
            c.lower().replace(" ", "_") for c in earnings.columns
        ]

        # Find estimate and actual EPS columns
        # yfinance column names vary slightly by version
        est_col = next(
            (c for c in earnings.columns if "estimate" in c.lower()), None
        )
        act_col = next(
            (c for c in earnings.columns if "actual" in c.lower() or "reported" in c.lower()), None
        )
        date_col = next(
            (c for c in earnings.columns if "date" in c.lower()), None
        )

        if not est_col or not act_col or not date_col:
            df["earnings_surprise"] = 0.0
            return df

        # Parse dates and compute surprise
        earnings["parsed_date"] = pd.to_datetime(
            earnings[date_col], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        earnings = earnings[
            ["parsed_date", est_col, act_col]
        ].dropna()

        # Only use rows where estimate is non-zero to avoid division issues
        earnings = earnings[earnings[est_col].abs() > 0.001]

        if earnings.empty:
            df["earnings_surprise"] = 0.0
            return df

        earnings["surprise"] = (
            (earnings[act_col] - earnings[est_col]) /
            earnings[est_col].abs()
        ).clip(-2.0, 2.0)

        surprise_map = dict(
            zip(earnings["parsed_date"], earnings["surprise"])
        )

        # Map to signal dataframe and forward fill between earnings dates
        df["earnings_surprise"] = (
            df["date"]
            .map(surprise_map)
            .ffill()
            .fillna(0.0)
        )

    except Exception as e:
        print(f"[Signals] Earnings surprise failed for {ticker}: {e}")
        df["earnings_surprise"] = 0.0

    return df

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

        # Sector relative momentum — Track A
        df = compute_sector_momentum(ticker, df)

        # Earnings surprise — Track A
        df = compute_earnings_surprise(ticker, df)

        # ── Target: closing price 30 days forward ───────────────────────────────


        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["target"] = df["close"].shift(-30)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["target"], inplace=True)
        # We don't drop rows with NaN target here because we want to predict for today!
        # The model training step will filter out NaN targets.
        feature_cols = ["rsi", "macd_hist", "bb_width", "obv", "sma_20", "ema_50", "bb_upper", "bb_lower", "ema_9", "ema_21", 
                       "atr_14", "stoch_k", "williams_r", "roc_10", "vroc_10", "prox_52w", 
                       "lag1_ret", "lag5_ret", "dev_sma50", "hurst", "sector_rel_5d", "sector_rel_10d", "sector_rel_20d", "earnings_surprise"]
                       
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
