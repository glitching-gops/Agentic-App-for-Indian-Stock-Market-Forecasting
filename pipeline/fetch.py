# fetch.py — Downloads 2 years of daily OHLCV data
# Stores results in the ohlcv table in SQLite

import yfinance as yf
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from data.db import get_engine
from data.tickers import TICKERS

PERIOD = "2y"
INTERVAL = "1d"
MIN_ROWS = 20

# Schema applied to every ticker's OHLCV DataFrame before it is written to the DB.
# Validation failures are non-fatal: a warning is printed and the data is still
# stored so a single bad ticker doesn't abort the full 53-ticker run.
# The one hard gate is MIN_ROWS — fewer than 20 rows cannot produce any signals.
_OHLCV_SCHEMA = DataFrameSchema(
    columns={
        "date":   Column(str,   nullable=False),
        "ticker": Column(str,   nullable=False),
        "open":   Column(float, Check.gt(0), nullable=False),
        "high":   Column(float, Check.gt(0), nullable=False),
        "low":    Column(float, Check.gt(0), nullable=False),
        "close":  Column(float, Check.gt(0), nullable=False),
        "volume": Column(float, Check.ge(0), nullable=True),
    },
    checks=[
        Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high < low in one or more rows",
        ),
    ],
    coerce=True,
)

def fetch_and_store(single_ticker=None):
    engine = get_engine()
    
    tickers_to_fetch = [single_ticker] if single_ticker else list(TICKERS.keys())
    
    total_new_rows = 0
    for ticker in tickers_to_fetch:
        print(f"Fetching data for {ticker}...")
        try:
            # Download from yfinance
            fetch_period = "5y" if ticker == "TATAMOTORS.NS" else PERIOD
            df = yf.download(ticker, period=fetch_period, interval=INTERVAL, auto_adjust=True)
            if df.empty:
                print(f"No data found for {ticker}")
                # Try alternative ticker if TATAMOTORS.NS fails
                if ticker == "TATAMOTORS.NS":
                    print("Trying alternative ticker TATAMTRDVR.NS...")
                    df = yf.download("TATAMTRDVR.NS", period=fetch_period, interval=INTERVAL, auto_adjust=True)
                if df.empty:
                    continue
            
            df.reset_index(inplace=True)
            
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[0]).lower() for col in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]

            df.rename(columns={"date": "date"}, inplace=True)
            df["date"] = df["date"].astype(str).str[:10]  # Just the date part
            df["ticker"] = ticker

            # Keep only relevant columns
            if "volume" not in df.columns:
                df["volume"] = 0
                
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            df.dropna(inplace=True)

            # Hard gate: too few rows to compute any signals
            if len(df) < MIN_ROWS:
                print(f"[Fetch] {ticker}: only {len(df)} rows — skipping (need {MIN_ROWS}+)")
                continue

            # Pandera data quality check — warns on schema violations, doesn't abort
            try:
                _OHLCV_SCHEMA.validate(df, lazy=True)
            except pa.errors.SchemaErrors as exc:
                samples = exc.failure_cases[["check", "failure_case"]].head(3).to_dict("records")
                print(f"[Fetch] {ticker}: data quality warning — {samples} (storing anyway)")

            # Check existing records to avoid duplicates
            existing_dates = pd.read_sql(
                f"SELECT date FROM ohlcv WHERE ticker = '{ticker}'", con=engine
            )["date"].tolist()
            
            # Filter new rows
            new_rows = df[~df["date"].isin(existing_dates)]
            
            if not new_rows.empty:
                new_rows.to_sql("ohlcv", con=engine, if_exists="append", index=False)
                total_new_rows += len(new_rows)
                print(f"Stored {len(new_rows)} new rows for {ticker}.")
            else:
                print(f"No new rows to store for {ticker}.")
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            
    print(f"Fetch complete. Total new rows added: {total_new_rows}")
    return total_new_rows
