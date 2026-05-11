# db.py — SQLite database setup using SQLAlchemy
# Creates tables: ohlcv, signals, sentiment, and macro

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SQLITE_PATH = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'stock_forecast.db')}"

# Use DATABASE_URL from environment if present, otherwise fall back to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", _SQLITE_PATH)

# SQLAlchemy requires postgresql:// not postgres:// for newer versions
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Use psycopg v3 driver (psycopg[binary] is installed; psycopg2-binary DLL is blocked by policy)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
        
    if DATABASE_URL.startswith("postgresql"):
        _ENGINE = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,
            connect_args={"prepare_threshold": None}
        )
    else:
        _ENGINE = create_engine(DATABASE_URL, echo=False)
        
    return _ENGINE

def init_db():
    engine = get_engine()
    with engine.connect() as conn:

        # Raw OHLCV table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                date        TEXT,
                ticker      TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                PRIMARY KEY (date, ticker)
            )
        """))

        # Computed signals table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS signals (
                date        TEXT,
                ticker      TEXT,
                close       REAL,
                rsi         REAL,
                macd_hist   REAL,
                bb_width    REAL,
                obv         REAL,
                sma_20      REAL,
                ema_9       REAL,
                ema_21      REAL,
                atr_14      REAL,
                stoch_k     REAL,
                williams_r  REAL,
                roc_10      REAL,
                vroc_10     REAL,
                prox_52w    REAL,
                lag1_ret    REAL,
                lag5_ret    REAL,
                dev_sma50   REAL,
                ema_50      REAL,
                bb_upper    REAL,
                bb_lower    REAL,
                hurst       REAL,
                target      REAL,
                PRIMARY KEY (date, ticker)
            )
        """))

        # Add new columns if they do not already exist (safe migration)
        new_columns = [
            "ema_50", "bb_upper", "bb_lower", "hurst",
            "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
            "earnings_surprise"  # new
        ]
        for col in new_columns:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists, skip

        # Sentiment table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment (
                date            TEXT,
                ticker          TEXT,
                headline        TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                PRIMARY KEY (date, ticker, headline)
            )
        """))

        # Macro table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS macro (
                date                TEXT PRIMARY KEY,
                usdinr              REAL,
                india_vix           REAL,
                nifty_5d_return     REAL,
                nifty_20d_return    REAL,
                fii_net_flow        REAL,
                dii_net_flow        REAL
            )
        """))

        # Add new macro columns (safe migration)
        macro_new_columns = ["fii_net_flow", "dii_net_flow"]
        for col in macro_new_columns:
            try:
                with conn.begin_nested():
                    conn.execute(text(f"ALTER TABLE macro ADD COLUMN {col} REAL"))
            except Exception:
                pass  # Column already exists, skip

        # Add indexes for faster querying by ticker and date

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv (ticker, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals (ticker, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment (ticker, date)"))

        # Step 6: Create the forecasts and leaderboard tables
        # Use SERIAL PRIMARY KEY for PostgreSQL, INTEGER PRIMARY KEY AUTOINCREMENT for SQLite
        id_col_type = "INTEGER PRIMARY KEY AUTOINCREMENT" if _SQLITE_PATH in DATABASE_URL or DATABASE_URL.startswith("sqlite") else "SERIAL PRIMARY KEY"
        
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS forecasts (
                id                       {id_col_type},
                ticker                   TEXT NOT NULL,
                company                  TEXT,
                sector                   TEXT,
                current_price            REAL,
                forecast_price           REAL,
                direction                TEXT,
                change_pct               REAL,
                mape                     REAL,
                directional_accuracy     REAL,
                forecast_confidence      TEXT,
                signal_narrative         TEXT,
                critic_verdict           TEXT,
                critic_reasoning         TEXT,
                critic_flags             TEXT,
                critic_confidence_adjustment TEXT,
                last_updated             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                ticker                   TEXT PRIMARY KEY,
                company                  TEXT,
                sector                   TEXT,
                current_price            REAL,
                forecast_price           REAL,
                upside_pct               REAL,
                composite_score          REAL,
                critic_verdict           TEXT,
                forecast_confidence      TEXT,
                mape                     REAL,
                directional_accuracy     REAL,
                last_updated             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                ticker              TEXT PRIMARY KEY,
                xgb_mape            REAL,
                xgb_dir_acc         REAL,
                lstm_val_mape       REAL,
                lstm_epochs_trained INTEGER,
                meta_xgb_coef       REAL,
                meta_lstm_coef      REAL,
                meta_hurst_coef     REAL,
                ensemble_in_use     INTEGER DEFAULT 1,
                last_trained        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.commit()
    print("Database initialised.")
