"""
pipeline/tft_model.py

Temporal Fusion Transformer (TFT) trained jointly across all 53 NSE tickers.

Architecture:
  - One shared model learns cross-ticker patterns
  - Static categoricals:        ticker, sector
  - Time-varying unknowns:      all 29 features (technical + sentiment + macro)
  - Target:                     30-day log-return (same as XGBoost)
  - Max encoder length:         60 trading days
  - Max prediction length:      1  (direct 30-day-ahead regression)
  - Output:                     7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

Training:
  - Weekly full retrain (force=True via scheduler)
  - Daily inference only (model loaded from disk)
  - Lightning Trainer, 50 epochs, early stopping (patience=10)

The median (q=0.5) quantile is used as the point forecast.
Quantiles are returned for the confidence interval displayed on the dashboard.

Saved as models/tft/tft_model.ckpt
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "tft",
)
CKPT_PATH = os.path.join(MODELS_DIR, "tft_model.ckpt")

TFT_FEATURES = [
    "rsi", "macd_hist", "bb_width", "obv", "sma_20",
    "ema_9", "ema_21", "ema_50", "atr_14", "stoch_k",
    "williams_r", "roc_10", "vroc_10", "prox_52w",
    "lag1_ret", "lag5_ret", "dev_sma50", "bb_upper",
    "bb_lower", "hurst",
    "sentiment_score",
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
    "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
    "earnings_surprise",
]

MAX_ENCODER_LEN = 60
MAX_PRED_LEN    = 1      # direct 30-day-ahead regression

_tft_model   = None
_tft_dataset = None
_tft_available = None


def _try_imports():
    """Returns (TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, Trainer) or raises."""
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import lightning as L
    return TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, L


def build_tft_dataframe(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """
    Combines per-ticker DataFrames (feature matrix + target + close) into the
    long-format TimeSeriesDataSet input.

    Args:
        all_data: {ticker: df} where df has TFT_FEATURES + ['target', 'close', 'sector']
                  indexed by date string.

    Returns:
        Long-format DataFrame with columns:
            time_idx (int), ticker (str), sector (str), <features>, target (float)
        or None if insufficient data.
    """
    from data.tickers import get_sector

    rows = []
    for ticker, df in all_data.items():
        df = df.copy().reset_index()
        df = df.rename(columns={"index": "date"}) if "date" not in df.columns else df

        # Keep only rows where all features and target are valid
        required = TFT_FEATURES + ["target"]
        df = df.dropna(subset=required)
        if len(df) < MAX_ENCODER_LEN + 10:
            continue

        df["ticker"] = ticker
        df["sector"] = get_sector(ticker)
        df = df.sort_values("date").reset_index(drop=True)
        df["time_idx"] = range(len(df))

        # Replace inf values
        for col in TFT_FEATURES:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        rows.append(df)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def train_tft(all_data: dict[str, pd.DataFrame], force: bool = False) -> bool:
    """
    Trains the TFT model jointly on all tickers and saves the checkpoint.

    Args:
        all_data: {ticker: df} — see build_tft_dataframe for expected schema
        force:    Retrain even if checkpoint exists (used by weekly scheduler)

    Returns:
        True on success, False on failure.
    """
    global _tft_model, _tft_dataset, _tft_available

    if not force and os.path.exists(CKPT_PATH):
        logger.info("[TFT] Checkpoint exists — skipping retrain (use force=True for weekly retune)")
        return True

    try:
        TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, L = _try_imports()
    except ImportError as e:
        logger.warning(f"[TFT] pytorch-forecasting not available — {e}")
        _tft_available = False
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)

    combined = build_tft_dataframe(all_data)
    if combined is None or len(combined) < 500:
        logger.warning("[TFT] Not enough combined data to train TFT")
        return False

    # 80/20 time-ordered split within each ticker
    split_mask = combined.groupby("ticker")["time_idx"].transform(
        lambda x: x <= x.quantile(0.8)
    )
    train_df = combined[split_mask].copy()
    val_df   = combined[~split_mask].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        static_categoricals=["ticker", "sector"],
        time_varying_unknown_reals=TFT_FEATURES,
        max_encoder_length=MAX_ENCODER_LEN,
        max_prediction_length=MAX_PRED_LEN,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,   # 7 quantiles
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        log_interval=-1,
    )

    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=[early_stop],
        enable_model_summary=False,
        logger=False,
    )

    logger.info(f"[TFT] Starting training on {len(train_df)} rows across {combined['ticker'].nunique()} tickers...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(CKPT_PATH)
    _tft_model   = tft
    _tft_dataset = training
    _tft_available = True
    logger.info(f"[TFT] Training complete. Checkpoint saved to {CKPT_PATH}")
    return True


def _load_tft():
    """Lazily loads the saved TFT checkpoint."""
    global _tft_model, _tft_available

    if _tft_available is False:
        return None
    if _tft_model is not None:
        return _tft_model

    if not os.path.exists(CKPT_PATH):
        logger.info("[TFT] No checkpoint found — model not yet trained")
        return None

    try:
        from pytorch_forecasting import TemporalFusionTransformer
        _tft_model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
        _tft_model.eval()
        _tft_available = True
        logger.info("[TFT] Checkpoint loaded.")
        return _tft_model
    except Exception as e:
        _tft_available = False
        logger.warning(f"[TFT] Failed to load checkpoint — {e}")
        return None


def predict_tft(ticker: str, df: pd.DataFrame) -> float | None:
    """
    Generates a 30-day ahead log-return forecast for one ticker using the
    jointly-trained TFT model.

    Args:
        ticker: NSE ticker string
        df:     DataFrame with TFT_FEATURES + ['target', 'close'] columns,
                sorted ascending by date.  Must have at least MAX_ENCODER_LEN rows.

    Returns:
        Predicted 30-day log-return (median quantile), or None if unavailable.
    """
    model = _load_tft()
    if model is None:
        return None

    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from data.tickers import get_sector

        df = df.copy().reset_index()
        df = df.rename(columns={"index": "date"}) if "date" not in df.columns else df
        df["ticker"] = ticker
        df["sector"] = get_sector(ticker)
        df = df.sort_values("date").reset_index(drop=True)
        df["time_idx"] = range(len(df))

        for col in TFT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # target must exist but can be NaN for the prediction row
        if "target" not in df.columns:
            df["target"] = np.nan

        # Use the last MAX_ENCODER_LEN rows as the encoder context
        context = df.tail(MAX_ENCODER_LEN + MAX_PRED_LEN).copy().reset_index(drop=True)
        context["time_idx"] = range(len(context))

        encoder_data = context.head(MAX_ENCODER_LEN)
        encoder_data = encoder_data.dropna(subset=TFT_FEATURES)
        if len(encoder_data) < MAX_ENCODER_LEN // 2:
            return None

        pred_data = context.copy()
        pred_data.loc[pred_data.index[-1:], "target"] = np.nan

        dataset = TimeSeriesDataSet.from_dataset(
            _tft_dataset,
            pred_data,
            predict=True,
            stop_randomization=True,
        ) if _tft_dataset is not None else None

        if dataset is None:
            return None

        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        predictions = model.predict(loader, mode="quantiles")
        # predictions shape: (n_samples, pred_len, n_quantiles)
        # Median quantile is index 3 (q=0.5 in [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        median_pred = float(predictions[0, -1, 3])
        return median_pred

    except Exception as e:
        logger.error(f"[TFT] {ticker}: inference failed — {e}")
        return None


def get_tft_quantiles(ticker: str, df: pd.DataFrame) -> dict | None:
    """
    Returns all 7 quantile predictions for a ticker, enabling confidence intervals
    on the dashboard.

    Returns:
        {'q02': ..., 'q10': ..., 'q25': ..., 'q50': ..., 'q75': ..., 'q90': ..., 'q98': ...}
        or None if model unavailable.
    """
    model = _load_tft()
    if model is None:
        return None

    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from data.tickers import get_sector

        df = df.copy().reset_index()
        df = df.rename(columns={"index": "date"}) if "date" not in df.columns else df
        df["ticker"] = ticker
        df["sector"] = get_sector(ticker)
        df = df.sort_values("date").reset_index(drop=True)
        df["time_idx"] = range(len(df))

        for col in TFT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if "target" not in df.columns:
            df["target"] = np.nan

        context = df.tail(MAX_ENCODER_LEN + MAX_PRED_LEN).copy().reset_index(drop=True)
        context["time_idx"] = range(len(context))

        if _tft_dataset is None:
            return None

        dataset = TimeSeriesDataSet.from_dataset(
            _tft_dataset, context, predict=True, stop_randomization=True
        )
        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        preds = model.predict(loader, mode="quantiles")
        q = preds[0, -1, :]

        return {
            "q02": float(q[0]), "q10": float(q[1]), "q25": float(q[2]),
            "q50": float(q[3]), "q75": float(q[4]), "q90": float(q[5]),
            "q98": float(q[6]),
        }

    except Exception as e:
        logger.error(f"[TFT] {ticker}: quantile inference failed — {e}")
        return None
