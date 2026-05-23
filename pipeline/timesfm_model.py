"""
pipeline/timesfm_model.py

Zero-shot 30-day log-return forecast using Google TimesFM
(google/timesfm-1.0-200m-pytorch).

No training required — the foundation model is pre-trained on 100B
real-world time-points.  We feed the raw log-return series and take
the 30th step of the multi-step output as the 30-day forecast.

The model is loaded once and cached in _tfm_instance to avoid
reloading on every call.

Usage:
    log_return = predict_timesfm(ticker, close_prices_array)
    if log_return is not None:
        forecast_price = current_price * np.exp(log_return)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

_tfm_instance = None
_tfm_available = None   # None = untested, True/False after first attempt


def _load_timesfm():
    """Lazily loads the TimesFM model and caches it."""
    global _tfm_instance, _tfm_available

    if _tfm_available is False:
        return None
    if _tfm_instance is not None:
        return _tfm_instance

    # Honour kill-switch for memory-constrained environments (e.g. Render free tier)
    import os
    if os.getenv("DISABLE_TIMESFM", "0") not in ("0", "", None):
        _tfm_available = False
        logger.info("[TimesFM] Disabled via DISABLE_TIMESFM env var.")
        return None

    try:
        import timesfm

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="torch",
                per_core_batch_size=32,
                horizon_len=30,        # predict 30 steps ahead
                num_layers=20,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        _tfm_instance = tfm
        _tfm_available = True
        logger.info("[TimesFM] Model loaded from HuggingFace.")
        return tfm

    except Exception as e:
        _tfm_available = False
        logger.warning(f"[TimesFM] Could not load model — {e}. Falling back gracefully.")
        return None


def predict_timesfm(ticker: str, close_prices: np.ndarray) -> float | None:
    """
    Generates a 30-day ahead log-return forecast using TimesFM zero-shot inference.

    Args:
        ticker:       NSE ticker string (used only for logging)
        close_prices: 1D numpy array of daily closing prices, ascending order.
                      Must have at least 64 values (TimesFM minimum context).

    Returns:
        Predicted 30-day log-return (float), or None if model unavailable.
        Back-transform to price: forecast_price = current_price * exp(log_return)
    """
    tfm = _load_timesfm()
    if tfm is None:
        return None

    close_prices = np.asarray(close_prices, dtype=np.float32)
    close_prices = close_prices[np.isfinite(close_prices)]

    if len(close_prices) < 64:
        logger.warning(f"[TimesFM] {ticker}: only {len(close_prices)} prices, need 64+")
        return None

    try:
        # Convert to log-returns for the input series
        log_returns = np.log(close_prices[1:] / close_prices[:-1])
        log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # TimesFM expects a list of 1D arrays; freq=0 means daily
        point_forecast, _ = tfm.forecast(
            inputs=[log_returns],
            freq=[0],
        )

        # point_forecast shape: (1, horizon_len=30)
        # Sum of 30 daily log-return forecasts = 30-day cumulative log-return
        cumulative_log_return = float(np.sum(point_forecast[0]))
        logger.debug(f"[TimesFM] {ticker}: 30d log-return = {cumulative_log_return:.4f}")
        return cumulative_log_return

    except Exception as e:
        logger.error(f"[TimesFM] {ticker}: inference failed — {e}")
        return None
