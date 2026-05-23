"""
pipeline/meta_learner.py

Ridge regression meta-learner combining XGBoost, TFT, and TimesFM predictions.

Inputs per sample (4 features):
  1. XGBoost predicted price   (back-transformed from log-return)
  2. TFT predicted price       (back-transformed from log-return; fallback = XGBoost)
  3. TimesFM predicted price   (back-transformed from log-return; fallback = XGBoost)
  4. Hurst exponent            (regime signal — weights trending vs mean-reverting models)

When TFT or TimesFM predictions are unavailable (model not yet trained, or
inference failed), those inputs are filled with the XGBoost prediction so the
meta-learner degrades gracefully without changing its input dimensionality.

A 25% price change cap is applied to the final output relative to the
current price to prevent implausible forecasts from reaching the dashboard.

Saved as models/meta/{ticker}_meta.joblib.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge

MODELS_DIR     = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "meta",
)
MAX_CHANGE_PCT = 0.25


def get_meta_path(ticker: str) -> str:
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_meta.joblib")


def train_meta_learner(
    ticker: str,
    xgb_val_preds: np.ndarray,
    tft_val_preds: np.ndarray,
    tfm_val_preds: np.ndarray,
    hurst_val: np.ndarray,
    y_val: np.ndarray,
    force: bool = False,
) -> Ridge | None:
    """
    Trains a Ridge regression meta-learner on validation predictions from
    XGBoost, TFT, and TimesFM.

    Args:
        ticker:        NSE ticker string
        xgb_val_preds: XGBoost predicted prices on validation set (n_samples,)
        tft_val_preds: TFT predicted prices on validation set; use xgb_val_preds
                       as fallback when TFT not yet available (n_samples,)
        tfm_val_preds: TimesFM predicted prices; same fallback convention (n_samples,)
        hurst_val:     Hurst exponent values for validation rows (n_samples,)
        y_val:         True target prices for validation set (n_samples,)
        force:         If True, retrain even if saved model exists

    Returns:
        Trained Ridge model, or None if insufficient validation data.
    """
    meta_path = get_meta_path(ticker)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not force and os.path.exists(meta_path):
        return joblib.load(meta_path)

    min_len = min(
        len(xgb_val_preds),
        len(tft_val_preds),
        len(tfm_val_preds),
        len(hurst_val),
        len(y_val),
    )

    if min_len < 10:
        print(
            f"[Meta] {ticker}: only {min_len} validation samples — "
            "using equal weight fallback"
        )
        return None

    X_meta = np.column_stack([
        xgb_val_preds[-min_len:],
        tft_val_preds[-min_len:],
        tfm_val_preds[-min_len:],
        hurst_val[-min_len:],
    ])
    y_meta = y_val[-min_len:]

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_meta)
    joblib.dump(meta, meta_path)

    print(
        f"[Meta] {ticker}: trained | "
        f"XGB={meta.coef_[0]:.3f} | "
        f"TFT={meta.coef_[1]:.3f} | "
        f"TFM={meta.coef_[2]:.3f} | "
        f"Hurst={meta.coef_[3]:.3f}"
    )
    return meta


def predict_ensemble(
    ticker: str,
    xgb_price: float,
    tft_price: float | None,
    tfm_price: float | None,
    current_hurst: float,
    current_price: float,
) -> float:
    """
    Generates the final ensemble price prediction.

    Unavailable model predictions (None) are replaced with xgb_price so the
    meta-learner always receives a 4-column input.  If no meta-learner is saved,
    falls back to equal weighting of available predictions.

    Args:
        ticker:        NSE ticker string
        xgb_price:     XGBoost back-transformed price
        tft_price:     TFT back-transformed price, or None
        tfm_price:     TimesFM back-transformed price, or None
        current_hurst: Most recent Hurst exponent value
        current_price: Current closing price (used for 25% cap)

    Returns:
        Final ensemble price prediction (float), capped at ±25% of current_price.
    """
    # Graceful fallback for unavailable models
    tft_price = tft_price if tft_price is not None else xgb_price
    tfm_price = tfm_price if tfm_price is not None else xgb_price

    meta_path = get_meta_path(ticker)
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        X    = np.array([[xgb_price, tft_price, tfm_price, current_hurst]])
        pred = float(meta.predict(X)[0])
    else:
        # Simple average until meta-learner is trained
        pred = (xgb_price + tft_price + tfm_price) / 3.0

    return _apply_cap(pred, current_price)


def _apply_cap(price: float, current_price: float) -> float:
    """Clips the forecast price to within MAX_CHANGE_PCT of the current price."""
    if current_price <= 0:
        return price
    upper = current_price * (1 + MAX_CHANGE_PCT)
    lower = current_price * (1 - MAX_CHANGE_PCT)
    return float(np.clip(price, lower, upper))
