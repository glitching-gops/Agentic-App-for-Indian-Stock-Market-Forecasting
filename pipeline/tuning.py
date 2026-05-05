"""
pipeline/tuning.py

Optuna-based hyperparameter tuning for the per-stock XGBoost regressor.
Runs 50 trials per stock using expanding window cross-validation.
Saves best parameters to a JSON file per ticker so they can be reused
during daily fast retrains without re-running Optuna.

Tuned parameters:
  - n_estimators:    [100, 800]
  - learning_rate:   [0.01, 0.3]  (log scale)
  - max_depth:       [3, 8]
  - subsample:       [0.6, 1.0]
  - colsample_bytree:[0.6, 1.0]
  - min_child_weight:[1, 10]
  - gamma:           [0, 5]
  - reg_alpha:       [0, 2]       (L1 regularisation)
  - reg_lambda:      [0, 2]       (L2 regularisation)
"""

import os
import json
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from data.db import get_engine

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "tuned_params"
)
os.makedirs(PARAMS_DIR, exist_ok=True)

N_TRIALS   = 2
N_FOLDS    = 5


def get_params_path(ticker: str) -> str:
    """Returns the path to the saved tuned parameters JSON for a ticker."""
    safe = ticker.replace(".", "_")
    return os.path.join(PARAMS_DIR, f"{safe}_params.json")


def save_params(ticker: str, params: dict) -> None:
    """Saves the best Optuna parameters for a ticker to disk."""
    with open(get_params_path(ticker), "w") as f:
        json.dump(params, f, indent=2)


def load_params(ticker: str) -> dict | None:
    """
    Loads saved tuned parameters for a ticker if they exist.
    Returns None if no saved params are found.
    """
    path = get_params_path(ticker)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def expanding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_folds: int = N_FOLDS
) -> float:
    """
    Expanding window cross-validation for time series data.
    Trains on the first k/n_folds of data, tests on the next fold.
    Returns the mean MAPE across all folds.

    Example with 5 folds on 300 rows:
      Fold 1: train rows 0-60,   test rows 60-120
      Fold 2: train rows 0-120,  test rows 120-180
      Fold 3: train rows 0-180,  test rows 180-240
      Fold 4: train rows 0-240,  test rows 240-300
    """
    n = len(X)
    fold_size = n // (n_folds + 1)

    if fold_size < 20:
        # Not enough data for this many folds — fall back to single split
        split = int(n * 0.7)
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X.iloc[:split], y.iloc[:split])
        preds = model.predict(X.iloc[split:])
        return mean_absolute_percentage_error(y.iloc[split:], preds)

    mapes = []
    for fold in range(1, n_folds + 1):
        train_end = fold * fold_size
        test_end  = min(train_end + fold_size, n)

        if test_end <= train_end:
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[train_end:test_end]
        y_test  = y.iloc[train_end:test_end]

        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mapes.append(mean_absolute_percentage_error(y_test, preds))

    return float(np.mean(mapes)) if mapes else 1.0


def tune_hyperparameters(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_TRIALS,
    force: bool = False
) -> dict:
    """
    Runs Optuna hyperparameter tuning for a single stock's XGBoost model.

    If saved params exist and force=False, returns saved params immediately
    without running Optuna. This allows daily fast retrains to reuse
    Tuesday's tuned params without re-running 50 trials.

    Set force=True to re-tune from scratch (e.g. weekly Sunday retuning).

    Args:
        ticker:   NSE ticker string e.g. 'RELIANCE.NS'
        X:        Feature matrix (signals + sentiment + macro)
        y:        Target series (30-day forward close price)
        n_trials: Number of Optuna trials (default 50)
        force:    If True, ignores saved params and re-tunes

    Returns:
        dict of best XGBoost hyperparameters
    """
    if not force:
        saved = load_params(ticker)
        if saved:
            print(f"[Tuning] {ticker}: Using saved params (run with force=True to retune)")
            return saved

    print(f"[Tuning] {ticker}: Running {n_trials} Optuna trials...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
            "tree_method":      "hist",
            "device":           "cuda",
        }
        return expanding_window_cv(X, y, params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["tree_method"] = "hist"
    best_params["device"]      = "cuda"

    save_params(ticker, best_params)
    print(f"[Tuning] {ticker}: Best MAPE {study.best_value:.4f} | Params saved")

    return best_params
