# Fix Plan

Issues are ordered by severity. All changes are surgical — no refactors beyond what is needed to fix the issue.

---

## 🔴 Critical Bugs

---

### Fix 1 — `main.py:62` Wrong dashboard path

**Problem:** `subprocess.run` points to `app/dashboard.py`, which does not exist. The file is `app/main.py`. Local startup always fails at the final step.

**Fix:** Change the path string on line 62.

```python
# Before
dashboard_path = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")

# After
dashboard_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
```

**File:** `stock_forecast/main.py`, line 62  
**Risk:** None. One-line string change.

---

### Fix 2 — `forecasting_agent.py:26` Wrong model path

**Problem:** The forecasting agent looks for the model at `models/{ticker}.joblib` but `model.py` saves it to `models/joblib/{ticker}.joblib`. The file is never found, so `needs_training` is always `True` and the model retrains on every agent invocation.

**Fix:** Add the missing `joblib` subdirectory to the path.

```python
# Before
model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"{ticker}.joblib")

# After
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "joblib", f"{ticker}.joblib")
```

**File:** `stock_forecast/agents/forecasting_agent.py`, line 26  
**Risk:** None. The path now matches `model.py:145`. No other logic changes.

---

### Fix 3 — `signals.py:304` Prediction rows never stored

**Problem:** `df.dropna(subset=["target"], inplace=True)` on line 304 removes all rows where `target` is NaN — i.e., the last 30 trading days that have no 30-day-forward close yet. These are exactly the rows needed for forecasting. Because they are dropped before being written to the DB, `model.py`'s `pred_mask = y.isna()` is always `False`, and the model falls back to using features from ~30 days ago.

**Fix:** Remove the `dropna(subset=["target"])` call and let the feature-column dropna on line 316 remain (it already correctly removes incomplete signal rows). The misleading comment on line 305 should be deleted since it will no longer contradict anything.

The rows with `NaN` target will be stored in the DB. `model.py` already handles this correctly: it filters `train_mask = y.notna()` for training and `pred_mask = y.isna()` for prediction.

```python
# Remove these two lines (304-305):
df.dropna(subset=["target"], inplace=True)
# We don't drop rows with NaN target here because we want to predict for today!
# The model training step will filter out NaN targets.
```

The `signal_cols` list already includes `"target"`, so rows with `NaN` target will be written as-is to the DB.

**File:** `stock_forecast/pipeline/signals.py`, lines 304–305  
**Risk:** Low. The DB will now store ~30 extra rows per ticker with `target = NULL`. All existing queries that filter on `target` (model training) are already guarded with `notna()` / `dropna(subset=[TARGET])`. The only new behaviour is that `y.isna()` in `model.py` now correctly finds the prediction row.

---

## 🟠 Configuration Bugs

---

### Fix 4 — `tuning.py:37-38` Optuna almost disabled

**Problem:** `N_TRIALS = 2` and `N_FOLDS = 1` reduce Optuna to a near-trivial search. The README, docstring, and scheduler all describe 50 trials and 5-fold expanding window CV.

**Fix:** Restore the intended values.

```python
# Before
N_TRIALS   = 2
N_FOLDS    = 1

# After
N_TRIALS   = 50
N_FOLDS    = 5
```

**File:** `stock_forecast/pipeline/tuning.py`, lines 37–38  
**Risk:** This makes the weekly retune significantly slower (as originally designed). No logic changes. The daily fast retrain path (`load_params` early-return) is unaffected.

---

### Fix 5 — `tuning.py:159-169` Hardcoded `device: "cuda"` in saved params

**Problem:** `"device": "cuda"` is written into the Optuna params dict and saved to `tuned_params/*.json`. On CPU-only machines these params are reloaded and cause XGBoost warnings on every retrain.

**Fix:** Determine the device at runtime using `torch.cuda.is_available()` and set it once in `tune_hyperparameters` before saving. Also apply the same dynamic device in `expanding_window_cv` (which also hardcodes `"cuda"` in the objective).

```python
# Add near the top of tuning.py, after imports
import torch
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Then replace both `"device": "cuda"` occurrences (inside `objective` at line 161, and in `best_params` at line 169) with `"device": _DEVICE`.

**File:** `stock_forecast/pipeline/tuning.py`, lines 161 and 169  
**Risk:** None on CUDA machines. On CPU machines removes spurious warnings.

---

## 🟡 Logic Issues

---

### Fix 6 — Inconsistent confidence thresholds

**Problem:** `model.py` and `forecasting_agent.py` define different thresholds for High / Medium / Low confidence. The label stored in the DB (via `model.py`) and the one assigned when re-using an existing model (via `forecasting_agent.py`) can differ for the same stock.

**Fix:** Extract the thresholds into a single shared function in `pipeline/model.py` and call it from both places.

```python
# Add to pipeline/model.py (used as the source of truth)
def classify_confidence(mape: float, dir_acc: float) -> str:
    if mape < 8.0 and dir_acc > 65.0:
        return "High"
    if mape <= 12.0 or dir_acc >= 55.0:
        return "Medium"
    return "Low"
```

`model.py` already uses these thresholds so it needs no change. In `forecasting_agent.py:56-60`, replace the inline if/elif block with:

```python
from pipeline.model import classify_confidence
updates["forecast_confidence"] = classify_confidence(res["mape"], res["dir_acc"])
```

**Files:** `stock_forecast/pipeline/model.py` (add function), `stock_forecast/agents/forecasting_agent.py` (replace lines 56–60)  
**Risk:** None. Eliminates divergence without changing the authoritative thresholds.

---

### Fix 7 — `macro.py:92-96` Fallback writes stale data with today's date

**Problem:** When `yf.download` returns empty data, the fallback reads the last known macro row and re-inserts it labelled with today's date. This silently populates the table with stale macro values (USD/INR, VIX, Nifty returns) marked as current.

**Fix:** Remove the fallback re-insert entirely. If data is unavailable, log it and return 0. The `ffill` in the model's feature preparation already handles occasional missing macro rows for individual tickers.

```python
# Replace lines 90-98:
if data.empty:
    print("[Macro] No macro data returned from yfinance. Skipping update.")
    return 0
```

**File:** `stock_forecast/pipeline/macro.py`, lines 89–98  
**Risk:** Low. On days where yfinance genuinely fails, macro table won't get a new row. The model training already forward-fills macro data for missing dates, so this is handled gracefully.

---

## 🟡 Dead Code Removal

---

### Fix 8 — `api/dependencies.py:get_db()` never used

**Problem:** `get_db()` is a FastAPI dependency wrapper around `get_engine()` that no router actually uses.

**Fix:** Delete the `get_db` function. Keep `verify_api_key` which is used.

```python
# Remove these lines from dependencies.py:
def get_db():
    return get_engine()
```

**File:** `stock_forecast/api/dependencies.py`, lines 8–9  
**Risk:** None. Verify no router uses `Depends(get_db)` before deleting (confirmed: none do).

---

### Fix 9 — `app/components/critic_view.py` never imported

**Problem:** The file exists as a clean reusable component but `stock_detail.py` Tab 4 inlines the entire critic review as raw HTML instead of calling it.

**Decision:** Two options:
- **Option A (preferred):** Delete `critic_view.py` since `stock_detail.py`'s inline version is more feature-rich (timeline layout, structured flags, confidence badge).
- **Option B:** Refactor Tab 4 to use `render_critic_view()` from the component.

**Fix:** Delete `app/components/critic_view.py`.

**File:** `stock_forecast/app/components/critic_view.py`  
**Risk:** None. Confirmed unused.

---

### Fix 10 — `lstm_model.py:SCALER_DIR` unused variable

**Problem:** `SCALER_DIR` is declared but `get_scaler_path()` uses `MODELS_DIR`. `SCALER_DIR` is a dead constant.

**Fix:** Remove the `SCALER_DIR` declaration on lines 55–58.

```python
# Remove:
SCALER_DIR  = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "lstm"
)
```

**File:** `stock_forecast/pipeline/lstm_model.py`, lines 55–58  
**Risk:** None. Confirmed `SCALER_DIR` is never referenced.

---

## 🟡 Code Quality Fixes

---

### Fix 11 — `stock_detail.py:16` Duplicate `import streamlit as st`

**Problem:** `streamlit` is imported on both line 5 and line 16.

**Fix:** Delete the second import at line 16.

**File:** `stock_forecast/app/pages/stock_detail.py`, line 16  
**Risk:** None.

---

### Fix 12 — `stock_detail.py:47,58` Bare `except:` clauses

**Problem:** Two `fetch_*` functions use bare `except:` which catches `KeyboardInterrupt`, `SystemExit`, and other non-`Exception` base classes.

**Fix:** Replace both with `except Exception:`.

```python
# fetch_forecast (line 47)
except Exception:
    return None

# fetch_signals (line 58)
except Exception:
    return {"signals_df": [], "latest_signals": {}}
```

**File:** `stock_forecast/app/pages/stock_detail.py`, lines 47 and 58  
**Risk:** None.

---

### Fix 13 — `signals.py:231-244` `BollingerBands` instantiated 3 times

**Problem:** `BollingerBands(close=df["close"], window=20, window_dev=2)` is constructed three separate times for `bb_width`, `bb_upper`, and `bb_lower`. Each call recomputes the same rolling statistics.

**Fix:** Create one instance and read all three bands from it.

```python
bb = BollingerBands(close=df["close"], window=20, window_dev=2)
df["bb_width"]  = bb.bollinger_wband()
df["bb_upper"]  = bb.bollinger_hband()
df["bb_lower"]  = bb.bollinger_lband()
```

**File:** `stock_forecast/pipeline/signals.py`, lines 231–244  
**Risk:** None. Identical output, lower compute cost.

---

### Fix 14 — `model.py:273-274` Late imports inside `for` loop

**Problem:** `from sqlalchemy import text` and `from datetime import datetime` are imported inside the training loop, on every ticker iteration.

**Fix:** Move both to the top-level imports section of `model.py`.

**File:** `stock_forecast/pipeline/model.py`, lines 273–274 (move to top of file)  
**Risk:** None. Both are already available as installed packages.

---

### Fix 15 — `leaderboard.py:150-157` Hardcoded sector list

**Problem:** The filter selectbox in `render_filters()` contains a hardcoded list of sectors. If `TICKERS` in `data/tickers.py` changes, the filter silently becomes stale.

**Fix:** Import `get_all_sectors()` and prepend `"All"` to the dynamic list.

```python
from data.tickers import get_all_sectors
sectors = ["All"] + get_all_sectors()
```

**File:** `stock_forecast/app/pages/leaderboard.py`, lines 150–157  
**Risk:** None. `get_all_sectors()` returns a sorted list — the selectbox output is identical to today's hardcoded list.

---

### Fix 16 — `stock_detail.py:13` Redundant `sys.path.append`

**Problem:** The path manipulation on line 13 is not needed when the app is launched correctly via `streamlit run app/main.py` from `stock_forecast/`.

**Fix:** Delete line 13.

```python
# Remove:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```

**File:** `stock_forecast/app/pages/stock_detail.py`, line 13  
**Risk:** Low. If any edge case launch method relied on it, imports would break. The standard launch path (`streamlit run app/main.py`) does not need it.

---

## Execution Order

Apply in this sequence to avoid introducing new issues while fixing old ones:

1. Fix 3 (signals.py — prediction rows) — prerequisite for forecasts being meaningful
2. Fix 2 (forecasting_agent.py — model path) — prerequisite for model reuse to work
3. Fix 1 (main.py — dashboard path) — prerequisite for local testing
4. Fix 4 (tuning.py — N_TRIALS/N_FOLDS) — affects next scheduled retune
5. Fix 5 (tuning.py — CUDA device) — apply alongside Fix 4
6. Fix 6 (confidence thresholds) — requires Fix 2 to be meaningful
7. Fix 7 (macro.py — stale fallback) — data integrity
8. Fixes 8–10 (dead code removal) — safe at any point
9. Fixes 11–16 (code quality) — safe at any point
