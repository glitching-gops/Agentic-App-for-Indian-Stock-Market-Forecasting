"""
forecasting_agent.py
LangGraph node that generates a 30-day price forecast and a natural language signal narrative.
"""
import os
import time
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from groq import Groq
from agents.state import AgentState
from pipeline.model import train_and_forecast, FEATURES, classify_confidence

def forecasting_node(state: AgentState) -> dict:
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
    if groq_api_key and groq_api_key != "your_groq_key_here":
        groq_client = Groq(api_key=groq_api_key)
    else:
        groq_client = None

    ticker = state["ticker"]
    updates = {}
    
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "joblib", f"{ticker}.joblib")
    
    # Check if model exists and was created today
    needs_training = True
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        model_date = datetime.fromtimestamp(mtime).date()
        if model_date == datetime.today().date():
            needs_training = False
            
    if needs_training:
        print(f"[{ticker}] Retraining XGBoost model...")
        results_dict = train_and_forecast(ticker)
        if ticker in results_dict:
            res = results_dict[ticker]
        else:
            res = None
    else:
        print(f"[{ticker}] Loading existing model from today...")
        res = _generate_forecast_from_existing(state, model_path)
        
    if res:
        updates["forecast_price"] = res["forecast_price"]
        updates["forecast_direction"] = res["direction"]
        updates["forecast_change_pct"] = res["change_pct"]
        updates["model_mape"] = res["mape"]
        updates["model_directional_accuracy"] = res["dir_acc"]
        updates["feature_importances"] = res["feature_importance"]
        updates["forecast_confidence"] = classify_confidence(res["mape"], res["dir_acc"])
        updates["xgb_forecast_price"] = res.get("xgb_forecast_price")
        updates["tft_forecast_price"] = res.get("tft_forecast_price")
        updates["tfm_forecast_price"] = res.get("tfm_forecast_price")
    else:
        # Fallback values
        updates["forecast_price"] = state["current_price"]
        updates["forecast_direction"] = "UNKNOWN"
        updates["forecast_change_pct"] = 0.0
        updates["forecast_confidence"] = "Low"
        updates["model_mape"] = 100.0
        updates["model_directional_accuracy"] = 0.0
        updates["feature_importances"] = {}

    # LLM Signal Narrative Generation
    # Use 8b model by default to stay within free tier rate limits
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    updates["signal_narrative"] = "Signal narrative unavailable."
    
    if groq_client:
        try:
            sig_dict = state.get('latest_signals', {})
            importances = updates.get("feature_importances", {})
            if importances and len(sig_dict) > 10:
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                top_keys = [k for k, v in top_features]
                sig_dict = {k: v for k, v in sig_dict.items() if k in top_keys or k == 'close'}

            prompt = f"""You are a quantitative analyst summarising technical and sentiment signals for an Indian stock.

Stock: {state['company_name']} ({ticker})
Current Price: ₹{state['current_price']}
Signals (latest values):
{sig_dict}

Write exactly 3 sentences summarising what these signals collectively suggest about the stock's near-term momentum. Be specific about the signals. Do not make a price prediction."""

            # Simple retry mechanism for 429 Rate Limits
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    completion = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                        temperature=0.3
                    )
                    updates["signal_narrative"] = completion.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(2) # Short wait before retry
                        continue
                    raise e
        except Exception as e:
            print(f"[{ticker}] Error generating Groq narrative: {e}")
            # Fallback to rule-based narrative if Groq fails
            rsi = state.get("latest_signals", {}).get("rsi", 50)
            macd = state.get("latest_signals", {}).get("macd_hist", 0)
            sentiment = state.get("sentiment_score", 0)
            
            narrative = f"Technical indicators show {ticker} is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'in a neutral zone'} with RSI at {rsi:.1f}. "
            narrative += f"MACD histogram is {'positive' if macd > 0 else 'negative'}, suggesting {'bullish' if macd > 0 else 'bearish'} momentum. "
            narrative += f"Overall sentiment score is {sentiment:.2f}, indicating {'positive' if sentiment > 0.2 else 'negative' if sentiment < -0.2 else 'neutral'} market interest."
            updates["signal_narrative"] = narrative

    return updates

def _generate_forecast_from_existing(state: AgentState, model_path: str) -> dict:
    """Helper to generate forecast from loaded model using the state data."""
    try:
        model = joblib.load(model_path)
        ticker = state["ticker"]

        signals_df = pd.DataFrame(state.get("signals_df", []))
        if not signals_df.empty and "date" in signals_df.columns:
            signals_df.set_index("date", inplace=True)

        macro_df = pd.DataFrame(state.get("macro_df", []))
        if not macro_df.empty and "date" in macro_df.columns:
            macro_df.set_index("date", inplace=True)

        df = signals_df.join(macro_df, how="left")
        df["sentiment_score"] = state["sentiment_score"]
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        TARGET = "target"
        df_clean = df.dropna(subset=FEATURES).copy()
        train_df = df_clean.dropna(subset=[TARGET]).copy()

        if len(train_df) > 50:
            val_idx = int(len(train_df) * 0.85)
            X_test = train_df[FEATURES].iloc[val_idx:]
            y_test = train_df[TARGET].iloc[val_idx:]   # log-returns
            y_pred = model.predict(X_test)             # log-return predictions

            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            # Directional accuracy: positive log-return = UP
            actual_dir = (y_test.values > 0).astype(int)
            pred_dir   = (y_pred > 0).astype(int)
            dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)
        else:
            mape = 100.0
            dir_acc = 0.0

        current_price = state["current_price"]

        # Inference row: last row where target is NaN (no 30-day future yet)
        latest_features = df_clean[df_clean[TARGET].isna()][FEATURES]
        if latest_features.empty:
            latest_features = df_clean[FEATURES].iloc[[-1]]

        # Back-transform log-return → price
        xgb_log_return = float(model.predict(latest_features.iloc[[-1]])[0])
        xgb_price      = current_price * np.exp(xgb_log_return)

        # TFT inference
        tft_price = None
        try:
            from pipeline.tft_model import predict_tft
            df_full = df_clean.copy()
            if "close" not in df_full.columns and "close" in df.columns:
                df_full["close"] = df["close"]
            tft_log_return = predict_tft(ticker, df_full.copy())
            if tft_log_return is not None:
                tft_price = current_price * np.exp(tft_log_return)
        except Exception as e:
            print(f"[{ticker}] TFT inference skipped: {e}")

        # TimesFM inference
        tfm_price = None
        try:
            from pipeline.timesfm_model import predict_timesfm
            close_col = df["close"] if "close" in df.columns else pd.Series(dtype=float)
            if len(close_col) >= 64:
                tfm_log_return = predict_timesfm(ticker, close_col.values)
                if tfm_log_return is not None:
                    tfm_price = current_price * np.exp(tfm_log_return)
        except Exception as e:
            print(f"[{ticker}] TimesFM inference skipped: {e}")

        # Ensemble via meta-learner
        try:
            from pipeline.meta_learner import predict_ensemble
            current_hurst = float(df_clean["hurst"].iloc[-1]) if "hurst" in df_clean.columns else 0.5
            ensemble_price = predict_ensemble(
                ticker=ticker,
                xgb_price=xgb_price,
                tft_price=tft_price,
                tfm_price=tfm_price,
                current_hurst=current_hurst,
                current_price=current_price,
            )
        except Exception:
            ensemble_price = xgb_price

        return {
            "forecast_price":     ensemble_price,
            "xgb_forecast_price": xgb_price,
            "tft_forecast_price": tft_price,
            "tfm_forecast_price": tfm_price,
            "direction":  "UP" if ensemble_price > current_price else "DOWN",
            "change_pct": round(((ensemble_price - current_price) / current_price) * 100, 2),
            "mape":    round(mape, 2),
            "dir_acc": round(dir_acc, 2),
            "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
        }
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
