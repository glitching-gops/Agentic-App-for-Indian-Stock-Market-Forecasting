"""
agents/critic_agent.py

LangGraph Critic node.  Reviews the Forecasting Agent's output via Groq and
returns a structured APPROVED / FLAGGED / REJECTED verdict.

On reflection retries (reflection_count > 0), the previous rejection reason
is injected as additional context so the LLM can evaluate whether the new
forecast addressed the flagged issues.

Hard overrides are limited to two objective, unambiguous conditions:
  - MAPE > 15%   → always REJECTED  (model is statistically unreliable)
  - |change| > 30% → always REJECTED  (price move is implausible in 30 days)
Everything else is left to the LLM.
"""

import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv
from agents.state import AgentState

load_dotenv(override=True)
_groq_api_key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
groq_client = (
    Groq(api_key=_groq_api_key)
    if _groq_api_key and _groq_api_key != "your_groq_key_here"
    else None
)


def critic_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    reflection_count = state.get("reflection_count", 0)
    critic_feedback  = state.get("critic_feedback", "")

    updates = {
        "critic_verdict":               "FLAGGED",
        "critic_reasoning":             "Critic Agent unavailable.",
        "critic_flags":                 [],
        "critic_confidence_adjustment": "MAINTAINED",
    }

    if not groq_client:
        return updates

    # Build reflection context shown to LLM on retries
    reflection_context = ""
    if reflection_count > 0 and critic_feedback:
        reflection_context = f"""
REFLECTION CONTEXT (attempt {reflection_count + 1} of 3):
The previous forecast was REJECTED for this reason:
\"{critic_feedback}\"
Evaluate whether the updated forecast has adequately addressed this concern.
"""

    try:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        prompt = f"""You are a senior quantitative analyst reviewing a 30-day stock forecast.
Only raise a flag if you observe a CLEAR and SIGNIFICANT issue.
{reflection_context}
Stock: {state['company_name']} ({ticker})
Current Price: ₹{state['current_price']}
Forecast Price: ₹{state.get('forecast_price', 0)} ({state.get('forecast_direction', 'UNKNOWN')}, {state.get('forecast_change_pct', 0):.2f}%)
Ensemble MAPE: {state.get('model_mape', 0):.2f}%
Directional Accuracy: {state.get('model_directional_accuracy', 0):.2f}%
Forecast Confidence: {state.get('forecast_confidence', 'Low')}
XGBoost forecast: ₹{state.get('xgb_forecast_price') or 'N/A'}
TFT forecast:     ₹{state.get('tft_forecast_price') or 'N/A'}
TimesFM forecast: ₹{state.get('tfm_forecast_price') or 'N/A'}

Signal Snapshot: {state['latest_signals']}
Sentiment Score: {state['sentiment_score']} (range: -1 to +1)
Signal Narrative: {state.get('signal_narrative', '')}

Raise a flag ONLY if one of these specific conditions is clearly present:
1. SIGNAL CONFLICT: RSI above 75 AND MACD histogram is strongly negative
2. SENTIMENT DIVERGENCE: sentiment below -0.3 AND forecast is UP, or above +0.3 AND forecast is DOWN
3. MODEL DISAGREEMENT: XGBoost, TFT, and TimesFM point in different directions
4. DATA QUALITY: OBV flat AND volume ROC near zero (very thin trading volume)

Do NOT flag mild signal ambiguity, MAPEs between 8-15%, or normal volatility.

Respond ONLY in this exact JSON format with no other text:
{{
  "verdict": "APPROVED" | "FLAGGED" | "REJECTED",
  "reasoning": "2-3 sentence overall assessment",
  "flags": ["flag 1"] or [],
  "confidence_adjustment": "UPGRADED" | "MAINTAINED" | "DOWNGRADED"
}}"""

        response_text = ""
        for attempt in range(3):
            try:
                completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=0.2,
                )
                response_text = completion.choices[0].message.content.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise e

        response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\s*```$", "", response_text)

        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError:
            parsed = {}

        verdict   = parsed.get("verdict", "FLAGGED")
        reasoning = parsed.get("reasoning", "")
        flags     = parsed.get("flags", [])

        # ── Hard overrides — objective conditions only ────────────────────────
        mape       = state.get("model_mape", 0.0)
        change_pct = abs(state.get("forecast_change_pct", 0.0))

        if mape > 15.0:
            verdict   = "REJECTED"
            reasoning += f" Auto-rejected: MAPE {mape:.1f}% exceeds the 15% reliability threshold."
        elif change_pct > 30.0:
            verdict   = "REJECTED"
            reasoning += (
                f" Auto-rejected: {change_pct:.1f}% forecast move exceeds the "
                "30% plausibility cap for a 30-day horizon."
            )

        # PSU oil sector note (flag only, no verdict override)
        PSU_OIL = {"ONGC.NS", "BPCL.NS"}
        if ticker in PSU_OIL:
            psu_flag = (
                "PSU oil stock: price is materially driven by crude oil prices "
                "and government fuel pricing policy — risk not captured in current signal library."
            )
            if not any("PSU" in f or "crude" in f.lower() for f in flags):
                flags.append(psu_flag)

        updates["critic_verdict"]               = verdict
        updates["critic_reasoning"]             = reasoning
        updates["critic_flags"]                 = flags
        updates["critic_confidence_adjustment"] = parsed.get("confidence_adjustment", "MAINTAINED")

    except Exception as e:
        print(f"[{ticker}] Critic Agent error: {e}")
        updates["critic_reasoning"] = f"Groq API Error: {e}"

    return updates
