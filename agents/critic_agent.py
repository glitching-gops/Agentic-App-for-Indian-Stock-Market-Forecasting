"""
critic_agent.py
LangGraph node that critically reviews the Forecasting Agent's output using Groq.
"""
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
from agents.state import AgentState

load_dotenv(override=True)
_groq_api_key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
groq_client = Groq(api_key=_groq_api_key) if _groq_api_key and _groq_api_key != "your_groq_key_here" else None

def critic_node(state: AgentState) -> dict:
        
    ticker = state["ticker"]
    
    # Default fallback state
    updates = {
        "critic_verdict": "FLAGGED",
        "critic_reasoning": "Critic Agent failed to parse LLM response or API key missing.",
        "critic_flags": [],
        "critic_confidence_adjustment": "MAINTAINED"
    }
    
    if not groq_client:
        return updates
        
    try:
        prompt = f"""You are a senior quantitative analyst reviewing a 30-day stock forecast.
Only raise a flag if you observe a CLEAR and SIGNIFICANT issue. Do not
raise flags for minor signal ambiguity or normal market noise.

Stock: {state['company_name']} ({ticker})
Current Price: ₹{state['current_price']}
Forecast Price: ₹{state.get('forecast_price', 0)} ({state.get('forecast_direction', 'UNKNOWN')}, {state.get('forecast_change_pct', 0)}%)
Model MAPE: {state.get('model_mape', 0)}%
Directional Accuracy: {state.get('model_directional_accuracy', 0)}%
Forecast Confidence: {state.get('forecast_confidence', 'Low')}

Signal Snapshot:
{state['latest_signals']}

Sentiment Score: {state['sentiment_score']} (range: -1 to +1)
Signal Narrative: {state.get('signal_narrative', '')}

Raise a flag ONLY if one of these specific conditions is clearly present:
1. SIGNAL CONFLICT: RSI above 75 AND MACD histogram is strongly negative
   (both must be true simultaneously, not just one)
2. SENTIMENT DIVERGENCE: sentiment score below -0.3 AND forecast direction
   is UP, or sentiment above +0.3 AND forecast direction is DOWN
3. EXTREME FORECAST: predicted price change exceeds 28% in 30 days
4. DATA QUALITY: OBV has been flat (near zero change) for the entire
   signal window AND volume ROC is also near zero — indicating very thin
   trading volume

Do NOT flag:
- MAPE between 8-15% (this is acceptable for a 30-day horizon)
- Mild sentiment disagreement (scores between -0.3 and +0.3 are neutral)
- Normal volatility in any single indicator
- Stocks that are simply in a downtrend

Respond ONLY in this exact JSON format with no other text:
{{
  "verdict": "APPROVED" | "FLAGGED" | "REJECTED",
  "reasoning": "2-3 sentence overall assessment",
  "flags": ["flag 1"] or [],
  "confidence_adjustment": "UPGRADED" | "MAINTAINED" | "DOWNGRADED"
}}"""

        # Use 8b model by default to stay within free tier rate limits
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        # Simple retry mechanism for 429 Rate Limits
        response_text = ""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=0.2
                )
                response_text = completion.choices[0].message.content.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                raise e

        
        # Clean markdown formatting if present robustly
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\s*```$", "", response_text)
            
        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"[{ticker}] Critic Agent JSON parse error: {e}")
            parsed = {}
            
        verdict   = parsed.get("verdict", "FLAGGED")
        reasoning = parsed.get("reasoning", "")
        flags     = parsed.get("flags", [])

        # ── Hard verdict override rules ───────────────────────────────────────
        mape       = state.get("model_mape", 0.0)
        change_pct = abs(state.get("forecast_change_pct", 0.0))
        dir_acc    = state.get("model_directional_accuracy", 0.0)
        num_flags  = len(flags)

        # REJECTED — only for genuinely unreliable forecasts
        if mape > 15.0:
            verdict    = "REJECTED"
            reasoning += f" Auto-rejected: MAPE of {mape:.1f}% exceeds the 15% threshold."
        elif change_pct > 30.0:
            verdict    = "REJECTED"
            reasoning += (
                f" Auto-rejected: forecast change of {change_pct:.1f}% "
                f"exceeds the 30% plausibility cap."
            )
        elif num_flags >= 3:
            verdict    = "REJECTED"
            reasoning += f" Auto-rejected: {num_flags} simultaneous flags raised."

        # APPROVED — model quality overrides LLM flags progressively
        # Tier 1: exceptional model — overrides any number of flags
        elif mape < 6.0 and dir_acc > 75.0:
            verdict = "APPROVED"
        # Tier 2: strong model — overrides up to 2 flags
        elif mape < 8.0 and dir_acc > 70.0 and num_flags <= 2:
            verdict = "APPROVED"
        # Tier 3: good model — overrides 1 flag
        elif mape < 10.0 and dir_acc > 65.0 and num_flags <= 1:
            verdict = "APPROVED"
        # Tier 4: no flags at all — even average models get approved
        elif num_flags == 0 and mape < 12.0:
            verdict = "APPROVED"

        updates["critic_verdict"]               = verdict
        updates["critic_reasoning"]             = reasoning
        updates["critic_flags"]                 = flags
        updates["critic_confidence_adjustment"] = parsed.get("confidence_adjustment", "MAINTAINED")

        # PSU oil override — always at least FLAGGED
        PSU_OIL_TICKERS = ["ONGC.NS", "BPCL.NS"]
        if ticker in PSU_OIL_TICKERS:
            if not any("PSU" in f or "crude" in f.lower() for f in updates["critic_flags"]):
                updates["critic_flags"].append(
                    "PSU oil stock: price driven by crude oil prices and "
                    "government fuel pricing policy — risks outside current signal library."
                )
            if updates["critic_verdict"] == "APPROVED":
                updates["critic_verdict"]   = "FLAGGED"
                updates["critic_reasoning"] += (
                    " PSU oil sector override applied: crude oil price risk "
                    "not captured in current signal library."
                )
        
    except Exception as e:
        print(f"[{ticker}] Critic Agent error: {e}")
        updates["critic_reasoning"] = f"Groq API Error: {e}"
        
    return updates
