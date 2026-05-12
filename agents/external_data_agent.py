# external_data_agent.py
import pandas as pd
from datetime import datetime
from data.db import get_engine
from pipeline.sentiment import get_aggregate_sentiment, fetch_and_score
from pipeline.macro import fetch_and_store
from agents.state import AgentState

def external_data_node(state: AgentState) -> dict:
    """
    Fetches the latest sentiment and macro data for the given ticker.
    Fetches new data if today's data is missing.
    Returns plain dictionaries for JSON serializability.
    """
    ticker = state["ticker"]
    engine = get_engine()
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    # Check if we have today's sentiment
    sentiment_df = pd.read_sql(f"SELECT * FROM sentiment WHERE ticker = '{ticker}' AND date = '{today_str}'", con=engine)
    if sentiment_df.empty:
        print(f"[{ticker}] Sentiment not up to date for today. Fetching...")
        fetch_and_score(ticker)
        
    agg_score = get_aggregate_sentiment(ticker)
    
    # Load macro data
    macro_df = pd.read_sql(
        "SELECT * FROM macro ORDER BY date ASC", 
        con=engine
    )
    if macro_df.empty or macro_df.iloc[-1]["date"] != today_str:
        print("Macro not up to date for today. Fetching...")
        fetch_and_store()
        macro_df = pd.read_sql("SELECT * FROM macro ORDER BY date ASC", con=engine)
        
    if not macro_df.empty:
        macro_dict = macro_df.to_dict(orient="records")
    else:
        macro_dict = []
        
    return {
        "sentiment_score": agg_score,
        "macro_df": macro_dict
    }
