# trading_data_agent.py
import pandas as pd
from datetime import datetime
from data.db import get_engine
from agents.state import AgentState
from pipeline.signals import compute_and_store

def trading_data_node(state: AgentState) -> dict:
    """
    Fetches the latest technical signals for the given ticker from the database.
    Computes them if today's data is missing.
    Returns plain dictionaries to ensure LangGraph state is JSON serializable.
    """
    ticker = state["ticker"]
    engine = get_engine()
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    # Load all signals for the ticker
    signals_df = pd.read_sql(
        f"SELECT * FROM signals WHERE ticker = '{ticker}' ORDER BY date ASC", 
        con=engine
    )
    
    # Run compute if missing today's data
    if signals_df.empty or signals_df.iloc[-1]["date"] != today_str:
        print(f"[{ticker}] Signals not up to date for today. Recomputing...")
        compute_and_store(ticker)
        signals_df = pd.read_sql(
            f"SELECT * FROM signals WHERE ticker = '{ticker}' ORDER BY date ASC", 
            con=engine
        )
    
    if not signals_df.empty:
        signals_df.set_index("date", inplace=True)
        latest_row = signals_df.iloc[-1].to_dict()
        current_price = latest_row.get("close", 0.0)
        signals_dict = signals_df.reset_index().to_dict(orient="records")
    else:
        latest_row = {}
        current_price = 0.0
        signals_dict = []
        
    return {
        "signals_df": signals_dict,
        "latest_signals": latest_row,
        "current_price": current_price
    }
