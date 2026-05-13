import pandas as pd
from data.db import get_engine
from sqlalchemy import text

def find_errors():
    engine = get_engine()
    df = pd.read_sql(text("SELECT ticker, critic_reasoning, signal_narrative FROM forecasts WHERE critic_reasoning LIKE '%%429%%' OR signal_narrative LIKE '%%429%%'"), con=engine)
    print(f"Found {len(df)} rows with 429 errors")
    print("Tickers:", df['ticker'].tolist())

if __name__ == "__main__":
    find_errors()
