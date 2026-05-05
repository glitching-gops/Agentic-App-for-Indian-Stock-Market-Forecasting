from data.db import get_engine
import pandas as pd
from data.tickers import TICKERS

engine = get_engine()
leaderboard = pd.read_sql("SELECT ticker FROM leaderboard", con=engine)
all_tickers = list(TICKERS.keys())
missing = [t for t in all_tickers if t not in leaderboard["ticker"].values]
print(f"Total on leaderboard: {len(leaderboard)}")
print(f"Missing: {missing if missing else 'None'}")
