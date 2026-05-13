import sqlite3
import pandas as pd
import os

db_path = "stock_forecast.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM leaderboard", conn)
        print(df.to_markdown())
    except Exception as e:
        print(f"Error: {e}")
    conn.close()
else:
    print("Database not found.")
