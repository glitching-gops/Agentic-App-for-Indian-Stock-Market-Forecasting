import sys
import os
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()

from data.db import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.begin() as conn:
    conn.execute(text("DELETE FROM signals"))
    
print("Signals table cleared.")
