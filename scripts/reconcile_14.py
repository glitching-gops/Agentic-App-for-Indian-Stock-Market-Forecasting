import subprocess
import time
import os
import requests

missing_tickers = [
    'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 
    'INDUSINDBK.NS', 'MUTHOOTFIN.NS', 'PNBHOUSING.NS', 'SBILIFE.NS',
    'SHRIRAMFIN.NS', 'CHOLAFIN.NS', 'CANBK.NS', 'DIXON.NS', 
    'ZYDUSLIFE.NS', 'SUPREMEIND.NS'
]

# Load dotenv if needed
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv("ADMIN_API_KEY", "")
batch_size = 5

for i in range(0, len(missing_tickers), batch_size):
    batch = missing_tickers[i:i + batch_size]
    print(f"Batch {i//batch_size + 1}: {batch}")
    for ticker in batch:
        try:
            r = requests.post(
                f"http://127.0.0.1:8000/api/admin/run/{ticker}",
                headers={"x-api-key": api_key},
                timeout=10
            )
            print(f"  {ticker}: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"  {ticker} failed to trigger: {e}")
            
    if i + batch_size < len(missing_tickers):
        print("Waiting 10 seconds before checking next batch... (FastAPI background tasks are running)")
        # Usually it's better to wait a few minutes, let's wait 3 minutes as per instructions
        time.sleep(180)
