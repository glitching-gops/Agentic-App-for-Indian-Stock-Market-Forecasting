import subprocess
import time
import os
import requests

API_KEY = "zero123"

batches = [
    ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
    ["BAJFINANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    ["TECHM.NS", "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "ONGC.NS"],
    ["BPCL.NS", "POWERGRID.NS", "MARUTI.NS", "TATAMOTORS.NS", "LT.NS"]
]

for idx, batch in enumerate(batches):
    print(f"Starting batch {idx+1}...")
    for ticker in batch:
        try:
            res = requests.post(f"http://localhost:8000/api/admin/run/{ticker}", headers={"x-api-key": API_KEY})
            print(f"{ticker}: {res.status_code} {res.text}")
        except Exception as e:
            print(f"Failed to trigger {ticker}: {e}")
    
    if idx < len(batches) - 1:
        print("Waiting 180 seconds...")
        time.sleep(180)

print("All batches triggered. Waiting 5 minutes before ending...")
time.sleep(300)
print("Done.")
