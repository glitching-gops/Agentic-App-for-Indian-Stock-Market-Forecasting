import requests
import json

base = "http://localhost:8000/api"
endpoints = [
    ("1. Leaderboard — top 10 by composite score", f"{base}/leaderboard?limit=10&sort_by=composite_score"),
    ("2. Leaderboard — Banking sector only", f"{base}/leaderboard?sector=Banking%20%26%20Finance"),
    ("3. Leaderboard — Approved verdicts only", f"{base}/leaderboard?verdict=APPROVED"),
    ("4. Leaderboard — sorted by MAPE ascending", f"{base}/leaderboard?sort_by=mape&limit=5"),
    ("5. Forecast for a high-performing stock (AXISBANK is missing, using HDFCBANK)", f"{base}/forecasts/HDFCBANK.NS"),
    ("6. Forecast for a flagged stock (ONGC)", f"{base}/forecasts/ONGC.NS")
]

for title, url in endpoints:
    print(f"=== {title} ===")
    try:
        r = requests.get(url)
        # Just print summary to avoid massive output
        data = r.json()
        if "entries" in data:
            print(f"Total: {data['total']}")
            for e in data['entries'][:3]:
                print(f"  {e['ticker']}: {e['composite_score']} (Verdict: {e['critic_verdict']})")
        else:
            print(f"Ticker: {data.get('ticker')}")
            print(f"Verdict: {data.get('critic_verdict')}")
            print(f"MAPE: {data.get('mape')}")
    except Exception as e:
        print(f"Error: {e}")
    print()
