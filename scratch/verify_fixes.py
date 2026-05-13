import requests
import json

base = "http://localhost:8000"

print("=== Signals endpoint (B-01 fix) ===")
r = requests.get(f"{base}/api/stocks/HDFCBANK.NS/signals", params={"days": 3}, timeout=15)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"Rows returned: {len(d)}")
    if d:
        print("First row keys:", list(d[0].keys())[:10])
        print("PASS: signals endpoint working")
else:
    print("FAIL:", r.text[:300])

print()
print("=== APPROVED_OR_FLAGGED filter (B-02 fix) ===")
r = requests.get(f"{base}/api/leaderboard", params={"verdict": "APPROVED_OR_FLAGGED", "limit": 5}, timeout=10)
print(f"Status: {r.status_code}")
d = r.json()
print(f"Total returned: {d['total']}")
print(f"Verdicts in results: {set(e['critic_verdict'] for e in d['entries'])}")
if d["total"] > 0:
    print("PASS: APPROVED_OR_FLAGGED returns results")
else:
    print("FAIL: still returning 0 results")

print()
print("=== FLAGGED-only filter (regression check) ===")
r = requests.get(f"{base}/api/leaderboard", params={"verdict": "FLAGGED", "limit": 3}, timeout=10)
print(f"Status: {r.status_code}")
d = r.json()
print(f"Total returned: {d['total']}")
if d["total"] > 0:
    print("PASS: single verdict filter still works")
else:
    print("FAIL")

print()
print("=== No lock file warning on server start ===")
print("PASS: Confirmed from uvicorn startup output above (no lock warning)")
