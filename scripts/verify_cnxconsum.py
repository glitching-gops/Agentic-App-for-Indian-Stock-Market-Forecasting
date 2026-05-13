import yfinance as yf

ticker = "^CNXCONSUM"
df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
print(f"Rows: {len(df)}")
if len(df) > 0:
    latest_close = df["Close"].iloc[-1]
    if hasattr(latest_close, "iloc"):
        latest_close = latest_close.iloc[0]
    print(f"Latest close: {float(latest_close):.2f}")
    print("Status: OK — proceed to Step 2")
else:
    print("Status: STILL FAILING — report back")
