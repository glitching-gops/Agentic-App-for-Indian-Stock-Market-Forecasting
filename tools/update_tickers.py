import ast

with open('data/tickers.py', 'r', encoding='utf-8') as f:
    code = f.read()

# We know the selected tickers:
selected = ['M&M.NS', 'TVSMOTOR.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'BOSCHLTD.NS', 'HDFCBANK.NS', 'PNBHOUSING.NS', 'BAJFINANCE.NS', 'ICICIPRULI.NS', 'BAJAJFINSV.NS', 'TRENT.NS', 'BERGEPAINT.NS', 'PIDILITIND.NS', 'ASIANPAINT.NS', 'DMART.NS', 'RELIANCE.NS', 'ADANIENT.NS', 'GAIL.NS', 'ADANIPORTS.NS', 'NTPC.NS', 'COLPAL.NS', 'EMAMILTD.NS', 'VBL.NS', 'GODREJCP.NS', 'DABUR.NS', 'MPHASIS.NS', 'WIPRO.NS', 'PERSISTENT.NS', 'HCLTECH.NS', 'TECHM.NS', 'HAVELLS.NS', 'SUPREMEIND.NS', 'ADANIGREEN.NS', 'SIEMENS.NS', 'CUMMINSIND.NS', 'JSWSTEEL.NS', 'VEDL.NS', 'NMDC.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'CIPLA.NS', 'SUNPHARMA.NS', 'ALKEM.NS', 'DRREDDY.NS', 'ABBOTINDIA.NS', 'OBEROIRLTY.NS', 'GODREJPROP.NS', 'DLF.NS', 'PRESTIGE.NS', 'PHOENIXLTD.NS', 'BHARTIARTL.NS', 'IDEA.NS', 'INDUSTOWER.NS']

# Import original to get the definitions
sys_path_added = False
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
    sys_path_added = True

from data.tickers import TICKERS, SECTOR_INDICES

new_tickers = {}
for t in selected:
    if t in TICKERS:
        new_tickers[t] = TICKERS[t]

# Generate the new TICKERS string
lines = ["TICKERS = {"]
for t, v in new_tickers.items():
    lines.append(f'    "{t}": {{"company": "{v["company"]}", "sector": "{v["sector"]}"}},')
lines.append("}")
new_tickers_str = "\n".join(lines)

# Now we need to replace the TICKERS block in the file string
# We can find where 'TICKERS = {' starts and where 'SECTOR_INDICES = {' starts
start_idx = code.find('TICKERS = {')
end_idx = code.find('SECTOR_INDICES = {')

if start_idx != -1 and end_idx != -1:
    new_code = code[:start_idx] + new_tickers_str + "\n\n" + code[end_idx:]
    with open('data/tickers.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    print("Successfully updated data/tickers.py")
else:
    print("Could not find TICKERS or SECTOR_INDICES block")
