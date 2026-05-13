import sys
import os
sys.path.append(os.getcwd())

from pipeline.macro import fetch_fii_dii_flows
df = fetch_fii_dii_flows()
print(f"Rows fetched: {len(df)}")
if not df.empty:
    print(df.tail(10).to_string())
    print(f"\nNaN count: {df.isna().sum().to_dict()}")
    print(f"FII range: {df['fii_net_flow'].min():.0f} to {df['fii_net_flow'].max():.0f} crores")
    print(f"DII range: {df['dii_net_flow'].min():.0f} to {df['dii_net_flow'].max():.0f} crores")
