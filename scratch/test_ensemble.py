from pipeline.model import train_and_forecast

print("Testing ensemble on RELIANCE.NS...")
results = train_and_forecast("RELIANCE.NS")
result = results["RELIANCE.NS"]

print(f"\n=== RELIANCE.NS Ensemble Result ===")
print(f"Current price:     {result['current_price']:.2f}")
print(f"XGBoost forecast:  {result.get('xgb_forecast_price', 'N/A')}")
print(f"LSTM forecast:     {result.get('lstm_forecast_price', 'N/A')}")
print(f"Ensemble forecast: {result['forecast_price']:.2f}")
print(f"Direction:         {result['direction']}")
print(f"Change:            {result['change_pct']}%")
print(f"MAPE:              {result['mape']:.2f}%")
print(f"Dir Accuracy:      {result['directional_accuracy']:.2f}%")
print(f"Device used:       {'CUDA' if result.get('device') == 'cuda' else 'CPU'}")
