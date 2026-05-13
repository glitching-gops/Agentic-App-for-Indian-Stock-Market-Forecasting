# Agentic Stock Forecast Prototype

This is an end-to-end stock market forecasting prototype that fetches data, computes technical indicators, trains an XGBoost model, and displays the results in a Streamlit dashboard.

## Features
- **Data Ingestion**: Fetches OHLCV data for `RELIANCE.NS` using `yfinance`.
- **Technical Signals**: Computes RSI, MACD, Bollinger Bands, OBV, and SMA using `pandas_ta`.
- **Machine Learning**: Trains an XGBoost model to predict the 30-day stock price.
- **Interactive Dashboard**: A Streamlit interface to visualize historical prices, signals, and forecasts.

## User Manual

### Prerequisites
Ensure that all dependencies are installed. You can install them using:
```bash
pip install -r requirements.txt
```

### Running the App
To run the full pipeline (which initializes the database, fetches data, computes signals, and then automatically launches the dashboard), execute the `main.py` script from the project root:
```bash
python main.py
```
This script will process the data and then open the Streamlit dashboard in your default web browser.

Alternatively, if you have already run the pipeline and the data is stored in the database, you can run the dashboard directly using Streamlit:
```bash
streamlit run app/dashboard.py
```

### Using the Dashboard
Once the Streamlit dashboard is running, you will see three main sections:
1. **Historical Price Chart**: An interactive chart displaying the historical closing price of the stock alongside its 20-day Simple Moving Average (SMA).
2. **Latest Signal Values**: A table showing the most recent values for the computed technical indicators (RSI, MACD, etc.) and a brief interpretation of what they mean.
3. **Forecast Output Card**: Key metrics including the current stock price, the 30-day forecasted price, the expected price direction (Up/Down), and the model's Mean Absolute Percentage Error (MAPE).
