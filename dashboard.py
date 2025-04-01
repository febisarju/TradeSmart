import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf  # <-- NEW: Fetch live stock data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model and preprocessors
try:
    rf_model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
except FileNotFoundError:
    rf_model, scaler, label_encoder = None, None, None

# Sidebar for user input
st.sidebar.header("Stock Prediction Input")
selected_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").upper()

# **Date Input Widgets**
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2021-01-01"))

# **Fetch Live Stock Data with Date Range**
try:
    stock_data = yf.Ticker(selected_ticker)
    hist_data = stock_data.history(start=start_date, end=end_date)  # Get data for selected date range

    if hist_data.empty:
        st.error("Invalid Ticker Symbol or No Data Available.")
    else:
        # Extract latest values
        latest_data = hist_data.iloc[-1]
        open_price = latest_data["Open"]
        high_price = latest_data["High"]
        low_price = latest_data["Low"]
        close_price = latest_data["Close"]
        volume = latest_data["Volume"]

        # Compute moving averages & volatility
        hist_data["10_MA"] = hist_data["Close"].rolling(window=10).mean()
        hist_data["20_MA"] = hist_data["Close"].rolling(window=20).mean()
        hist_data["Volatility"] = hist_data["Close"].pct_change().rolling(window=10).std()

        ten_ma = hist_data["10_MA"].iloc[-1]
        twenty_ma = hist_data["20_MA"].iloc[-1]
        volatility = hist_data["Volatility"].iloc[-1]

        st.sidebar.write(f"**Live Data for {selected_ticker}:**")
        st.sidebar.write(f"📈 Open: ${open_price:.2f}")
        st.sidebar.write(f"📊 High: ${high_price:.2f}")
        st.sidebar.write(f"📉 Low: ${low_price:.2f}")
        st.sidebar.write(f"🔄 Volume: {volume:,}")

        # Prediction Button with Loading Spinner
        if st.sidebar.button("Predict Closing Price"):
            if rf_model is None or scaler is None or label_encoder is None:
                st.error("Model files not found. Please ensure all required files are in the directory.")
            else:
                with st.spinner("Predicting... Please wait."):
                    # Encode the selected ticker
                    if selected_ticker in label_encoder.classes_:
                        ticker_encoded = label_encoder.transform([selected_ticker])[0]
                    else:
                        ticker_encoded = len(label_encoder.classes_)  # Assign a new ID for unseen tickers

                    # Prepare input features
                    features = np.array([[open_price, high_price, low_price, volume, ten_ma, twenty_ma, volatility, ticker_encoded]])
                    features_scaled = scaler.transform(features)

                    # Predict
                    predicted_price = rf_model.predict(features_scaled)[0]
                    st.success(f"🎯 Predicted Closing Price: ${predicted_price:.2f}")

        # **Plot Live Stock Price Trends**
        st.subheader(f"📊 Live Stock Price Trends for {selected_ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist_data.index, hist_data["Close"], label="Closing Price", color="blue")
        ax.plot(hist_data.index, hist_data["10_MA"], label="10-day MA", linestyle="--", color="green")
        ax.plot(hist_data.index, hist_data["20_MA"], label="20-day MA", linestyle="--", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"{selected_ticker} Stock Price Trends")
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error fetching stock data: {str(e)}")

# Display Stock Ticker Description
st.sidebar.header(" ")
with st.sidebar.expander("Stock Tickers"):
    st.write("""
    ### NOTE:
    ### ✅ U.S. Stocks (NYSE & NASDAQ)
    | Company        | Ticker  |
    |----------------|---------|
    | Apple    | `AAPL` |
    | Tesla    | `TSLA` |
    | Microsoft| `MSFT` |
    | Amazon   | `AMZN` |
    | Nvidia   | `NVDA` |
    | Google   | `GOOGL` |
    | Meta (FB)| `META` |
    | Ford Motor     | `F` |
    | Boeing         | `BA` |
    | General Electric| `GE`|
    | Adobe          | `ADBE`|

    ### ✅ Indian Stocks (NSE & BSE)
    | Company         | Ticker     |
    |-----------------|------------|
    | Reliance        | `RELIANCE.NS` |
    | Infosys         | `INFY.NS`     |
    | TCS             | `TCS.NS`      |
    | HDFC Bank       | `HDFCBANK.NS` |
    | HCL Technologies| `HCLTECH.NS` |
    | ICICI Bank      | `ICICIBANK.NS` |
    | Bharti Airtel   | `BHARTIARTL.NS` |

    ### ✅ Cryptocurrencies
    | Crypto    | Ticker   |
    |-----------|----------|
    | Bitcoin | `BTC-USD` |
    | Ethereum| `ETH-USD` |
    | Litecoin  | `LTC-USD`|
    | Ripple    | `XRP-USD`|
    | Cardano   | `ADA-USD`|

    ### ✅ Global Stock Market Indexes
    | Index      | Ticker    |
    |------------|-----------|
    | S&P 500 | `^GSPC` |
    | Dow Jones | `^DJI` |
    | NASDAQ | `^IXIC` |
    | FTSE 100   | `^FTSE`   |
    | Nikkei 225 | `^N225`   |
    | Hang Seng  | `^HSI`    |
    
    ### -------MANY MORE-------
    """)

