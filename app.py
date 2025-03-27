from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model, scaler and label encoder
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl") 

app = Flask(__name__)

@app.route('/')
def home():
    return "Stock Market Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_features = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Ticker', '50_MA', 'Volatility']
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'})
        input_df = pd.DataFrame([data], columns=required_features)
        try:
            input_df['Ticker'] = label_encoder.transform([data['Ticker']])[0]  
        except ValueError:
            return jsonify({'error': f"Unknown Ticker: {data['Ticker']}. Model only supports trained tickers."})
        features_scaled = scaler.transform(input_df)
        predicted_price = model.predict(features_scaled)[0]
        return jsonify({'actual_price': data['Adj Close'], 'predicted_price': float(predicted_price)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
