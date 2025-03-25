from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        input_df['Ticker'] = label_encoder.transform([data['Ticker']])[0]  # Convert text ticker to numerical
        features_scaled = scaler.transform(input_df)
        prediction = model.predict(features_scaled)
        return jsonify({'predicted_price': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Ticker', '50_MA', 'Volatility']):
            return jsonify({'error': 'Missing required features in some records'})

        df['Ticker'] = label_encoder.transform(df['Ticker'])
        features_scaled = scaler.transform(df)
        predictions = model.predict(features_scaled)

        return jsonify({'predicted_prices': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    