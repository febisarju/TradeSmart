from flask import Flask, request, jsonify 
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model, scaler, and label encoder
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    model, scaler, label_encoder = None, None, None
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return "✅ Stock Market Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoder is None:
        return jsonify({'error': '❌ Model files not found. Ensure all required .pkl files are in the directory.'}), 500

    try:
        data = request.get_json()
        
        # Define required features based on training
        required_features = ['Open', 'High', 'Low', 'Volume', '10_MA', '20_MA', 'Volatility', 'Ticker']
        missing_features = [f for f in required_features if f not in data]

        # Check for missing features
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Convert JSON input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure Ticker exists in label encoder
        if data['Ticker'] not in label_encoder.classes_.tolist():
            return jsonify({'error': f"❌ Unknown Ticker: {data['Ticker']}. Model only supports: {list(label_encoder.classes_)}"}), 400

        # Encode 'Ticker'
        input_df['Ticker_Encoded'] = label_encoder.transform([data['Ticker']])[0]

        # Drop original 'Ticker' column
        input_df = input_df.drop(columns=['Ticker'])

        # Ensure the input features match model training
        input_df = input_df[['Open', 'High', 'Low', 'Volume', '10_MA', '20_MA', 'Volatility', 'Ticker_Encoded']]

        # Scale input features
        features_scaled = scaler.transform(input_df)

        # Predict
        predicted_price = model.predict(features_scaled)[0]

        return jsonify({
            'predicted_price': float(predicted_price),
            'status': '✅ Prediction Successful'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
