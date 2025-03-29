# TradeSmart: Stock Market trends & predictions
This is a stock market analysis and prediction project that leverages historical stock price data for Apple, Microsoft, Netflix, and Google. It includes machine learning models for predicting stock prices and a deployed Flask API for real-time predictions.                                    
**[Analysis Overview (html file)] (https://febisarju.github.io/TradeSmart)**
                         
## Features:
  
- Identify trends and patterns in stock prices.
- Compute moving averages (10_MA, 20_MA) and stock volatility.
- Conduct correlation analysis to examine relationships between stocks.
- Train machine learning models to predict stock prices.
- Deploy a Flask API for real-time stock price predictions.         

## Tools & Technologies Used:

- Python (Jupyter Notebook and VS code).
- Machine Learning (ML) (Random Forest, Linear Regression, Decision Tree).
- Pandas, NumPy, Matplotlib, Seaborn.
- SQL & Excel for data storage.
- Flask API for real-time predictions.

## Dataset(stocks.csv):

The dataset contains historical stock price data, including:
- Ticker (Stock Symbol)
- Date
- Open, High, Low, Close, Adj Close Prices
- Volume

## Workflow:

1️. Data Collection
- Load historical stock data.
- Clean and preprocess the dataset.
  
2️. Exploratory Data Analysis (EDA)
- Visualizations using Matplotlib & Seaborn.
- Correlation Matrix to find relationships.
- Moving Averages & Volatility analysis.
  
3️. Feature Engineering
- Compute 10-day and 20-day moving averages.
- Calculate stock price volatility.
- Encode Ticker as Ticker_Encoded for ML models.

4️. Machine Learning Models
- Random Forest Regressor 
- Linear Regression 
- Decision Tree Regressor               
Models are evaluated using Mean Squared Error (MSE) and R² Score.

5️. Hyperparameter Tuning & Cross-Validation
- Optimize Random Forest using GridSearchCV.
- Use 5-fold Cross-Validation for better performance.

6️. Model Deployment (Flask API)
- Develop a Flask API (app.py) to serve predictions.
- Accepts stock features as input and returns the predicted stock price.(API Inputs: Open, High, Low, Volume, 10_MA, 20_MA, Volatility, Ticker and API Output: Predicted stock price.)

## Running the Flask API:

1. Run the API    
python app.py

2. Test in Postman   
Send a POST request to http://127.0.0.1:5000/predict with JSON input:       
![Screenshot 2025-03-29 215803](https://github.com/user-attachments/assets/8dde808e-8d51-4a0d-a4b1-b8a03df32fb6)

## Contact:

- Github- https://github.com/febisarju
