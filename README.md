# Stock Direction Predictor

A Python tool that uses a Random Forest classifier to predict whether a stock's closing price will go up or down the next trading day, based on technical indicators.

## Features
- Fetches 10 years of historical stock data for any ticker
- Engineers technical features: moving averages, spread, volatility, and daily return
- Trains a Random Forest model on an 80/20 train/test split
- Outputs test set accuracy score

## Technologies Used
- Python
- yfinance
- pandas
- scikit-learn

## How to Use
1. Clone the repository
2. Install dependencies: `pip install yfinance pandas scikit-learn`
3. Edit the `get_data()` call in `main.py` with your desired ticker
4. Run `python main.py`

## Sample Output
```
0.5321
```