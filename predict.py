import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_data(Token):
    data = yf.download(Token, period="10y")
    return data

df = get_data("AAPL")
df = df.reset_index()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()
df['Spread'] = df['MA_50'] - df['MA_200']
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Close'].rolling(window=10).std()
df = df.dropna()

Train = df.iloc[:int(len(df) * 0.8)]
Test = df.iloc[int(len(df) * 0.8):]

X_train = Train[['MA_50', 'MA_200', 'Spread', 'Volume', 'Daily_Return', 'Volatility']]
Y_train = Train['Target']
X_test = Test[['MA_50', 'MA_200', 'Spread', 'Volume', 'Daily_Return', 'Volatility']]
Y_test = Test['Target']

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
print(accuracy_score(Y_test, prediction))