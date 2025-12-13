import yfinance as yf
import pandas_ta as ta
from datetime import datetime 
import matplotlib.pyplot as plt
import joblib


today = datetime.now() 

start_date = "2022-12-01"

data = yf.download("BTC-USD", start=start_date, end=today)

data.columns = data.columns.get_level_values(0)

data["Close"] = data["Close"].astype(float)

data["HL2"] = (data["High"] + data["Low"]) / 2
data["EMA400"] = ta.ema(data["HL2"], length=400)
data["RSI"] = ta.rsi(data["Close"], length=14)
data["dist_ema400"] = (data["Close"] / data["EMA400"]) - 1

data = data.dropna()

features = ["RSI", "EMA400", "Close", "dist_ema400"]

X = data[features]

model = joblib.load("models/decision_tree.plk")

data["prediction"] = model.predict(X)

plt.style.use('dark_background')

plt.figure(figsize=(18,9))

plt.plot(data.index, data["Close"], label="BTC Price", alpha=0.8, color='white')
plt.plot(data.index, data["EMA400"], label="EMA400", linestyle="--", color='orange')

buys = data[data["prediction"] == 1]
sells = data[data["prediction"] == -1]

plt.scatter(buys.index, buys["Close"], marker="^", label="Buy", s=40, color='green')
plt.scatter(sells.index, sells["Close"], marker="v", label="Sell", s=40, color='red')

plt.legend(facecolor='black')  # legend background dark
plt.title("BTC Decision Tree Signals", color='white')
plt.xlabel("Date", color='white')
plt.ylabel("Price (USD)", color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()

plt.savefig("content/images/btc-latest.png", dpi=200)
plt.close()