import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "JPM", "LLY", "AVGO",
    "XOM", "WMT", "COST", "V", "MA",
    "SPY"
]

START_DATE = "2018-01-01"

raw = yf.download(
    TICKERS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
    group_by="column",
    threads=True
)

if raw.empty:
    raise RuntimeError("No price data returned. Check the connection.")

prices = raw["Close"].copy()
volumes = raw["Volume"].copy()

prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
volumes = volumes.sort_index().apply(pd.to_numeric, errors="coerce")

# Remove rows where every requested ticker is missing.
prices = prices.dropna(how="all")
volumes = volumes.reindex(prices.index)

# basic tests.
print("Price data shape:", prices.shape)
print("Date range:", prices.index.min(), "to", prices.index.max())
print("\nMissing price observations:")
print(prices.isna().sum().sort_values(ascending=False))
print("\nLatest available prices:")
print(prices.iloc[-1].sort_values())

prices.to_csv("iqic_adjusted_prices.csv")
volumes.to_csv("iqic_volumes.csv")


# daily percentage returns
daily_returns = prices.pct_change(fill_method=None)

momentum = prices.shift(21) / prices.shift(126) - 1

# Approximately 3-month annualized historical volatility.
volatility = (
    daily_returns
    .rolling(63, min_periods=50)
    .std()
    * np.sqrt(252)
)

# risk-adjusted momentum score is unavailable until we have both inputs
score = momentum / volatility.where(volatility > 0)

latest = pd.DataFrame({
    "momentum": momentum.iloc[-1],
    "annualized_volatility": volatility.iloc[-1],
    "score": score.iloc[-1]
})

latest = latest.drop(index="SPY", errors="ignore")
latest = latest.dropna().sort_values("score", ascending=False)

print(latest.to_string())
