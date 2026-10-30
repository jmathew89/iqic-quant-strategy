import numpy as np
import pandas as pd

PRICE_FILE = "iqic_adjusted_prices.csv"
TOP_N = 10
COST_BPS = 5
TRADING_DAYS = 252

prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
prices.columns = prices.columns.astype(str)
prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
prices = prices.dropna(how="all")

if "SPY" not in prices.columns or prices.isna().any().any():
    raise ValueError("SPY is missing or the price data contains missing values.")

tickers = [t for t in prices.columns if t != "SPY"]
returns = prices.pct_change(fill_method=None)
momentum = prices.shift(21) / prices.shift(126) - 1

volatility = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
score = momentum / volatility

spy_sma200 = prices["SPY"].rolling(200).mean()
week_groups = prices.groupby(prices.index.to_period("W-FRI"))
rebalance_dates = set(
    week_groups.apply(lambda group: group.index[-1]).tolist()
)

first_valid_date = prices.index[max(200, 126 + 63)]
cost_rate = COST_BPS / 10_000

# compare each exclusion against the same original stock list.
exclusion_scenarios = {
    "Original universe": [],
    "Exclude NVDA": ["NVDA"],
    "Exclude AVGO": ["AVGO"],
    "Exclude LLY": ["LLY"],
    "Exclude all three": ["NVDA", "AVGO", "LLY"],
}


def run_strategy(excluded):
    available = [t for t in tickers if t not in excluded]
    weights = pd.Series(0.0, index=available)
    portfolio_returns = pd.Series(0.0, index=prices.index)

    for i in range(1, len(prices)):
        previous_date = prices.index[i - 1]
        date = prices.index[i]
        cost = 0.0

        if previous_date in rebalance_dates:
            target = pd.Series(0.0, index=available)

            # hold cash unless SPY is above its 200-day average.
            if prices.loc[previous_date, "SPY"] > spy_sma200.loc[previous_date]:
                signal = momentum.loc[previous_date, available]
                ranking = score.loc[previous_date, available]

                eligible = signal[signal > 0].dropna().index
                ranking = ranking.loc[eligible].dropna()
                selected = ranking.sort_values(ascending=False).head(TOP_N)

                if len(selected):
                    target.loc[selected.index] = 1.0 / TOP_N

            turnover = (target - weights).abs().sum()
            cost = turnover * cost_rate
            weights = target.copy()

        asset_returns = returns.loc[date, available]
        net_return = (weights * asset_returns).sum() - cost
        portfolio_returns.loc[date] = net_return

        if 1 + net_return <= 0:
            raise ValueError(f"Nonpositive portfolio value on {date}")

        weights = weights * (1 + asset_returns) / (1 + net_return)

    return portfolio_returns.loc[first_valid_date:]


def metrics(strategy_returns):
    r = strategy_returns.dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS

    annual_return = (
        wealth.iloc[-1] ** (1 / years) - 1
        if years > 0 and wealth.iloc[-1] > 0
        else np.nan
    )

    volatility = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if r.std(ddof=1) > 0
        else np.nan
    )

    max_drawdown = (wealth / wealth.cummax() - 1).min()

    return {
        "Cumulative Return": wealth.iloc[-1] - 1,
        "Annualized Return": annual_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
    }


results = {}

for name, excluded in exclusion_scenarios.items():
    results[name] = metrics(run_strategy(excluded))

spy_returns = returns.loc[first_valid_date:, "SPY"].fillna(0.0)
results["SPY Buy-and-Hold"] = metrics(spy_returns)

report = pd.DataFrame(results).T
report.to_csv("sensitivity_results.csv")

print("\nLEAVE-THE-WINNERS-OUT SENSITIVITY")
print("=" * 90)
print(report.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nEvaluation period: {first_valid_date.date()} to {prices.index[-1].date()}")
print("\nSaved sensitivity_results.csv")