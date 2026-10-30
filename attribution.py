
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

if "SPY" not in prices.columns:
    raise ValueError("SPY is not in the price data.")
if prices.isna().any().any():
    raise ValueError("Some of the prices are missing.")

tickers = [t for t in prices.columns if t != "SPY"]
returns = prices.pct_change(fill_method=None)

momentum = prices.shift(21) / prices.shift(126) - 1
volatility = returns.rolling(63, min_periods=50).std() * np.sqrt(252)
score = momentum / volatility.where(volatility > 0)

spy_sma = prices["SPY"].rolling(200, min_periods=200).mean()
bull_market = prices["SPY"] > spy_sma

week_groups = prices.groupby(prices.index.to_period("W-FRI"))
rebalance_dates = set(
    week_groups.apply(lambda group: group.index[-1]).tolist()
)

first_valid_date = prices.index[max(200, 126 + 63)]
cost_rate = COST_BPS / 10_000


def target_weights(date, model):
    """Calculate target holdings using data available at this close."""
    if model == "regime" and  not bool(bull_market.loc[date]):
        return pd.Series(0.0, index=tickers)

    signal = (
        momentum.loc[date, tickers]
        if model == "raw"
        else score.loc[date, tickers])

    eligible = momentum.loc[date, tickers] > 0
    signal = signal.where(eligible).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    selected = signal.sort_values(ascending=False).head(TOP_N)

    weights = pd.Series(0.0, index=tickers)
    if len(selected):
        weights.loc[selected.index] = 1.0 / TOP_N

    return weights


def run_attribution(model):
    """
    Record daily holdings and each stock's contribution to portfolio return (daily). Holdings may drift between weekly rebalances.
    """
    holdings = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    contributions = pd.DataFrame(
        0.0, index=prices.index, columns=tickers
    )
    daily_costs = pd.Series(0.0, index=prices.index)
    portfolio_returns = pd.Series(0.0, index=prices.index)

    weights = pd.Series(0.0, index=tickers)

    for i in range(1, len(prices)):
        previous_date = prices.index[i - 1]
        date = prices.index[i]

        if previous_date in rebalance_dates:
            new_weights = target_weights(previous_date, model)
            turnover = (new_weights - weights).abs().sum()
            daily_costs.loc[date] = turnover * cost_rate
            weights = new_weights.copy()

        asset_returns = returns.loc[date, tickers]

        # record weights ofc
        holdings.loc[date] = weights
        contributions.loc[date] = weights * asset_returns

        gross_return = contributions.loc[date].sum()
        net_return = gross_return - daily_costs.loc[date]
        portfolio_returns.loc[date] = net_return

        # drift holdings according to market movements instead of rebalancing every day
        if 1 + net_return <= 0:
            raise ValueError(f"Portfolio value became nonpositive: {date}")

        weights = weights * (1 + asset_returns) / (1 + net_return)

    return holdings, contributions, daily_costs, portfolio_returns


all_holdings = {}
all_contributions = {}
summary_tables = {}

strategies = {
    "Raw Momentum": "raw",
    "Volatility-Adjusted Momentum": "volatility",
    "Volatility-Adjusted Momentum + Regime Filter": "regime"}

for name, model in strategies.items():
    holdings, contributions, costs, portfolio_returns = run_attribution(model)

    holdings = holdings.loc[first_valid_date:]
    contributions = contributions.loc[first_valid_date:]
    costs = costs.loc[first_valid_date:]
    portfolio_returns = portfolio_returns.loc[first_valid_date:]

    all_holdings[name] = holdings
    all_contributions[name] = contributions

    contribution_sum = contributions.sum()
    days_held = (holdings > 0).sum()
    average_weight = holdings.mean()
    positive_days = (contributions > 0).sum()
    negative_days = (contributions < 0).sum()

    table = pd.DataFrame({
        "Daily Contribution Sum": contribution_sum,
        "Days Held": days_held,
        "Average Portfolio Weight": average_weight,
        "Positive Contribution Days": positive_days,
        "Negative Contribution Days": negative_days,
    })

    table = table[table["Days Held"] > 0]
    table = table.sort_values("Daily Contribution Sum", ascending=False)
    summary_tables[name] = table

    print(f"\n{'=' * 90}")
    print(name.upper())
    print("=" * 90)
    print(
        table.head(10).to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nBottom 5 contributors:")
    print(
        table.tail(5).sort_values("Daily Contribution Sum").to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    gross_contribution = contributions.sum().sum()
    total_cost = costs.sum()
    total_net_daily_returns = portfolio_returns.sum()

    print(f"\nSum of stock contributions: {gross_contribution:.4f}")
    print(f"Sum of transaction costs:   {total_cost:.4f}")
    print(f"Sum of daily net returns:   {total_net_daily_returns:.4f}")

for name, holdings in all_holdings.items():
    replaced = (
        name.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
    )
    holdings.to_csv(f"holdings_{replaced}.csv")
    all_contributions[name].to_csv(f"contributions_{replaced}.csv")
    summary_tables[name].to_csv(f"attribution_summary_{replaced}.csv")

print("\nSaved holdings and contribution histories, and summary tables.")
print(f"Evaluation period: {first_valid_date.date()} to {prices.index[-1].date()}")
