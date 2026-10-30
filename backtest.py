import numpy as np
import pandas as pd

PRICE_FILE = "iqic_adjusted_prices.csv"
VOLUME_FILE = "iqic_volumes.csv"

TOP_N = 10
TRANSACTION_COST_BPS = 5  # 5 basis points per dollar  
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0      


prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
prices.columns = prices.columns.astype(str)
prices = prices.sort_index()
prices = prices.apply(pd.to_numeric, errors="coerce")
prices = prices.dropna(how="all")

if "SPY" not in prices.columns:
    raise ValueError("SPY is not in the price data.")

if prices.index.has_duplicates:
    raise ValueError("duplicate dates are found in the price data.")

if prices.isna().any().any():
    missing = prices.columns[prices.isna().any()].tolist()
    raise ValueError(f"Missing prices remain for these: {missing}")

if (prices <= 0).any().any():
    raise ValueError("all prices must be positive.")

tickers = [ticker for ticker in prices.columns if ticker != "SPY"]

if len(tickers) < TOP_N:
    print(f"warning: only {len(tickers)} candidate stocks are available.")


daily_returns = prices.pct_change(fill_method=None)

# momentum: approximately six month return, excluding the latest month.
momentum = prices.shift(21) / prices.shift(126) - 1

# ts annual volatility estimated from the previous 63 trading sessions.
volatility = (
    daily_returns.rolling(63, min_periods=63).std()
    * np.sqrt(TRADING_DAYS)
)

# momentum has to be adjusted for volatility
score = momentum / volatility.where(volatility > 0)

# market-regime filter: SPY above 200day moving average.
spy_sma_200 = prices["SPY"].rolling(200, min_periods=200).mean()
bull_market = prices["SPY"] > spy_sma_200

# rebalance at the final available trading session of each week, which is friday.
week_groups = prices.groupby(prices.index.to_period("W-FRI"))
rebalance_dates = set(week_groups.apply(lambda group: group.index[-1]).tolist())

def make_weights(signal_date, model):
    """
    Select stocks using information available at signal_date, and return target stock weights. Unallocated money will stay in cash.
    """
    if model == "momentum_volatility_regime":
        if not bool(bull_market.loc[signal_date]):
            return pd.Series(0.0, index=tickers)

    if model == "momentum":
        signal = momentum.loc[signal_date, tickers]
    else:
        signal = score.loc[signal_date, tickers]

    # only stocks with positive  momentum are eligible to be selected.
    eligible = momentum.loc[signal_date, tickers] > 0
    signal = signal.where(eligible).dropna()

    signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
    selected = signal.sort_values(ascending=False).head(TOP_N)

    weights = pd.Series(0.0, index=tickers)

    if len(selected) > 0:
        weights.loc[selected.index] = 1.0 / TOP_N

    return weights


def run_backtest(model):
    """
    Signals are evaluated after a session closes (which is usually on Fridays).
    Target weights alwaysapply to the next session's return.

    Between rebalances, holdings drift with stock prices.
    Transaction costs are deducted when target weights change.
    Unallocated money remains in cash earning the assumed 0% rate.
    """
    portfolio_returns = pd.Series(0.0, index=prices.index, dtype=float)
    turnover_series = pd.Series(0.0, index=prices.index, dtype=float)

    current_weights = pd.Series(0.0, index=tickers)
    cost_rate = TRANSACTION_COST_BPS / 10_000

    for i in range(1, len(prices)):
        previous_date = prices.index[i - 1]
        current_date = prices.index[i]

        transaction_cost = 0.0

        if previous_date in rebalance_dates:
            new_weights = make_weights(previous_date, model)

            gross_turnover = (new_weights - current_weights).abs().sum()
            transaction_cost = gross_turnover * cost_rate

            turnover_series.loc[current_date] = gross_turnover
            current_weights = new_weights.copy()

        asset_returns = daily_returns.loc[current_date, tickers]

        gross_return = (current_weights * asset_returns).sum()
        net_return = gross_return - transaction_cost

        portfolio_returns.loc[current_date] = net_return

        
        portfolio_growth = 1 + net_return

        if portfolio_growth <= 0:
            raise ValueError(
                f"portfolio value became nonpositive on {current_date}."
            )

        current_weights = (
            current_weights * (1 + asset_returns) / portfolio_growth
        )

    return portfolio_returns, turnover_series


def calculate_metrics(strategy_returns, benchmark_returns):
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()

    strategy = aligned["strategy"]
    benchmark = aligned["benchmark"]

    if len(strategy) < 2:
        raise ValueError("There aren't enough observations to calculate the metrics.")

    wealth = (1 + strategy).cumprod()
    benchmark_wealth = (1 + benchmark).cumprod()

    years = len(strategy) / TRADING_DAYS

    cumulative_return = wealth.iloc[-1] - 1
    annualized_return = wealth.iloc[-1] ** (1 / years) - 1

    annualized_volatility = strategy.std(ddof=1) * np.sqrt(TRADING_DAYS)

    daily_rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    excess_returns = strategy - daily_rf

    if excess_returns.std(ddof=1) > 0:
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )
    else:
        sharpe_ratio = np.nan

    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = drawdown.min()

    benchmark_excess = benchmark - daily_rf
    strategy_excess = strategy - daily_rf

    benchmark_variance = benchmark_excess.var(ddof=1)

    if benchmark_variance > 0:
        beta = strategy_excess.cov(benchmark_excess) / benchmark_variance
    else:
        beta = np.nan

    alpha = (
        strategy_excess.mean() - beta * benchmark_excess.mean()
    ) * TRADING_DAYS

    if pd.notna(beta) and abs(beta) > 1e-12:
        treynor_ratio = (
            strategy_excess.mean() * TRADING_DAYS / beta
        )
    else:
        treynor_ratio = np.nan

    benchmark_cumulative_return = benchmark_wealth.iloc[-1] - 1

    return {
        "Observations": len(strategy),
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Beta vs SPY": beta,
        "Annualized Alpha": alpha,
        "Treynor Ratio": treynor_ratio,
        "SPY Cumulative Return": benchmark_cumulative_return,
    }



models = {
    "SPY Buy-and-Hold": None,
    "Raw Momentum": "momentum",
    "Volatility-Adjusted Momentum": "momentum_volatility",
    "Volatility-Adjusted Momentum + Regime Filter":
        "momentum_volatility_regime",
}

all_returns = {}
all_turnover = {}
benchmark_returns = daily_returns["SPY"].fillna(0.0)

# SPY benchmark (buy and hold)
all_returns["SPY Buy-and-Hold"] = benchmark_returns.copy()
all_turnover["SPY Buy-and-Hold"] = pd.Series(0.0, index=prices.index)

for name, model in models.items():
    if model is None:
        continue

    returns, turnover = run_backtest(model)
    all_returns[name] = returns
    all_turnover[name] = turnover

first_valid_date = prices.index[200]

results = {}

for name, returns in all_returns.items():
    evaluated_returns = returns.loc[first_valid_date:]
    evaluated_benchmark = benchmark_returns.loc[first_valid_date:]

    results[name] = calculate_metrics(
        evaluated_returns,
        evaluated_benchmark,
    )

metrics = pd.DataFrame(results).T

# save outputs
metrics.to_csv("backtest_metrics.csv")

returns_df = pd.DataFrame(all_returns)
returns_df.to_csv("backtest_daily_returns.csv")

turnover_df = pd.DataFrame(all_turnover)
turnover_df.to_csv("backtest_turnover.csv")

print("\nBACKTEST RESULTS")
print("=" * 90)
print(metrics.to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\nEvaluation period: {first_valid_date.date()} to {prices.index[-1].date()}")
print(f"Transaction cost: {TRANSACTION_COST_BPS} bps per dollar of turnover")
print("\nSaved:")
print("  backtest_metrics.csv")
print("  backtest_daily_returns.csv")
print("  backtest_turnover.csv")