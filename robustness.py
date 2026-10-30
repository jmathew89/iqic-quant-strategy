import pandas as pd
import numpy as np

returns = pd.read_csv(
    "backtest_daily_returns.csv",
    index_col=0,
    parse_dates=True,
).sort_index()

periods = {
    "2018-10 to 2020-12": ("2018-10-17", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026 YTD": ("2026-01-01", "2026-10-23"),
}

strategies = [
    "SPY Buy-and-Hold",
    "Raw Momentum",
    "Volatility-Adjusted Momentum",
    "Volatility-Adjusted Momentum + Regime Filter",
]


def summarize(series):
    series = series.dropna()

    if len(series) == 0:
        return None

    wealth = (1 + series).cumprod()
    years = len(series) / 252

    annualized_return = (
        wealth.iloc[-1] ** (1 / years) - 1
        if wealth.iloc[-1] > 0 and years > 0
        else np.nan
    )

    volatility = series.std(ddof=1) * np.sqrt(252)
    sharpe = (
        series.mean() / series.std(ddof=1) * np.sqrt(252)
        if series.std(ddof=1) > 0
        else np.nan
    )

    max_drawdown = (wealth / wealth.cummax() - 1).min()

    return {
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
    }


rows = []

for period, (start, end) in periods.items():
    for strategy in strategies:
        if strategy not in returns.columns:
            continue

        result = summarize(returns.loc[start:end, strategy])

        if result is not None:
            rows.append({
                "Period": period,
                "Strategy": strategy,
                **result,
            })

report = pd.DataFrame(rows)
report.to_csv("robustness_results.csv", index=False)

print("\nPERIOD-BY-PERIOD RESULTS")
print("=" * 100)

for period in periods:
    subset = report[report["Period"] == period]

    print(f"\n{period}")
    print(
        subset.drop(columns="Period")
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
))

print("\nSaved robustness_results.csv")
