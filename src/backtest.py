"""Weekly, next-period long-short backtest with transaction costs."""

from __future__ import annotations

import pandas as pd


def weekly_rebalance_dates(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available signal date in each Friday-ending week."""
    index = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period("W-FRI")).max().to_numpy())


def portfolio_turnover(current: pd.DataFrame, previous: pd.DataFrame | None) -> float:
    """One-way turnover: half the absolute change in all security weights."""
    if previous is None or previous.empty:
        return current["weight"].abs().sum() / 2
    now = current.set_index("ticker")["weight"]
    before = previous.set_index("ticker")["weight"]
    return now.sub(before, fill_value=0).abs().sum() / 2


def run_weekly_backtest(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    return_column: str = "stock_forward_return",
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Calculate gross/net weekly returns from signal-date weights and labels.

    ``realized_returns`` must contain returns beginning after each signal date;
    the target module's ``stock_forward_return`` satisfies this for a weekly
    rebalance. This function never uses a same-day execution return.
    """
    required_weights = {"date", "ticker", "weight"}
    required_returns = {"date", "ticker", return_column}
    if missing := required_weights.difference(weights.columns):
        raise ValueError(f"Weights missing: {sorted(missing)}")
    if missing := required_returns.difference(realized_returns.columns):
        raise ValueError(f"Returns missing: {sorted(missing)}")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    # Portfolio constructors often retain the realized-return column as context.
    # Reusing it avoids a duplicate merge that would create suffixed columns.
    portfolio = weights.copy() if return_column in weights.columns else weights.merge(
        realized_returns[["date", "ticker", return_column]], on=["date", "ticker"], how="left", validate="one_to_one"
    )
    rows = []
    previous: pd.DataFrame | None = None
    for date, daily in portfolio.groupby("date"):
        tradable = daily.dropna(subset=[return_column])
        if tradable.empty:
            continue
        turnover = portfolio_turnover(tradable, previous)
        gross = (tradable["weight"] * tradable[return_column]).sum()
        long_return = (tradable.loc[tradable["weight"] > 0, "weight"] * tradable.loc[tradable["weight"] > 0, return_column]).sum()
        short_return = (tradable.loc[tradable["weight"] < 0, "weight"] * tradable.loc[tradable["weight"] < 0, return_column]).sum()
        cost = turnover * transaction_cost_bps / 10_000
        rows.append({"date": date, "gross_return": gross, "net_return": gross - cost, "long_leg_return": long_return, "short_leg_return": short_return, "turnover": turnover, "transaction_cost": cost, "n_positions": len(tradable)})
        previous = tradable[["ticker", "weight"]].copy()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
