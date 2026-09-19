"""Weekly, next-period long-short backtest with transaction costs."""

from __future__ import annotations

import pandas as pd
import numpy as np


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


def estimated_transaction_cost(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    *,
    cost_model: str = "flat",
    flat_bps: float = 10.0,
    half_spread_bps: float = 2.0,
    impact_coefficient: float = 0.1,
    portfolio_notional: float = 1_000_000.0,
    dollar_volume_column: str = "dollar_volume_20d",
    volatility_column: str = "volatility_20d",
) -> tuple[float, float]:
    """Estimate trading cost and one-way turnover for a rebalance.

    The liquidity model applies a half-spread plus square-root impact model per
    security.  It is deliberately an *assumption model*, not evidence that the
    public daily data captures intraday execution.  Its inputs must therefore
    be reported with every run.
    """
    if cost_model not in {"flat", "liquidity"}:
        raise ValueError("cost_model must be 'flat' or 'liquidity'")
    if min(flat_bps, half_spread_bps, impact_coefficient, portfolio_notional) < 0:
        raise ValueError("Cost parameters must be non-negative")
    now = current.set_index("ticker").copy()
    previous_weights = pd.Series(dtype=float) if previous is None or previous.empty else previous.set_index("ticker")["weight"]
    trade_weights = now["weight"].sub(previous_weights, fill_value=0.0)
    if previous is not None and not previous.empty:
        exited = previous.set_index("ticker")["weight"].loc[lambda values: ~values.index.isin(now.index)]
        trade_weights = pd.concat([trade_weights, -exited])
    turnover = trade_weights.abs().sum() / 2
    if cost_model == "flat":
        return float(turnover * flat_bps / 10_000), float(turnover)
    required = {dollar_volume_column, volatility_column}
    if missing := required.difference(now.columns):
        raise ValueError(f"Liquidity cost model requires: {sorted(missing)}")
    traded = trade_weights.reindex(now.index, fill_value=0.0).abs()
    adv = now[dollar_volume_column].astype(float)
    volatility = now[volatility_column].astype(float)
    if adv.isna().any() or volatility.isna().any() or (adv <= 0).any() or (volatility < 0).any():
        raise ValueError("Liquidity inputs must be non-missing; dollar volume must be positive")
    participation = (traded * portfolio_notional).div(adv)
    impact_bps = impact_coefficient * volatility * np.sqrt(participation) * 10_000
    per_name_bps = half_spread_bps + impact_bps
    # Half the L1 change is the convention used by ``portfolio_turnover``.
    trading_cost = (traded / 2 * per_name_bps / 10_000).sum()
    # An exited security may have no current ADV/volatility.  Apply the stated
    # half-spread assumption rather than silently excluding its exit cost.
    exited_cost = (trade_weights.loc[~trade_weights.index.isin(now.index)].abs() / 2 * half_spread_bps / 10_000).sum()
    return float(trading_cost + exited_cost), float(turnover)


def run_weekly_backtest(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    return_column: str = "stock_forward_return",
    transaction_cost_bps: float = 10.0,
    cost_model: str = "flat",
    half_spread_bps: float = 2.0,
    impact_coefficient: float = 0.1,
    portfolio_notional: float = 1_000_000.0,
    annual_borrow_bps: float = 0.0,
    periods_per_year: int = 52,
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
    if transaction_cost_bps < 0 or annual_borrow_bps < 0 or periods_per_year < 1:
        raise ValueError("Cost parameters must be non-negative and periods_per_year positive")
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
        trading_cost, turnover = estimated_transaction_cost(
            tradable, previous, cost_model=cost_model, flat_bps=transaction_cost_bps,
            half_spread_bps=half_spread_bps, impact_coefficient=impact_coefficient,
            portfolio_notional=portfolio_notional,
        )
        gross = (tradable["weight"] * tradable[return_column]).sum()
        long_return = (tradable.loc[tradable["weight"] > 0, "weight"] * tradable.loc[tradable["weight"] > 0, return_column]).sum()
        short_return = (tradable.loc[tradable["weight"] < 0, "weight"] * tradable.loc[tradable["weight"] < 0, return_column]).sum()
        borrow_cost = tradable.loc[tradable["weight"] < 0, "weight"].abs().sum() * annual_borrow_bps / 10_000 / periods_per_year
        cost = trading_cost + borrow_cost
        rows.append({"date": date, "gross_return": gross, "net_return": gross - cost, "long_leg_return": long_return, "short_leg_return": short_return, "turnover": turnover, "trading_cost": trading_cost, "borrow_cost": borrow_cost, "transaction_cost": cost, "n_positions": len(tradable)})
        previous = tradable[["ticker", "weight"]].copy()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
