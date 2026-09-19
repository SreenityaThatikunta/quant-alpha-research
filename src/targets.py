"""Forward labels for next-week residual-equity-return research."""

from __future__ import annotations

import pandas as pd


def add_residual_return_target(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon_days: int = 5,
    beta_lookback_days: int = 60,
) -> pd.DataFrame:
    """Attach a strictly forward residual-return target to a stock panel.

    ``beta`` is estimated using returns ending on the signal date. Both stock
    and benchmark forward returns begin on the next trading day and end after
    ``horizon_days`` sessions.
    """
    required_panel = {"date", "ticker", "open", "close"}
    required_benchmark = {"date", "open", "close"}
    if missing := required_panel.difference(panel.columns):
        raise ValueError(f"Panel missing: {sorted(missing)}")
    if missing := required_benchmark.difference(benchmark.columns):
        raise ValueError(f"Benchmark missing: {sorted(missing)}")
    if horizon_days < 1 or beta_lookback_days < 2:
        raise ValueError("horizon_days >= 1 and beta_lookback_days >= 2 are required")

    stocks = panel.copy()
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["ticker", "date"])
    market = benchmark.loc[:, ["date", "open", "close"]].copy().rename(columns={"open": "benchmark_open", "close": "benchmark_close"})
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date")
    market["benchmark_return_1d"] = market["benchmark_close"].pct_change()
    # Signals use the completed signal-date close. Enter at the next session's
    # open and mark the five-session holding period at its final close.
    market["benchmark_forward_return"] = market["benchmark_close"].shift(-horizon_days).div(market["benchmark_open"].shift(-1)).sub(1)

    merged = stocks.merge(market, on="date", how="left", validate="many_to_one")
    merged["stock_return_1d"] = merged.groupby("ticker")["close"].pct_change()
    merged["stock_forward_return"] = merged.groupby("ticker")["close"].shift(-horizon_days).div(merged.groupby("ticker")["open"].shift(-1)).sub(1)
    # Express rolling covariance through rolling moments. ``transform`` keeps
    # the original panel index, avoiding accidental cross-ticker alignment.
    grouped = merged.groupby("ticker")
    rolling_mean = lambda series: grouped[series].transform(
        lambda values: values.rolling(beta_lookback_days, min_periods=beta_lookback_days).mean()
    )
    merged["_xy"] = merged["stock_return_1d"] * merged["benchmark_return_1d"]
    merged["_yy"] = merged["benchmark_return_1d"] ** 2
    covariance = rolling_mean("_xy") - rolling_mean("stock_return_1d") * rolling_mean("benchmark_return_1d")
    variance = rolling_mean("_yy") - rolling_mean("benchmark_return_1d") ** 2
    merged["beta"] = covariance.div(variance.where(variance.ne(0)))
    merged["target_residual_return_5d"] = merged["stock_forward_return"] - merged["beta"] * merged["benchmark_forward_return"]
    return merged.drop(columns=["_xy", "_yy"])
