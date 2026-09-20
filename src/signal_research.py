"""Evaluate individually motivated signals under the same portfolio rules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.backtest import run_weekly_backtest, weekly_rebalance_dates
from src.metrics import deflated_sharpe_ratio, performance_metrics, prediction_metrics
from src.portfolio import construct_portfolio


def evaluate_signal_library(
    frame: pd.DataFrame,
    signal_columns: Iterable[str],
    transaction_cost_bps: float = 10.0,
    cost_model: str = "flat",
    max_weight: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest each signal independently using only its OOS holding-period return.

    The evaluation never chooses a signal based on a combined backtest. It
    returns a compact promotion table plus individual backtests so researchers
    can inspect turnover, costs, and constraint feasibility per candidate.
    """
    signals = list(signal_columns)
    required = {"date", "ticker", "sector", "beta", "stock_forward_return", "target_residual_return_5d", *signals}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    context_columns = [
        column for column in ("date", "ticker", "sector", "beta", "stock_forward_return", "target_residual_return_5d", "dollar_volume_20d", "volatility_20d")
        if column in frame.columns
    ]
    weekly = frame.loc[frame["date"].isin(weekly_rebalance_dates(frame["date"])), context_columns + signals].copy()
    summaries = []
    backtests: dict[str, pd.DataFrame] = {}
    for signal in signals:
        predictions = weekly[context_columns + [signal]].rename(columns={signal: "prediction"})
        predictions = predictions.dropna(subset=["prediction"])
        weights = construct_portfolio(predictions, max_weight=max_weight)
        if weights.empty:
            summaries.append({"signal": signal, "status": "no_feasible_portfolios"})
            continue
        backtest = run_weekly_backtest(
            weights, predictions, transaction_cost_bps=transaction_cost_bps, cost_model=cost_model
        )
        backtests[signal] = backtest
        _, prediction_summary = prediction_metrics(predictions)
        performance = performance_metrics(backtest["net_return"])
        summaries.append({
            "signal": signal, "status": "evaluated", "mean_turnover": backtest["turnover"].mean(),
            "deflated_sharpe_probability": deflated_sharpe_ratio(backtest["net_return"], trials=len(signals)),
            **prediction_summary.to_dict(), **performance.to_dict(),
        })
    return pd.DataFrame(summaries), backtests
