"""Prediction and portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_metrics(predictions: pd.DataFrame, prediction_column: str = "prediction", target_column: str = "target_residual_return_5d") -> tuple[pd.DataFrame, pd.Series]:
    """Calculate date-wise ICs and an out-of-sample aggregate summary."""
    required = {"date", prediction_column, target_column}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Predictions missing: {sorted(missing)}")
    rows = []
    for date, daily in predictions.groupby("date"):
        sample = daily[[prediction_column, target_column]].dropna()
        if len(sample) >= 5:
            rows.append({"date": date, "pearson_ic": sample.corr(method="pearson").iloc[0, 1], "rank_ic": sample.corr(method="spearman").iloc[0, 1], "directional_accuracy": (np.sign(sample[prediction_column]) == np.sign(sample[target_column])).mean(), "n": len(sample)})
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.Series(dtype=float)
    summary = pd.Series({
        "mean_pearson_ic": daily["pearson_ic"].mean(),
        "mean_rank_ic": daily["rank_ic"].mean(),
        "rank_ic_ir": daily["rank_ic"].mean() / daily["rank_ic"].std(ddof=1) if daily["rank_ic"].std(ddof=1) else np.nan,
        "directional_accuracy": daily["directional_accuracy"].mean(),
        "observations": daily["n"].sum(),
    })
    return daily, summary


def performance_metrics(returns: pd.Series, periods_per_year: int = 52) -> pd.Series:
    """Calculate conventional annualized performance metrics from periodic returns."""
    returns = returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative.div(cumulative.cummax()).sub(1)
    annual_return = cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1
    annual_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return pd.Series({
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": annual_return / annual_volatility if annual_volatility else np.nan,
        "maximum_drawdown": drawdown.min(),
        "periods": len(returns),
    })
