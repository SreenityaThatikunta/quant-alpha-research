"""Prediction and portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statistics import NormalDist


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


def deflated_sharpe_ratio(returns: pd.Series, trials: int = 1) -> float:
    """Estimate the probability that observed Sharpe exceeds selection bias.

    This is the Deflated Sharpe Ratio approximation from Bailey and López de
    Prado. ``trials`` is the number of materially distinct strategies examined,
    not the number of parameter combinations logged after the fact.
    """
    values = returns.dropna()
    if len(values) < 4 or trials < 1:
        return float("nan")
    volatility = values.std(ddof=1)
    if volatility == 0:
        return float("nan")
    # The non-normality correction is defined on the single-period Sharpe.
    # Annualizing here would distort the skew/kurtosis denominator.
    observed = values.mean() / volatility
    skewness = values.skew()
    excess_kurtosis = values.kurt()
    normal = NormalDist()
    if trials == 1:
        expected_maximum = 0.0
    else:
        euler_gamma = 0.5772156649
        expected_maximum = (
            (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
            + euler_gamma * normal.inv_cdf(1 - 1 / (trials * np.e))
        )
    denominator_squared = 1 - skewness * observed + (excess_kurtosis / 4) * observed**2
    if denominator_squared <= 0:
        return float("nan")
    statistic = (observed - expected_maximum) * np.sqrt(len(values) - 1) / np.sqrt(denominator_squared)
    return normal.cdf(statistic)
