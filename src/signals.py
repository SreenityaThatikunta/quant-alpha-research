"""Transparent, economically distinct cross-sectional alpha candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


TECHNICAL_SIGNAL_COLUMNS = (
    "intermediate_momentum",
    "short_horizon_reversal",
    "low_risk",
    "liquidity_quality",
)


def _cross_sectional_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date")[column].rank(pct=True)


def build_technical_signal_library(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a small, interpretable set of signals before any ML combination.

    The signals intentionally use different economic intuitions: trend,
    short-term reversal, preference for lower realized risk, and liquidity.
    They are candidates, not claims of predictive power; every one must pass
    its own OOS IC, decay, cost, and correlation checks before promotion.
    """
    required = {
        "date", "return_1d_zscore", "return_5d_zscore", "return_60d_zscore",
        "volatility_20d_zscore", "downside_volatility_20d_zscore", "atr_proxy_20d_zscore",
        "dollar_volume_20d_zscore", "relative_volume_20d_zscore",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    result = frame.copy()
    result["intermediate_momentum"] = result["return_60d_zscore"] - result["return_5d_zscore"]
    result["short_horizon_reversal"] = -result["return_1d_zscore"]
    result["low_risk"] = -result[["volatility_20d_zscore", "downside_volatility_20d_zscore", "atr_proxy_20d_zscore"]].mean(axis=1)
    result["liquidity_quality"] = result[["dollar_volume_20d_zscore", "relative_volume_20d_zscore"]].mean(axis=1)
    for column in TECHNICAL_SIGNAL_COLUMNS:
        result[f"{column}_rank"] = _cross_sectional_rank(result, column)
    return result


def neutralize_signal(
    frame: pd.DataFrame,
    signal_column: str,
    factor_columns: Iterable[str] = ("beta",),
    sector_column: str = "sector",
) -> pd.Series:
    """Residualize a signal daily against available style and sector exposures."""
    factors = list(factor_columns)
    required = {"date", signal_column, *factors}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, daily in frame.groupby("date"):
        columns = [signal_column, *factors]
        if sector_column in daily:
            columns.append(sector_column)
        sample = daily[columns].dropna()
        if len(sample) <= len(factors) + 1:
            continue
        design = [np.ones(len(sample))]
        design.extend(sample[factor].to_numpy(dtype=float) for factor in factors)
        if sector_column in sample:
            # Drop one dummy to avoid a singular intercept-plus-sector design.
            dummies = pd.get_dummies(sample[sector_column], dtype=float)
            design.extend(dummies.iloc[:, 1:].to_numpy().T)
        matrix = np.column_stack(design)
        coefficients, *_ = np.linalg.lstsq(matrix, sample[signal_column].to_numpy(dtype=float), rcond=None)
        residual.loc[sample.index] = sample[signal_column].to_numpy(dtype=float) - matrix @ coefficients
    return residual


def combine_signals(frame: pd.DataFrame, weights: Mapping[str, float], output_column: str = "combined_signal") -> pd.DataFrame:
    """Create an explicit weighted signal composite; weights are never inferred in-sample."""
    if not weights:
        raise ValueError("At least one signal weight is required")
    if missing := set(weights).difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    total_weight = sum(abs(weight) for weight in weights.values())
    if total_weight == 0:
        raise ValueError("Signal weights cannot all be zero")
    result = frame.copy()
    result[output_column] = sum(result[signal] * weight for signal, weight in weights.items()) / total_weight
    result[f"{output_column}_rank"] = _cross_sectional_rank(result, output_column)
    return result
