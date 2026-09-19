"""Leakage-safe technical and cross-sectional feature engineering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = (
    "return_1d", "return_5d", "return_20d", "return_60d", "return_120d",
    "skip_momentum_20d", "distance_ma_20d", "volatility_20d",
    "downside_volatility_20d", "atr_proxy_20d", "relative_volume_20d",
    "volume_zscore_20d", "dollar_volume_20d",
)


def add_features(panel: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    """Create daily features using each security's history through signal date.

    Raw price features use the signal-date close. This is valid only when the
    strategy generates signals after the close and executes the following day.
    Missing histories remain missing; no forward fill is performed.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if missing := required.difference(panel.columns):
        raise ValueError(f"Panel missing: {sorted(missing)}")
    frame = panel.copy().sort_values(["ticker", "date"])
    grouped = frame.groupby("ticker", group_keys=False)
    close = frame["close"]
    frame["return_1d"] = grouped["close"].pct_change()
    for window in (5, 20, 60, 120):
        frame[f"return_{window}d"] = grouped["close"].pct_change(window)
    frame["skip_momentum_20d"] = grouped["close"].shift(5).div(grouped["close"].shift(25)).sub(1)
    moving_average = grouped["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    frame["distance_ma_20d"] = close.div(moving_average).sub(1)
    frame["volatility_20d"] = grouped["return_1d"].transform(lambda x: x.rolling(20, min_periods=20).std())
    frame["downside_volatility_20d"] = grouped["return_1d"].transform(
        lambda x: x.where(x < 0).rolling(20, min_periods=10).std()
    )
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()], axis=1
    ).max(axis=1)
    frame["atr_proxy_20d"] = true_range.groupby(frame["ticker"]).transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    ).div(close)
    volume_mean = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    volume_std = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=20).std())
    frame["relative_volume_20d"] = frame["volume"].div(volume_mean)
    frame["volume_zscore_20d"] = frame["volume"].sub(volume_mean).div(volume_std.where(volume_std.ne(0)))
    frame["dollar_volume_20d"] = (close * frame["volume"]).groupby(frame["ticker"]).transform(
        lambda x: x.rolling(20, min_periods=20).median()
    )
    return add_cross_sectional_transforms(frame, RAW_FEATURE_COLUMNS, sector_column)


def add_cross_sectional_transforms(
    frame: pd.DataFrame, feature_columns: Iterable[str], sector_column: str = "sector", winsor_limit: float = 0.01
) -> pd.DataFrame:
    """Winsorize by date, then add date-wide and optional sector-relative ranks."""
    if not 0 <= winsor_limit < 0.5:
        raise ValueError("winsor_limit must be in [0, 0.5)")
    result = frame.copy()
    for feature in feature_columns:
        if feature not in result:
            raise ValueError(f"Missing feature: {feature}")
        by_date = result.groupby("date")[feature]
        lower = by_date.transform(lambda x: x.quantile(winsor_limit))
        upper = by_date.transform(lambda x: x.quantile(1 - winsor_limit))
        clipped = result[feature].clip(lower=lower, upper=upper)
        result[f"{feature}_rank"] = clipped.groupby(result["date"]).rank(pct=True)
        mean = clipped.groupby(result["date"]).transform("mean")
        std = clipped.groupby(result["date"]).transform("std")
        result[f"{feature}_zscore"] = clipped.sub(mean).div(std.where(std.ne(0)))
        if sector_column in result:
            result[f"{feature}_sector_rank"] = clipped.groupby([result["date"], result[sector_column]]).rank(pct=True)
    return result


def feature_quality_diagnostics(
    frame: pd.DataFrame, feature_columns: Iterable[str], target_column: str = "target_residual_return_5d"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return coverage, correlations, and daily Pearson/rank IC diagnostics."""
    columns = list(feature_columns)
    coverage = pd.DataFrame({
        "coverage": frame[columns].notna().mean(),
        "missing_count": frame[columns].isna().sum(),
    })
    correlations = frame[columns].corr()
    records: list[dict[str, object]] = []
    if target_column in frame:
        for date, daily in frame.groupby("date"):
            for feature in columns:
                subset = daily[[feature, target_column]].dropna()
                if len(subset) >= 5:
                    records.append({"date": date, "feature": feature, "pearson_ic": subset.corr(method="pearson").iloc[0, 1], "rank_ic": subset.corr(method="spearman").iloc[0, 1]})
    return coverage, correlations, pd.DataFrame(records)
