"""Out-of-sample signal-library and stability diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def signal_library_summary(
    frame: pd.DataFrame,
    signal_columns: Iterable[str],
    target_column: str = "target_residual_return_5d",
    minimum_cross_section: int = 5,
) -> pd.DataFrame:
    """Summarize coverage and date-wise rank IC for a candidate signal library.

    IC is calculated separately within each date and then averaged.  This
    prevents dates with a larger universe from dominating a signal's score.
    """
    signals = list(signal_columns)
    required = {"date", target_column, *signals}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for signal in signals:
        daily_ics = []
        for _, daily in frame[["date", signal, target_column]].groupby("date"):
            sample = daily[[signal, target_column]].dropna()
            if len(sample) >= minimum_cross_section:
                daily_ics.append(sample.corr(method="spearman").iloc[0, 1])
        ic_series = pd.Series(daily_ics, dtype=float)
        records.append({
            "signal": signal,
            "coverage": frame[signal].notna().mean(),
            "rank_ic": ic_series.mean(),
            "rank_ic_ir": ic_series.mean() / ic_series.std(ddof=1) if len(ic_series) > 1 and ic_series.std(ddof=1) else np.nan,
            "ic_days": len(ic_series),
        })
    return pd.DataFrame(records).sort_values("rank_ic", ascending=False, na_position="last").reset_index(drop=True)


def average_cross_sectional_signal_correlation(
    frame: pd.DataFrame, signal_columns: Iterable[str], method: str = "spearman"
) -> pd.DataFrame:
    """Average within-date correlations, avoiding correlations induced by time trends."""
    signals = list(signal_columns)
    if missing := {"date", *signals}.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    matrices = []
    for _, daily in frame.groupby("date"):
        matrix = daily[signals].corr(method=method)
        if not matrix.empty:
            matrices.append(matrix)
    if not matrices:
        return pd.DataFrame(np.nan, index=signals, columns=signals)
    aligned = np.stack([matrix.reindex(index=signals, columns=signals).to_numpy() for matrix in matrices])
    with np.errstate(invalid="ignore"):
        values = np.nanmean(aligned, axis=0)
    return pd.DataFrame(values, index=signals, columns=signals)


def signal_decay(
    frame: pd.DataFrame,
    signal_columns: Iterable[str],
    horizon_targets: Mapping[int, str],
    minimum_cross_section: int = 5,
) -> pd.DataFrame:
    """Measure rank-IC decay against separately constructed forward horizons."""
    signals = list(signal_columns)
    required = {"date", *signals, *horizon_targets.values()}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for horizon, target in sorted(horizon_targets.items()):
        for signal in signals:
            daily_ics = []
            for _, daily in frame[["date", signal, target]].groupby("date"):
                sample = daily[[signal, target]].dropna()
                if len(sample) >= minimum_cross_section:
                    daily_ics.append(sample.corr(method="spearman").iloc[0, 1])
            rows.append({"signal": signal, "horizon_days": horizon, "rank_ic": pd.Series(daily_ics, dtype=float).mean(), "ic_days": len(daily_ics)})
    return pd.DataFrame(rows)


def regime_performance(returns: pd.DataFrame, regime_column: str, return_column: str = "net_return") -> pd.DataFrame:
    """Return count, mean, volatility, and Sharpe by a pre-defined regime label."""
    required = {regime_column, return_column}
    if missing := required.difference(returns.columns):
        raise ValueError(f"Returns missing: {sorted(missing)}")
    grouped = returns.dropna(subset=[regime_column, return_column]).groupby(regime_column)[return_column]
    result = grouped.agg(observations="count", mean_return="mean", volatility="std").reset_index()
    result["sharpe_per_period"] = result["mean_return"].div(result["volatility"].replace(0, np.nan))
    return result
