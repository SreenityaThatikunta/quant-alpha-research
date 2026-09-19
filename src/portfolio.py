"""Constrained, market-neutral long-short portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _neutralize(weights: pd.Series, sectors: pd.Series | None, betas: pd.Series | None) -> pd.Series:
    """Project candidate weights onto dollar, sector, and beta-neutral space."""
    constraints = [np.ones(len(weights))]
    if sectors is not None:
        for sector in sorted(sectors.dropna().unique()):
            constraints.append((sectors == sector).astype(float).to_numpy())
    if betas is not None and betas.notna().all():
        constraints.append(betas.to_numpy(dtype=float))
    matrix = np.column_stack(constraints)
    # w - A(A'A)^+A'w is the least-squares projection onto null(A').
    projected = weights.to_numpy(dtype=float) - matrix @ np.linalg.pinv(matrix.T @ matrix) @ matrix.T @ weights.to_numpy(dtype=float)
    return pd.Series(projected, index=weights.index)


def construct_portfolio(
    predictions: pd.DataFrame,
    prediction_column: str = "prediction",
    quantile: float = 0.10,
    sector_column: str = "sector",
    beta_column: str = "beta",
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Select top/bottom quantiles and neutralize their equal-weight exposures.

    Dates with too few names or degenerate constraints are skipped. The result
    is rescaled to one unit of gross exposure and includes diagnostics users
    should review before interpreting performance.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    required = {"date", "ticker", prediction_column}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Predictions missing: {sorted(missing)}")
    results = []
    for date, daily in predictions.groupby("date"):
        daily = daily.dropna(subset=[prediction_column]).copy()
        # Neutrality is a hard constraint, not a best-effort diagnostic. A name
        # without a contemporaneous sector or beta estimate cannot be traded.
        required_exposures = [column for column in (sector_column, beta_column) if column in daily]
        daily = daily.dropna(subset=required_exposures)
        names_per_leg = int(np.floor(len(daily) * quantile))
        if names_per_leg < 1:
            continue
        chosen = pd.concat([daily.nlargest(names_per_leg, prediction_column), daily.nsmallest(names_per_leg, prediction_column)]).drop_duplicates("ticker")
        chosen["raw_weight"] = np.where(chosen[prediction_column].rank(method="first", ascending=False) <= names_per_leg, 1.0, -1.0)
        sectors = chosen[sector_column] if sector_column in chosen else None
        betas = chosen[beta_column] if beta_column in chosen else None
        weights = _neutralize(chosen["raw_weight"], sectors, betas)
        gross = weights.abs().sum()
        if gross == 0:
            continue
        weights = weights / gross
        if weights.abs().max() > max_weight:
            # A proper constrained optimizer is required for exact neutralization with a binding cap.
            # Exclude this date rather than silently breaking neutrality by clipping.
            continue
        chosen["weight"] = weights
        chosen["gross_exposure"] = chosen["weight"].abs().sum()
        chosen["net_exposure"] = chosen["weight"].sum()
        chosen["beta_exposure"] = (chosen["weight"] * chosen[beta_column]).sum() if beta_column in chosen else np.nan
        results.append(chosen.drop(columns="raw_weight"))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def exposure_diagnostics(weights: pd.DataFrame, sector_column: str = "sector", beta_column: str = "beta") -> pd.DataFrame:
    """Summarize gross, net, beta, and sector exposures per rebalance."""
    rows = []
    for date, daily in weights.groupby("date"):
        row = {"date": date, "gross": daily["weight"].abs().sum(), "net": daily["weight"].sum()}
        if beta_column in daily:
            row["beta"] = (daily["weight"] * daily[beta_column]).sum()
        if sector_column in daily:
            row["max_abs_sector_exposure"] = daily.groupby(sector_column)["weight"].sum().abs().max()
        rows.append(row)
    return pd.DataFrame(rows)
