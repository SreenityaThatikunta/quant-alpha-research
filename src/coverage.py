"""Coverage audits for public-source point-in-time universe studies."""

from __future__ import annotations

import pandas as pd


def universe_price_coverage_audit(universe_history: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Report every historically eligible ticker missing from a price panel.

    A public-source price download can silently omit delisted symbols. This
    audit makes that missingness an output artifact instead of shrinking the
    historical universe until a backtest happens to run.
    """
    history_required = {"ticker", "effective_date", "in_universe"}
    price_required = {"ticker", "date"}
    if missing := history_required.difference(universe_history.columns):
        raise ValueError(f"Universe history missing: {sorted(missing)}")
    if missing := price_required.difference(prices.columns):
        raise ValueError(f"Prices missing: {sorted(missing)}")
    history = universe_history.copy()
    history["ticker"] = history["ticker"].astype(str).str.upper()
    history["effective_date"] = pd.to_datetime(history["effective_date"])
    prices = prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    members = history.loc[history["in_universe"].astype(bool)]
    expected = members.groupby("ticker").agg(
        first_membership_date=("effective_date", "min"),
        membership_add_events=("effective_date", "size"),
    )
    observed = prices.groupby("ticker").agg(
        price_observations=("date", "size"), first_price_date=("date", "min"), last_price_date=("date", "max")
    )
    audit = expected.join(observed, how="left")
    audit["price_observations"] = audit["price_observations"].fillna(0).astype(int)
    audit["missing_all_prices"] = audit["price_observations"].eq(0)
    audit["price_starts_after_membership"] = audit["first_price_date"].gt(audit["first_membership_date"])
    return audit.reset_index().sort_values(["missing_all_prices", "ticker"], ascending=[False, True]).reset_index(drop=True)
