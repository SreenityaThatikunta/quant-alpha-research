"""Data acquisition, validation, and point-in-time universe construction."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PANEL_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


def download_price_history(
    tickers: Iterable[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Download adjusted daily OHLCV from Yahoo Finance into a normalized panel.

    This is intentionally an acquisition convenience, not a point-in-time
    constituent database. Store the returned data under ``data/raw`` to make a
    research run reproducible.
    """
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if history.empty:
            continue
        history = history.rename(columns=str.lower).reset_index()
        history["date"] = pd.to_datetime(history["Date" if "Date" in history else "Datetime"]).dt.tz_localize(None).dt.normalize()
        history["ticker"] = ticker.upper()
        frames.append(history.loc[:, list(PANEL_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    return validate_panel(pd.concat(frames, ignore_index=True))


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize a daily OHLCV panel without filling gaps."""
    missing = set(PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")
    clean = panel.loc[:, PANEL_COLUMNS].copy()
    clean["date"] = pd.to_datetime(clean["date"]).dt.tz_localize(None).dt.normalize()
    clean["ticker"] = clean["ticker"].astype(str).str.upper()
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    if clean.duplicated(["date", "ticker"]).any():
        raise ValueError("Panel contains duplicate (date, ticker) rows.")
    if (clean["close"] <= 0).any() or (clean["volume"] < 0).any():
        raise ValueError("Close must be positive and volume must be non-negative.")
    return clean


def construct_universe(
    panel: pd.DataFrame,
    min_price: float = 5.0,
    min_median_dollar_volume: float = 5_000_000,
    lookback_days: int = 20,
    require_point_in_time_metadata: bool = False,
    membership_column: str = "in_universe",
    metadata_available_date_column: str = "metadata_available_date",
) -> pd.DataFrame:
    """Add point-in-time eligibility using only data available before each date.

    Dollar volume is based on close times volume. The rolling median and price
    are each shifted one trading day, preventing the signal-date close from
    determining same-date eligibility.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least one")
    if require_point_in_time_metadata:
        validate_point_in_time_metadata(panel, membership_column, metadata_available_date_column)
    result = validate_panel(panel)
    # Preserve non-price metadata such as point-in-time sector labels; it is
    # required later for portfolio constraints.
    metadata_columns = [column for column in panel.columns if column not in PANEL_COLUMNS]
    if metadata_columns:
        metadata = panel.loc[:, ["date", "ticker", *metadata_columns]].copy()
        metadata["date"] = pd.to_datetime(metadata["date"]).dt.tz_localize(None).dt.normalize()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        if metadata.duplicated(["date", "ticker"]).any():
            raise ValueError("Metadata contains duplicate (date, ticker) rows.")
        result = result.merge(metadata, on=["date", "ticker"], how="left", validate="one_to_one")
    result["dollar_volume"] = result["close"] * result["volume"]
    grouped = result.groupby("ticker", group_keys=False)
    result["median_dollar_volume"] = grouped["dollar_volume"].transform(
        lambda series: series.rolling(lookback_days, min_periods=lookback_days).median().shift(1)
    )
    result["eligible"] = (
        result.groupby("ticker")["close"].shift(1).ge(min_price)
        & result["median_dollar_volume"].ge(min_median_dollar_volume)
    )
    if require_point_in_time_metadata:
        result["eligible"] &= result[membership_column].astype(bool)
    return result


def validate_point_in_time_metadata(
    panel: pd.DataFrame,
    membership_column: str = "in_universe",
    metadata_available_date_column: str = "metadata_available_date",
) -> None:
    """Reject universe metadata that could have become known after a signal date.

    This check does not turn a vendor feed into survivorship-free data by
    itself; it makes the source's stated availability explicit and prevents an
    accidental current-constituent snapshot from being treated as point-in-time
    history.  Delisting returns and corporate-action methodology remain source
    level requirements documented in ``reports/data_provenance.md``.
    """
    required = {"date", "ticker", membership_column, metadata_available_date_column}
    if missing := required.difference(panel.columns):
        raise ValueError(f"Point-in-time panel missing: {sorted(missing)}")
    metadata_dates = pd.to_datetime(panel[metadata_available_date_column], errors="coerce")
    signal_dates = pd.to_datetime(panel["date"], errors="coerce")
    if metadata_dates.isna().any() or signal_dates.isna().any():
        raise ValueError("Point-in-time metadata dates must be valid timestamps.")
    if (metadata_dates > signal_dates).any():
        raise ValueError("Universe metadata is available after its signal date.")
    if panel[membership_column].isna().any():
        raise ValueError("Point-in-time universe membership must be explicitly recorded.")
