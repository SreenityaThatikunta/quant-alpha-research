"""Free public-source point-in-time S&P index universe ingestion."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def fetch_pitindex_snapshots(start: str, end: str, index: str = "sp500") -> tuple[pd.DataFrame, dict[str, object]]:
    """Load sparse point-in-time constituent snapshots from ``pitindex``.

    ``pitindex`` reconstructs membership from public sources. It improves on a
    current-constituent snapshot, but it is not a vendor security master: its
    upstream ticker history and event dates have documented limitations. The
    caller must persist the returned provenance and audit missing price history.
    """
    try:
        import pitindex
    except ImportError as error:
        raise ImportError(
            "Install the pinned pitindex dependency to download the free historical universe."
        ) from error
    snapshots = pitindex.get_constituents_history(start, end, index=index)
    if snapshots.empty:
        raise ValueError("pitindex returned no constituent snapshots for the requested range.")
    return snapshots, dict(pitindex.info())


def snapshots_to_change_log(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse snapshots to an as-of membership/classification change log.

    The first snapshot becomes the starting universe. Later snapshots emit only
    additions, removals, or classification changes. Availability is set to the
    snapshot's ``as_of`` date, which is conservative: the log never makes a
    reconstructed public record available before the date it was observed.
    """
    required = {"as_of", "ticker"}
    if missing := required.difference(snapshots.columns):
        raise ValueError(f"Snapshots missing: {sorted(missing)}")
    work = snapshots.copy()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce").dt.normalize()
    work["ticker"] = work["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    if work["as_of"].isna().any() or work["ticker"].eq("").any():
        raise ValueError("Snapshots require valid as-of dates and tickers.")
    if work.duplicated(["as_of", "ticker"]).any():
        raise ValueError("Snapshots contain duplicate as-of/ticker rows.")
    metadata_columns = [column for column in ("name", "cik", "gics_sector", "gics_sub_industry") if column in work]

    def unchanged(left: object, right: object) -> bool:
        """Treat a missing field in consecutive public snapshots as unchanged."""
        return bool(pd.isna(left) and pd.isna(right)) or left == right

    events: list[dict[str, object]] = []
    previous: dict[str, Mapping[str, object]] = {}
    for as_of, snapshot in work.sort_values(["as_of", "ticker"]).groupby("as_of", sort=True):
        current = {
            str(row["ticker"]): row.drop(labels=["as_of"]).to_dict()
            for _, row in snapshot.iterrows()
        }
        for ticker in sorted(set(current).union(previous)):
            now, before = current.get(ticker), previous.get(ticker)
            if now is None:
                # Retain the last known classification on removals so the event
                # itself remains auditable, while the security becomes ineligible.
                event = dict(before or {})
                event["in_universe"] = False
            elif before is None or any(not unchanged(now.get(column), before.get(column)) for column in metadata_columns):
                event = dict(now)
                event["in_universe"] = True
            else:
                continue
            event["ticker"] = ticker
            event["effective_date"] = as_of
            event["metadata_available_date"] = as_of
            if "gics_sector" in event:
                event["sector"] = event.pop("gics_sector")
            events.append(event)
        previous = current
    history = pd.DataFrame(events)
    if history.empty:
        raise ValueError("Snapshots produced no membership events.")
    ordered = [
        "ticker", "effective_date", "metadata_available_date", "in_universe", "sector",
        "cik", "name", "gics_sub_industry",
    ]
    return history.reindex(columns=[column for column in ordered if column in history.columns]).sort_values(
        ["ticker", "effective_date"]
    ).reset_index(drop=True)
