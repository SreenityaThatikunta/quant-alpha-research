"""Create a free-source point-in-time S&P 500 proxy input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from download_data import download_ohlcv
from src.coverage import universe_price_coverage_audit
from src.pit_universe import fetch_pitindex_snapshots, snapshots_to_change_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free public-source PIT S&P 500 proxy inputs.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=25, help="Tickers per resumable Yahoo request")
    parser.add_argument("--restart-prices", action="store_true", help="Discard saved price batches before downloading")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    snapshots, provenance = fetch_pitindex_snapshots(arguments.start, arguments.end)
    history = snapshots_to_change_log(snapshots)
    history_path = arguments.output / "sp500_pit_universe.parquet"
    history.to_parquet(history_path, index=False)
    provenance.update({"requested_start": arguments.start, "requested_end": arguments.end, "index": "sp500"})
    history_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tickers = sorted(history.loc[history["in_universe"], "ticker"].unique())
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    batches = arguments.output / "sp500_pit_price_batches"
    batches.mkdir(parents=True, exist_ok=True)
    if arguments.restart_prices:
        for path in batches.glob("*.parquet"):
            path.unlink()
    parts: list[pd.DataFrame] = []
    for batch_number, offset in enumerate(range(0, len(tickers), arguments.batch_size)):
        path = batches / f"batch_{batch_number:03d}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
            continue
        batch = download_ohlcv(tickers[offset: offset + arguments.batch_size], arguments.start, arguments.end)
        # Persist each completed request. A transient provider failure can then
        # be resumed without re-requesting the entire historical universe.
        batch.to_parquet(path, index=False)
        parts.append(batch)
        print(f"Completed price batch {batch_number + 1} of {(len(tickers) - 1) // arguments.batch_size + 1}.")
    prices = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    prices.to_parquet(arguments.output / "sp500_pit_prices.parquet", index=False)
    benchmark = download_ohlcv(["SPY"], arguments.start, arguments.end).drop(columns="ticker")
    benchmark.to_parquet(arguments.output / "spy_benchmark.parquet", index=False)
    audit = universe_price_coverage_audit(history, prices)
    audit.to_csv(arguments.output / "sp500_pit_price_coverage.csv", index=False)
    missing = int(audit["missing_all_prices"].sum())
    print(f"Saved {len(history):,} PIT membership events, {len(prices):,} price rows, and a coverage audit ({missing:,} tickers fully missing).")


if __name__ == "__main__":
    main()
