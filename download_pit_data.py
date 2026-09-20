"""Create a free-source point-in-time S&P 500 proxy input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_data import download_ohlcv
from src.coverage import universe_price_coverage_audit
from src.pit_universe import fetch_pitindex_snapshots, snapshots_to_change_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free public-source PIT S&P 500 proxy inputs.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    snapshots, provenance = fetch_pitindex_snapshots(arguments.start, arguments.end)
    history = snapshots_to_change_log(snapshots)
    history_path = arguments.output / "sp500_pit_universe.parquet"
    history.to_parquet(history_path, index=False)
    provenance.update({"requested_start": arguments.start, "requested_end": arguments.end, "index": "sp500"})
    history_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tickers = sorted(history.loc[history["in_universe"], "ticker"].unique())
    prices = download_ohlcv(tickers, arguments.start, arguments.end)
    prices.to_parquet(arguments.output / "sp500_pit_prices.parquet", index=False)
    benchmark = download_ohlcv(["SPY"], arguments.start, arguments.end).drop(columns="ticker")
    benchmark.to_parquet(arguments.output / "spy_benchmark.parquet", index=False)
    audit = universe_price_coverage_audit(history, prices)
    audit.to_csv(arguments.output / "sp500_pit_price_coverage.csv", index=False)
    missing = int(audit["missing_all_prices"].sum())
    print(f"Saved {len(history):,} PIT membership events, {len(prices):,} price rows, and a coverage audit ({missing:,} tickers fully missing).")


if __name__ == "__main__":
    main()
