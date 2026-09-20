"""Download a free public-source point-in-time S&P 500 membership change log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pit_universe import fetch_pitindex_snapshots, snapshots_to_change_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free point-in-time index membership from pitindex.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--index", choices=("sp500", "sp400", "sp600", "sp1500"), default="sp500")
    parser.add_argument("--output", type=Path, default=Path("data/raw/sp500_pit_universe.parquet"))
    arguments = parser.parse_args()
    snapshots, provenance = fetch_pitindex_snapshots(arguments.start, arguments.end, arguments.index)
    history = snapshots_to_change_log(snapshots)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(arguments.output, index=False)
    provenance.update({"requested_start": arguments.start, "requested_end": arguments.end, "index": arguments.index})
    arguments.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(history):,} membership events for {history['ticker'].nunique():,} securities to {arguments.output}.")


if __name__ == "__main__":
    main()
