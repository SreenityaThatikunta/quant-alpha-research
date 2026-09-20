"""Download point-in-time SEC XBRL fundamentals for a bounded ticker list."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.fundamentals import SEC_COMPANY_FACTS_URL, SEC_COMPANY_TICKERS_URL, company_facts_observations, fetch_sec_json, ticker_cik_map, write_fundamentals


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC XBRL facts with conservative availability timestamps.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tickers", help="Comma-separated US tickers, for example AAPL,MSFT,NVDA")
    source.add_argument("--tickers-file", type=Path, help="CSV containing a ticker column, such as the price-universe snapshot")
    parser.add_argument("--user-agent", required=True, help="Identifying contact string required for responsible SEC access")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path")
    parser.add_argument("--availability-lag-days", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=8.0, help="Maximum SEC request rate; keep at or below 10")
    parser.add_argument("--offset", type=int, default=0, help="Zero-based ticker offset for resumable batches")
    parser.add_argument("--limit", type=int, default=None, help="Maximum tickers to fetch after offset")
    arguments = parser.parse_args()
    if not 0 < arguments.requests_per_second <= 10:
        raise ValueError("requests-per-second must be in (0, 10] to respect SEC fair-access guidance")
    if arguments.offset < 0 or (arguments.limit is not None and arguments.limit < 1):
        raise ValueError("offset must be non-negative and limit must be positive when provided")
    mapping = ticker_cik_map(fetch_sec_json(SEC_COMPANY_TICKERS_URL, arguments.user_agent))
    tickers = arguments.tickers.split(",") if arguments.tickers else pd.read_csv(arguments.tickers_file)["ticker"].tolist()
    tickers = tickers[arguments.offset : arguments.offset + arguments.limit if arguments.limit else None]
    frames = []
    unresolved = []
    for ticker in (str(item).strip().upper() for item in tickers if str(item).strip()):
        if ticker not in mapping:
            unresolved.append(ticker)
            continue
        cik = mapping[ticker]
        observations = company_facts_observations(fetch_sec_json(SEC_COMPANY_FACTS_URL.format(cik=cik), arguments.user_agent), cik, arguments.availability_lag_days)
        observations["ticker"] = ticker
        frames.append(observations)
        time.sleep(1 / arguments.requests_per_second)
    write_fundamentals(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), arguments.output)
    print(f"Wrote SEC facts for {len(frames)} tickers.")
    if unresolved:
        print(f"Skipped {len(unresolved)} unresolved tickers: {','.join(unresolved)}")


if __name__ == "__main__":
    main()
