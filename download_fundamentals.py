"""Download point-in-time SEC XBRL fundamentals for a bounded ticker list."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.fundamentals import SEC_COMPANY_FACTS_URL, SEC_COMPANY_TICKERS_URL, company_facts_observations, fetch_sec_json, ticker_cik_map, write_fundamentals


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC XBRL facts with conservative availability timestamps.")
    parser.add_argument("--tickers", required=True, help="Comma-separated US tickers, for example AAPL,MSFT,NVDA")
    parser.add_argument("--user-agent", required=True, help="Identifying contact string required for responsible SEC access")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path")
    parser.add_argument("--availability-lag-days", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=8.0, help="Maximum SEC request rate; keep at or below 10")
    arguments = parser.parse_args()
    if not 0 < arguments.requests_per_second <= 10:
        raise ValueError("requests-per-second must be in (0, 10] to respect SEC fair-access guidance")
    mapping = ticker_cik_map(fetch_sec_json(SEC_COMPANY_TICKERS_URL, arguments.user_agent))
    frames = []
    unresolved = []
    for ticker in (item.strip().upper() for item in arguments.tickers.split(",") if item.strip()):
        if ticker not in mapping:
            unresolved.append(ticker)
            continue
        cik = mapping[ticker]
        observations = company_facts_observations(fetch_sec_json(SEC_COMPANY_FACTS_URL.format(cik=cik), arguments.user_agent), cik, arguments.availability_lag_days)
        observations["ticker"] = ticker
        frames.append(observations)
        time.sleep(1 / arguments.requests_per_second)
    write_fundamentals(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), arguments.output)
    if unresolved:
        print(f"Skipped {len(unresolved)} unresolved tickers: {','.join(unresolved)}")


if __name__ == "__main__":
    main()
