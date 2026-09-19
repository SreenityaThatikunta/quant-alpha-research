"""Download point-in-time SEC XBRL fundamentals for a bounded ticker list."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fundamentals import SEC_COMPANY_FACTS_URL, SEC_COMPANY_TICKERS_URL, company_facts_observations, fetch_sec_json, ticker_cik_map, write_fundamentals


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC XBRL facts with conservative availability timestamps.")
    parser.add_argument("--tickers", required=True, help="Comma-separated US tickers, for example AAPL,MSFT,NVDA")
    parser.add_argument("--user-agent", required=True, help="Identifying contact string required for responsible SEC access")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path")
    parser.add_argument("--availability-lag-days", type=int, default=1)
    arguments = parser.parse_args()
    mapping = ticker_cik_map(fetch_sec_json(SEC_COMPANY_TICKERS_URL, arguments.user_agent))
    frames = []
    for ticker in (item.strip().upper() for item in arguments.tickers.split(",") if item.strip()):
        if ticker not in mapping:
            raise ValueError(f"Ticker not found in SEC company mapping: {ticker}")
        cik = mapping[ticker]
        observations = company_facts_observations(fetch_sec_json(SEC_COMPANY_FACTS_URL.format(cik=cik), arguments.user_agent), cik, arguments.availability_lag_days)
        observations["ticker"] = ticker
        frames.append(observations)
    write_fundamentals(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), arguments.output)


if __name__ == "__main__":
    main()
