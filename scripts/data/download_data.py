"""Create a reproducible public-data input snapshot for the research runner.

This deliberately uses *current* S&P 500 constituents, so it is not a
point-in-time universe and must be interpreted as survivorship-biased.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def download_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV in batches and return normalized long-form data."""
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(tickers), 100):
        batch = tickers[offset : offset + 100]
        downloaded = yf.download(batch, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False, threads=True)
        for ticker in batch:
            if ticker not in downloaded.columns.get_level_values(0):
                continue
            prices = downloaded[ticker].dropna(how="all")
            if prices.empty:
                continue
            prices = prices.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
            prices["ticker"] = ticker
            frames.append(prices.loc[:, ["date", "ticker", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def current_constituents() -> pd.DataFrame:
    """Fetch the current constituent table with a standard descriptive header."""
    response = requests.get(WIKIPEDIA_SP500_URL, headers={"User-Agent": "quant-alpha-research/0.1 educational-research"}, timeout=30)
    response.raise_for_status()
    constituents = pd.read_html(StringIO(response.text))[0].loc[:, ["Symbol", "GICS Sector"]]
    constituents.columns = ["ticker", "sector"]
    return constituents


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public-data baseline snapshot.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-09-19")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, default=300, help="Number of current constituents; 300 is the documented baseline")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    constituents = current_constituents()
    constituents["ticker"] = constituents["ticker"].str.replace(".", "-", regex=False)
    constituents = constituents.head(arguments.limit)
    panel = download_ohlcv(constituents["ticker"].tolist(), arguments.start, arguments.end)
    panel = panel.merge(constituents, on="ticker", how="left", validate="many_to_one")
    benchmark = download_ohlcv(["SPY"], arguments.start, arguments.end).drop(columns="ticker")
    panel.to_parquet(arguments.output / "sp500_current_constituents_prices.parquet", index=False)
    benchmark.to_parquet(arguments.output / "spy_benchmark.parquet", index=False)
    constituents.to_csv(arguments.output / "sp500_current_constituents_sectors.csv", index=False)
    print(f"Saved {len(panel):,} price rows for {panel['ticker'].nunique()} tickers.")


if __name__ == "__main__":
    main()
