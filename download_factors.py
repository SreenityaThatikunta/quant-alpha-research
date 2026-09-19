"""Download the official daily Fama-French five-factor data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.risk import download_fama_french_daily, write_factor_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily Fama-French five-factor returns.")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path")
    arguments = parser.parse_args()
    write_factor_data(download_fama_french_daily(), arguments.output)


if __name__ == "__main__":
    main()
