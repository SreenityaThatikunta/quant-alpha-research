"""Run an apples-to-apples public-data study of the transparent signal library."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import construct_universe
from src.diagnostics import average_cross_sectional_signal_correlation, signal_library_summary
from src.features import add_features
from src.signal_research import evaluate_signal_library
from src.signals import TECHNICAL_SIGNAL_COLUMNS, build_technical_signal_library, neutralize_signal
from src.targets import add_residual_return_target


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transparent market-neutral alpha candidates.")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    labeled = add_residual_return_target(construct_universe(read_table(arguments.panel)), read_table(arguments.benchmark))
    featured = build_technical_signal_library(add_features(labeled))
    signals = [*TECHNICAL_SIGNAL_COLUMNS]
    for signal in TECHNICAL_SIGNAL_COLUMNS:
        neutral = f"{signal}_neutral"
        featured[neutral] = neutralize_signal(featured, signal)
        signals.append(neutral)
    summary, backtests = evaluate_signal_library(featured, signals, transaction_cost_bps=arguments.cost_bps)
    summary.to_csv(arguments.output / "signal_backtest_summary.csv", index=False)
    signal_library_summary(featured, signals).to_csv(arguments.output / "signal_ic_summary.csv", index=False)
    average_cross_sectional_signal_correlation(featured, signals).to_csv(arguments.output / "signal_correlation.csv")
    for signal, backtest in backtests.items():
        backtest.to_csv(arguments.output / f"backtest_{signal}.csv", index=False)


if __name__ == "__main__":
    main()
