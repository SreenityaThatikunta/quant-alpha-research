"""Run an apples-to-apples public-data study of the transparent signal library."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import construct_universe
from src.diagnostics import (
    average_cross_sectional_signal_correlation,
    grouped_signal_stability,
    probability_of_backtest_overfitting,
    signal_library_summary,
)
from src.experiments import build_run_manifest, write_run_manifest
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
    parser.add_argument("--signals", default=None, help="Comma-separated base signals; defaults to the complete library")
    parser.add_argument("--include-neutralized", action="store_true", help="Also evaluate beta/sector-neutralized versions")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    # A signal screen is an experiment, not an ad-hoc spreadsheet export.  The
    # manifest captures precisely which inputs and assumptions generated every
    # candidate decision in the output directory.
    manifest = build_run_manifest(
        parameters={
            "cost_bps": arguments.cost_bps,
            "signals": arguments.signals,
            "include_neutralized": arguments.include_neutralized,
            "rebalance_frequency": "weekly",
            "label_horizon_days": 5,
            "execution": "next_session_open",
            "portfolio_constraints": "dollar_sector_beta_neutral",
        },
        input_paths={"panel": arguments.panel, "benchmark": arguments.benchmark},
        repository=Path(__file__).resolve().parent,
    )

    labeled = add_residual_return_target(construct_universe(read_table(arguments.panel)), read_table(arguments.benchmark))
    featured = build_technical_signal_library(add_features(labeled))
    selected = arguments.signals.split(",") if arguments.signals else list(TECHNICAL_SIGNAL_COLUMNS)
    unknown = set(selected).difference(TECHNICAL_SIGNAL_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown base signals: {sorted(unknown)}")
    signals = list(selected)
    if arguments.include_neutralized:
        for signal in selected:
            neutral = f"{signal}_neutral"
            featured[neutral] = neutralize_signal(featured, signal)
            signals.append(neutral)
    summary, backtests = evaluate_signal_library(featured, signals, transaction_cost_bps=arguments.cost_bps)
    summary.to_csv(arguments.output / "signal_backtest_summary.csv", index=False)
    signal_library_summary(featured, signals).to_csv(arguments.output / "signal_ic_summary.csv", index=False)
    average_cross_sectional_signal_correlation(featured, signals).to_csv(arguments.output / "signal_correlation.csv")
    featured["calendar_year"] = pd.to_datetime(featured["date"]).dt.year
    grouped_signal_stability(featured, signals, "calendar_year").to_csv(
        arguments.output / "signal_stability_by_year.csv", index=False
    )
    grouped_signal_stability(featured, signals, "sector").to_csv(
        arguments.output / "signal_stability_by_sector.csv", index=False
    )
    for signal, backtest in backtests.items():
        backtest.to_csv(arguments.output / f"backtest_{signal}.csv", index=False)
    if backtests:
        aligned_returns = pd.concat(
            {signal: backtest.set_index("date")["net_return"] for signal, backtest in backtests.items()}, axis=1
        ).dropna()
        partitions = min(8, len(aligned_returns))
        partitions -= partitions % 2
        if aligned_returns.shape[1] >= 2 and partitions >= 2:
            pd.DataFrame([{
                "pre_specified_candidates": aligned_returns.shape[1],
                "aligned_return_periods": len(aligned_returns),
                "partitions": partitions,
                "probability_of_backtest_overfitting": probability_of_backtest_overfitting(
                    aligned_returns, partitions=partitions
                ),
            }]).to_csv(arguments.output / "selection_bias_diagnostics.csv", index=False)
    write_run_manifest(manifest, arguments.output / "run_manifest.json")


if __name__ == "__main__":
    main()
