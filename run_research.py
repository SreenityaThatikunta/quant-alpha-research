"""Run the complete research workflow from versioned input files.

Example:
    .venv/bin/python run_research.py --panel data/raw/prices.parquet \
        --benchmark data/raw/spy.parquet --output data/processed/run_001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import run_weekly_backtest, weekly_rebalance_dates
from src.data import construct_universe
from src.diagnostics import average_cross_sectional_signal_correlation, signal_library_summary
from src.experiments import build_run_manifest, write_run_manifest
from src.features import RAW_FEATURE_COLUMNS, add_features
from src.metrics import deflated_sharpe_ratio, performance_metrics, prediction_metrics
from src.models import walk_forward_predictions
from src.portfolio import construct_optimized_portfolios, construct_portfolio, exposure_diagnostics
from src.signals import TECHNICAL_SIGNAL_COLUMNS, build_technical_signal_library, neutralize_signal
from src.targets import add_residual_return_target


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Inputs must be CSV or Parquet files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe equity alpha research run.")
    parser.add_argument("--panel", type=Path, required=True, help="OHLCV panel with date/ticker columns")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark daily close data")
    parser.add_argument("--output", type=Path, required=True, help="Directory for reproducible outputs")
    parser.add_argument("--model", choices=("ridge", "xgboost"), default="ridge")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--cost-model", choices=("flat", "liquidity"), default="flat")
    parser.add_argument("--half-spread-bps", type=float, default=2.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.1)
    parser.add_argument("--portfolio-notional", type=float, default=1_000_000.0)
    parser.add_argument("--annual-borrow-bps", type=float, default=0.0)
    parser.add_argument("--portfolio-construction", choices=("projection", "optimizer"), default="projection")
    parser.add_argument("--max-turnover", type=float, default=None)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        parameters={
            "model": arguments.model,
            "train_days": arguments.train_days,
            "test_days": arguments.test_days,
            "cost_bps": arguments.cost_bps,
            "cost_model": arguments.cost_model,
            "half_spread_bps": arguments.half_spread_bps,
            "impact_coefficient": arguments.impact_coefficient,
            "portfolio_notional": arguments.portfolio_notional,
            "annual_borrow_bps": arguments.annual_borrow_bps,
            "portfolio_construction": arguments.portfolio_construction,
            "max_turnover": arguments.max_turnover,
            "portfolio_quantile": 0.10,
            "rebalance_frequency": "weekly",
            "label_horizon_days": 5,
            "embargo_days": 5,
        },
        input_paths={"panel": arguments.panel, "benchmark": arguments.benchmark},
        repository=Path(__file__).resolve().parent,
    )
    raw_panel = read_table(arguments.panel)
    if "sector" not in raw_panel:
        raise ValueError("The panel must include a point-in-time 'sector' classification for sector-neutral portfolios.")
    panel = construct_universe(raw_panel)
    labeled = add_residual_return_target(panel, read_table(arguments.benchmark))
    featured = build_technical_signal_library(add_features(labeled))
    for signal in TECHNICAL_SIGNAL_COLUMNS:
        featured[f"{signal}_neutral"] = neutralize_signal(featured, signal)
    feature_columns = [f"{feature}_zscore" for feature in RAW_FEATURE_COLUMNS]
    signal_columns = [*TECHNICAL_SIGNAL_COLUMNS, *[f"{signal}_neutral" for signal in TECHNICAL_SIGNAL_COLUMNS]]
    signal_library_summary(featured, signal_columns).to_csv(arguments.output / "signal_library_summary.csv", index=False)
    average_cross_sectional_signal_correlation(featured, signal_columns).to_csv(arguments.output / "signal_correlation.csv")
    eligible = featured.loc[featured["eligible"]].copy()
    predictions = walk_forward_predictions(
        eligible, feature_columns, model_name=arguments.model,
        train_days=arguments.train_days, test_days=arguments.test_days,
    )
    context = eligible[[column for column in ("date", "ticker", "sector", "beta", "stock_forward_return", "dollar_volume_20d", "volatility_20d") if column in eligible]].drop_duplicates(["date", "ticker"])
    predictions = predictions.merge(context, on=["date", "ticker"], how="left", validate="one_to_one")
    predictions = predictions.loc[predictions["date"].isin(weekly_rebalance_dates(predictions["date"]))]
    if arguments.portfolio_construction == "optimizer":
        weights, optimizer_diagnostics = construct_optimized_portfolios(predictions, max_turnover=arguments.max_turnover)
        optimizer_diagnostics.to_csv(arguments.output / "optimizer_diagnostics.csv", index=False)
    else:
        weights = construct_portfolio(predictions)
    if weights.empty:
        raise RuntimeError("No eligible neutral portfolios were formed. Check universe size, sector/beta coverage, and max-weight settings.")
    backtest = run_weekly_backtest(
        weights, predictions, transaction_cost_bps=arguments.cost_bps,
        cost_model=arguments.cost_model, half_spread_bps=arguments.half_spread_bps,
        impact_coefficient=arguments.impact_coefficient,
        portfolio_notional=arguments.portfolio_notional,
        annual_borrow_bps=arguments.annual_borrow_bps,
    )
    daily_ic, ic_summary = prediction_metrics(predictions)
    performance = performance_metrics(backtest["net_return"]) if not backtest.empty else pd.Series(dtype=float)
    if not backtest.empty:
        performance.loc["deflated_sharpe_ratio_one_trial"] = deflated_sharpe_ratio(backtest["net_return"])

    labeled.to_parquet(arguments.output / "labeled_panel.parquet", index=False)
    predictions.to_parquet(arguments.output / "oos_predictions.parquet", index=False)
    weights.to_parquet(arguments.output / "portfolio_weights.parquet", index=False)
    backtest.to_csv(arguments.output / "backtest.csv", index=False)
    daily_ic.to_csv(arguments.output / "daily_ic.csv", index=False)
    exposure_diagnostics(weights).to_csv(arguments.output / "exposures.csv", index=False)
    pd.concat([ic_summary.rename("value").to_frame().assign(metric=lambda x: x.index), performance.rename("value").to_frame().assign(metric=lambda x: x.index)]).reset_index(drop=True).to_csv(arguments.output / "summary.csv", index=False)
    write_run_manifest(manifest, arguments.output / "run_manifest.json")


if __name__ == "__main__":
    main()
