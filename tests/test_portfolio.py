import numpy as np
import pandas as pd

from src.portfolio import construct_optimized_portfolios, construct_portfolio, exposure_diagnostics


def test_portfolio_neutralizes_dollar_sector_and_beta_exposure():
    tickers = [f"T{i:02d}" for i in range(40)]
    predictions = pd.DataFrame({
        "date": pd.Timestamp("2024-01-05"),
        "ticker": tickers,
        "prediction": np.linspace(-1, 1, 40),
        "sector": ["A", "B"] * 20,
        "beta": np.linspace(0.6, 1.4, 40),
    })
    weights = construct_portfolio(predictions, quantile=0.2, max_weight=0.5)
    exposure = exposure_diagnostics(weights).iloc[0]
    assert abs(exposure["net"]) < 1e-10
    assert abs(exposure["beta"]) < 1e-10
    assert exposure["max_abs_sector_exposure"] < 1e-10


def test_portfolio_excludes_missing_required_exposures():
    predictions = pd.DataFrame({
        "date": pd.Timestamp("2024-01-05"), "ticker": [f"T{i}" for i in range(10)],
        "prediction": np.arange(10), "sector": ["A", "B"] * 5,
        "beta": [1.0] * 9 + [np.nan],
    })
    weights = construct_portfolio(predictions, quantile=0.2, max_weight=0.5)
    assert "T9" not in set(weights["ticker"])


def test_optimizer_enforces_neutrality_and_records_feasibility():
    tickers = [f"T{i:02d}" for i in range(40)]
    predictions = pd.DataFrame({
        "date": pd.Timestamp("2024-01-05"), "ticker": tickers,
        "prediction": np.linspace(-1, 1, 40), "sector": ["A", "B"] * 20,
        "beta": np.linspace(0.6, 1.4, 40), "size": np.linspace(-1, 1, 40) ** 2,
    })
    weights, diagnostics = construct_optimized_portfolios(
        predictions, quantile=0.2, max_weight=0.5, factor_columns=("beta", "size"), risk_aversion=0.1,
    )
    assert diagnostics.loc[0, "feasible"]
    assert abs(weights["weight"].sum()) < 1e-6
    assert abs((weights["weight"] * weights["beta"]).sum()) < 1e-6
    assert abs((weights["weight"] * weights["size"]).sum()) < 1e-6
