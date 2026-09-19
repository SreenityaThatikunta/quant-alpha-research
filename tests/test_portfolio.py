import numpy as np
import pandas as pd

from src.portfolio import construct_portfolio, exposure_diagnostics


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
