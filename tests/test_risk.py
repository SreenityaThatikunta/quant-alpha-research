import io
import zipfile

import numpy as np
import pandas as pd

from src.risk import FAMA_FRENCH_FACTOR_COLUMNS, aggregate_factors_for_holding_periods, factor_attribution, parse_fama_french_daily_zip


def test_parser_converts_fama_french_percentages_to_decimals():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("factors.csv", "Header\nDATE,Mkt-RF,SMB,HML,RMW,CMA,RF\n20240102,1.0,2.0,3.0,4.0,5.0,0.1\n\nFooter")
    factors = parse_fama_french_daily_zip(buffer.getvalue())
    assert factors.loc[0, "Mkt-RF"] == 0.01
    assert factors.loc[0, "RF"] == 0.001


def test_factor_attribution_recovers_market_beta():
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    market = np.linspace(-0.01, 0.01, 10)
    factors = pd.DataFrame({"date": dates, "Mkt-RF": market, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "RF": 0.0})
    returns = pd.DataFrame({"date": dates, "net_return": 0.001 + 2 * market})
    attribution = factor_attribution(returns, factors)
    assert np.isclose(attribution["alpha_per_period"], 0.001)
    assert np.isclose(attribution["beta_Mkt-RF"], 2.0)


def test_factor_aggregation_matches_the_portfolio_holding_window():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    factors = pd.DataFrame({"date": dates, **{column: 0.01 for column in FAMA_FRENCH_FACTOR_COLUMNS}})
    aggregated = aggregate_factors_for_holding_periods(factors, pd.Series([dates[0], dates[6]]), horizon_days=2)
    assert len(aggregated) == 1
    assert np.isclose(aggregated.loc[0, "Mkt-RF"], 1.01**2 - 1)
