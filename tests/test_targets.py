import numpy as np
import pandas as pd

from src.targets import add_residual_return_target


def test_target_is_strictly_forward_and_final_rows_are_missing():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    benchmark = pd.DataFrame({"date": dates, "open": np.arange(100, 108), "close": np.arange(100, 108)})
    panel = pd.DataFrame({"date": dates, "ticker": "ABC", "open": np.arange(50, 58), "close": np.arange(50, 58)})
    result = add_residual_return_target(panel, benchmark, horizon_days=2, beta_lookback_days=2)
    expected_stock_forward = panel.loc[2, "close"] / panel.loc[1, "open"] - 1
    assert np.isclose(result.loc[0, "stock_forward_return"], expected_stock_forward)
    assert result["target_residual_return_5d"].tail(2).isna().all()
