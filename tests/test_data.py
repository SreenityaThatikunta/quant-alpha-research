import pandas as pd
import pytest

from src.data import construct_universe, validate_panel


def test_validate_panel_rejects_duplicate_key():
    panel = pd.DataFrame({"date": ["2024-01-02", "2024-01-02"], "ticker": ["ABC", "ABC"], "open": [10, 10], "high": [11, 11], "low": [9, 9], "close": [10, 10], "volume": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(panel)


def test_universe_lags_price_and_liquidity():
    panel = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "ticker": ["ABC"] * 3, "open": [10] * 3, "high": [10] * 3, "low": [10] * 3, "close": [10, 10, 1], "volume": [1_000_000] * 3})
    result = construct_universe(panel, min_price=5, min_median_dollar_volume=1, lookback_days=1)
    assert result["eligible"].tolist() == [False, True, True]


def test_universe_retains_panel_metadata():
    panel = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["ABC"], "open": [10], "high": [10], "low": [10], "close": [10], "volume": [1_000_000], "sector": ["Technology"]})
    result = construct_universe(panel, lookback_days=1)
    assert result.loc[0, "sector"] == "Technology"
