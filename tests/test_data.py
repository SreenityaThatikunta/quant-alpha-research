import pandas as pd
import pytest

from src.data import attach_point_in_time_universe_metadata, construct_universe, validate_panel, validate_point_in_time_metadata


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


def test_point_in_time_metadata_rejects_future_availability():
    panel = pd.DataFrame({
        "date": ["2024-01-01"], "ticker": ["ABC"], "in_universe": [True],
        "metadata_available_date": ["2024-01-02"],
    })
    with pytest.raises(ValueError, match="after"):
        validate_point_in_time_metadata(panel)


def test_point_in_time_membership_is_applied_to_eligibility():
    panel = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2), "ticker": ["ABC"] * 2,
        "open": [10, 10], "high": [10, 10], "low": [10, 10], "close": [10, 10], "volume": [1_000_000] * 2,
        "in_universe": [True, False], "metadata_available_date": ["2023-12-31"] * 2,
    })
    result = construct_universe(panel, min_median_dollar_volume=1, lookback_days=1, require_point_in_time_metadata=True)
    assert result["eligible"].tolist() == [False, False]


def test_point_in_time_history_is_joined_as_of_each_signal_date():
    panel = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3), "ticker": ["ABC"] * 3,
        "open": [10] * 3, "high": [10] * 3, "low": [10] * 3, "close": [10] * 3, "volume": [1_000_000] * 3,
    })
    history = pd.DataFrame({
        "ticker": ["ABC", "ABC"], "effective_date": ["2023-12-01", "2024-01-03"],
        "metadata_available_date": ["2023-12-01", "2024-01-03"], "in_universe": [True, False],
        "sector": ["Technology", "Technology"],
    })
    joined = attach_point_in_time_universe_metadata(panel, history)
    assert joined["in_universe"].tolist() == [True, True, False]
    assert joined["sector"].tolist() == ["Technology"] * 3
