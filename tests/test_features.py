import numpy as np
import pandas as pd

from src.features import add_features


def test_features_require_history_and_do_not_backfill():
    dates = pd.date_range("2023-01-01", periods=130, freq="B")
    panel = pd.DataFrame({"date": dates, "ticker": "ABC", "open": np.arange(130) + 10, "high": np.arange(130) + 11, "low": np.arange(130) + 9, "close": np.arange(130) + 10, "volume": 1_000_000})
    result = add_features(panel)
    assert result.loc[0, "return_120d"] != result.loc[0, "return_120d"]  # NaN
    assert result.loc[120, "return_120d"] == 130 / 10 - 1
