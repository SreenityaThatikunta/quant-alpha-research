import numpy as np
import pandas as pd

from src.diagnostics import average_cross_sectional_signal_correlation, signal_decay, signal_library_summary
from src.metrics import deflated_sharpe_ratio


def _signal_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    frame = pd.DataFrame({"date": np.repeat(dates, 6), "ticker": np.tile(list("ABCDEF"), 4)})
    values = np.tile(np.arange(6, dtype=float), 4)
    frame["signal_a"] = values
    frame["signal_b"] = values[::-1]
    frame["target"] = values
    frame["target_10d"] = values
    return frame


def test_signal_library_summary_and_correlation_are_cross_sectional():
    frame = _signal_frame()
    summary = signal_library_summary(frame, ["signal_a", "signal_b"], target_column="target")
    correlation = average_cross_sectional_signal_correlation(frame, ["signal_a", "signal_b"])
    assert summary.loc[0, "signal"] == "signal_a"
    assert summary.loc[0, "rank_ic"] == 1.0
    assert correlation.loc["signal_a", "signal_b"] == -1.0


def test_signal_decay_and_deflated_sharpe_are_reported():
    frame = _signal_frame()
    decay = signal_decay(frame, ["signal_a"], {10: "target_10d"})
    returns = pd.Series([0.01, -0.002, 0.008, 0.005, -0.003, 0.007])
    assert decay.loc[0, "rank_ic"] == 1.0
    assert 0.0 <= deflated_sharpe_ratio(returns, trials=2) <= 1.0
