import numpy as np
import pandas as pd

from src.signal_research import evaluate_signal_library


def test_signal_library_evaluation_returns_a_summary_and_backtest():
    dates = pd.date_range("2024-01-01", periods=25, freq="W-FRI")
    tickers = [f"T{index:03d}" for index in range(120)]
    frame = pd.DataFrame({"date": np.repeat(dates, len(tickers)), "ticker": np.tile(tickers, len(dates))})
    cross_section = np.tile(np.linspace(-1, 1, len(tickers)), len(dates))
    frame["signal"] = cross_section
    frame["sector"] = np.tile(["A", "B"], len(frame) // 2)
    frame["beta"] = 1 + cross_section * 0.1
    frame["stock_forward_return"] = cross_section * 0.01
    frame["target_residual_return_5d"] = cross_section * 0.01
    summary, backtests = evaluate_signal_library(frame, ["signal"], max_weight=0.5)
    assert summary.loc[0, "status"] == "evaluated"
    assert "signal" in backtests
    assert not backtests["signal"].empty
