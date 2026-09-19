import numpy as np
import pandas as pd

from src.signals import TECHNICAL_SIGNAL_COLUMNS, build_technical_signal_library, combine_signals, neutralize_signal


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    frame = pd.DataFrame({"date": np.repeat(dates, 8), "ticker": np.tile(list("ABCDEFGH"), 2)})
    values = np.tile(np.arange(8, dtype=float), 2)
    for column in (
        "return_1d_zscore", "return_5d_zscore", "return_60d_zscore", "volatility_20d_zscore",
        "downside_volatility_20d_zscore", "atr_proxy_20d_zscore", "dollar_volume_20d_zscore", "relative_volume_20d_zscore",
    ):
        frame[column] = values
    frame["beta"] = values
    frame["sector"] = np.tile(["A", "B"], 8)
    return frame


def test_build_and_combine_signal_library():
    signals = build_technical_signal_library(_frame())
    assert set(TECHNICAL_SIGNAL_COLUMNS).issubset(signals.columns)
    assert signals.loc[0, "short_horizon_reversal"] == -signals.loc[0, "return_1d_zscore"]
    composite = combine_signals(signals, {"short_horizon_reversal": 1, "low_risk": 1})
    assert composite["combined_signal_rank"].between(0, 1).all()


def test_signal_neutralization_removes_linear_beta_exposure():
    frame = _frame()
    frame["signal"] = 2 * frame["beta"]
    residual = neutralize_signal(frame, "signal", factor_columns=("beta",), sector_column="missing_sector")
    assert np.nanmax(np.abs(residual)) < 1e-10
