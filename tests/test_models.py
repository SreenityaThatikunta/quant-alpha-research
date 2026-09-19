import numpy as np
import pandas as pd

from src.models import walk_forward_predictions


def test_walk_forward_outputs_only_future_blocks():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame({"date": np.repeat(dates, 5), "ticker": np.tile(list("ABCDE"), 30)})
    frame["feature"] = np.arange(len(frame))
    frame["target"] = frame["feature"] * 0.001
    predictions = walk_forward_predictions(frame, ["feature"], target_column="target", train_days=10, test_days=5, embargo_days=2)
    assert predictions["date"].min() == dates[12]
    assert predictions["fold"].nunique() > 1
