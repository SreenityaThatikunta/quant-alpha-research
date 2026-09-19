import numpy as np
import pandas as pd

from src.models import nested_walk_forward_predictions, walk_forward_predictions


def test_walk_forward_outputs_only_future_blocks():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame({"date": np.repeat(dates, 5), "ticker": np.tile(list("ABCDE"), 30)})
    frame["feature"] = np.arange(len(frame))
    frame["target"] = frame["feature"] * 0.001
    predictions = walk_forward_predictions(frame, ["feature"], target_column="target", train_days=10, test_days=5, embargo_days=2)
    assert predictions["date"].min() == dates[12]
    assert predictions["fold"].nunique() > 1


def test_nested_walk_forward_selects_inside_outer_training_window():
    dates = pd.date_range("2020-01-01", periods=45, freq="B")
    frame = pd.DataFrame({"date": np.repeat(dates, 6), "ticker": np.tile(list("ABCDEF"), 45)})
    ticker_effect = pd.Categorical(frame["ticker"]).codes
    frame["feature"] = ticker_effect + np.repeat(np.arange(45), 6) * 0.01
    frame["target"] = ticker_effect * 0.01
    predictions, selections = nested_walk_forward_predictions(
        frame,
        ["feature"],
        candidates={
            "low_regularization": {"model_name": "ridge", "model_parameters": {"alpha": 0.1}},
            "high_regularization": {"model_name": "ridge", "model_parameters": {"alpha": 100.0}},
        },
        target_column="target",
        train_days=15,
        test_days=5,
        embargo_days=2,
        inner_validation_days=5,
    )
    assert predictions["date"].min() == dates[17]
    assert set(predictions["selected_candidate"]).issubset({"low_regularization", "high_regularization"})
    assert len(selections) == predictions["fold"].nunique()
    assert selections["test_start"].min() == dates[17]
