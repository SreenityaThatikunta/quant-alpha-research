"""Walk-forward, out-of-sample model fitting utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(model_name: str, random_state: int = 7):
    """Create a small, deliberately conservative model configuration."""
    if model_name == "ridge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("Install xgboost to use model_name='xgboost'.") from error
        return XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror",
            random_state=random_state, n_jobs=1,
        )
    raise ValueError("model_name must be 'ridge' or 'xgboost'")


def walk_forward_predictions(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "target_residual_return_5d",
    model_name: str = "ridge",
    train_days: int = 504,
    test_days: int = 21,
    embargo_days: int = 5,
    expanding: bool = True,
) -> pd.DataFrame:
    """Fit only on historical data and return concatenated OOS predictions.

    The embargo separates the last label-bearing train date from each test
    block, preventing overlapping forward labels from leaking across folds.
    Preprocessing is fitted anew within every fold through the model pipeline.
    """
    features = list(feature_columns)
    required = {"date", "ticker", target_column, *features}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    if min(train_days, test_days) < 1 or embargo_days < 0:
        raise ValueError("train_days/test_days must be positive and embargo_days non-negative")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    dates = np.array(sorted(work["date"].dropna().unique()))
    outputs: list[pd.DataFrame] = []
    test_start = train_days + embargo_days
    fold = 0
    while test_start < len(dates):
        test_end = min(test_start + test_days, len(dates))
        train_end = test_start - embargo_days
        train_start = 0 if expanding else max(0, train_end - train_days)
        train_dates, test_dates = dates[train_start:train_end], dates[test_start:test_end]
        train = work.loc[work["date"].isin(train_dates)].dropna(subset=[target_column])
        test = work.loc[work["date"].isin(test_dates)]
        if len(train) and len(test):
            model = make_model(model_name)
            model.fit(train[features], train[target_column])
            out = test[["date", "ticker", target_column]].copy()
            out["prediction"] = model.predict(test[features])
            out["fold"] = fold
            outputs.append(out)
        test_start = test_end
        fold += 1
    if not outputs:
        return pd.DataFrame(columns=["date", "ticker", target_column, "prediction", "fold"])
    return pd.concat(outputs, ignore_index=True)


def ablation_predictions(frame: pd.DataFrame, feature_groups: dict[str, list[str]], **walk_forward_kwargs) -> pd.DataFrame:
    """Run a full model and leave-one-feature-family-out OOS ablations."""
    all_features = [feature for group in feature_groups.values() for feature in group]
    experiments = {"full": all_features}
    experiments.update({f"without_{name}": [f for f in all_features if f not in group] for name, group in feature_groups.items()})
    results = []
    for name, columns in experiments.items():
        output = walk_forward_predictions(frame, columns, **walk_forward_kwargs)
        output["experiment"] = name
        results.append(output)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
