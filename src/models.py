"""Walk-forward, out-of-sample model fitting utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(
    model_name: str, random_state: int = 7, model_parameters: Mapping[str, float | int] | None = None
):
    """Create a small, deliberately conservative model configuration."""
    parameters = dict(model_parameters or {})
    if model_name == "ridge":
        alpha = parameters.pop("alpha", 10.0)
        if parameters:
            raise ValueError(f"Unsupported Ridge parameters: {sorted(parameters)}")
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("Install xgboost to use model_name='xgboost'.") from error
        defaults = dict(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror",
            random_state=random_state, n_jobs=1,
        )
        unknown = set(parameters).difference(defaults)
        if unknown:
            raise ValueError(f"Unsupported XGBoost parameters: {sorted(unknown)}")
        return XGBRegressor(**(defaults | parameters))
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
    model_parameters: Mapping[str, float | int] | None = None,
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
            model = make_model(model_name, model_parameters=model_parameters)
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


def _mean_rank_ic(predictions: pd.DataFrame, target_column: str) -> float:
    """Score a prediction panel without pooling cross-sections across dates."""
    values = []
    for _, daily in predictions.groupby("date"):
        sample = daily[["prediction", target_column]].dropna()
        if len(sample) >= 5:
            values.append(sample.corr(method="spearman").iloc[0, 1])
    return float(np.mean(values)) if values else float("-inf")


def nested_walk_forward_predictions(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    candidates: Mapping[str, Mapping[str, object]],
    target_column: str = "target_residual_return_5d",
    train_days: int = 504,
    test_days: int = 21,
    embargo_days: int = 5,
    inner_validation_days: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select models inside each outer training window, then predict untouched OOS blocks.

    Each inner validation block is separated from its fitting history by the
    same label-overlap embargo used for the outer test.  The returned selection
    table is deliberately separate from final predictions so a report can show
    exactly which candidate was selected at each origin.

    ``candidates`` maps a stable experiment name to a mapping containing
    ``model_name`` and optional ``model_parameters``.  No candidate is scored
    on its outer test block before selection.
    """
    if not candidates:
        raise ValueError("At least one model candidate is required")
    if min(train_days, test_days, inner_validation_days) < 1 or embargo_days < 0:
        raise ValueError("window lengths must be positive and embargo_days non-negative")
    features = list(feature_columns)
    required = {"date", "ticker", target_column, *features}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Frame missing: {sorted(missing)}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    dates = np.array(sorted(work["date"].dropna().unique()))
    prediction_outputs: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    outer_start = train_days + embargo_days
    fold = 0
    while outer_start < len(dates):
        outer_end = min(outer_start + test_days, len(dates))
        outer_train_end = outer_start - embargo_days
        outer_train_dates = dates[:outer_train_end]
        inner_start = max(inner_validation_days + embargo_days, len(outer_train_dates) - inner_validation_days)
        candidate_scores: dict[str, float] = {}
        for candidate_name, candidate in candidates.items():
            model_name = str(candidate.get("model_name", "ridge"))
            parameters = candidate.get("model_parameters")
            if parameters is not None and not isinstance(parameters, Mapping):
                raise ValueError("model_parameters must be a mapping")
            inner_predictions: list[pd.DataFrame] = []
            cursor = inner_start
            while cursor < len(outer_train_dates):
                validation_end = min(cursor + inner_validation_days, len(outer_train_dates))
                fit_dates = outer_train_dates[: cursor - embargo_days]
                validation_dates = outer_train_dates[cursor:validation_end]
                train = work.loc[work["date"].isin(fit_dates)].dropna(subset=[target_column])
                validation = work.loc[work["date"].isin(validation_dates)]
                if len(train) and len(validation):
                    model = make_model(model_name, model_parameters=parameters)
                    model.fit(train[features], train[target_column])
                    scored = validation[["date", "ticker", target_column]].copy()
                    scored["prediction"] = model.predict(validation[features])
                    inner_predictions.append(scored)
                cursor = validation_end
            candidate_scores[candidate_name] = _mean_rank_ic(
                pd.concat(inner_predictions, ignore_index=True) if inner_predictions else pd.DataFrame(), target_column
            ) if inner_predictions else float("-inf")
        selected_name = max(candidate_scores, key=candidate_scores.get)
        selected = candidates[selected_name]
        train = work.loc[work["date"].isin(outer_train_dates)].dropna(subset=[target_column])
        test_dates = dates[outer_start:outer_end]
        test = work.loc[work["date"].isin(test_dates)]
        model = make_model(
            str(selected.get("model_name", "ridge")),
            model_parameters=selected.get("model_parameters"),  # type: ignore[arg-type]
        )
        model.fit(train[features], train[target_column])
        output = test[["date", "ticker", target_column]].copy()
        output["prediction"] = model.predict(test[features])
        output["fold"] = fold
        output["selected_candidate"] = selected_name
        prediction_outputs.append(output)
        selection_rows.append({
            "fold": fold,
            "test_start": dates[outer_start],
            "test_end": dates[outer_end - 1],
            "selected_candidate": selected_name,
            **{f"inner_rank_ic_{name}": score for name, score in candidate_scores.items()},
        })
        outer_start = outer_end
        fold += 1
    predictions = pd.concat(prediction_outputs, ignore_index=True) if prediction_outputs else pd.DataFrame()
    return predictions, pd.DataFrame(selection_rows)


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
