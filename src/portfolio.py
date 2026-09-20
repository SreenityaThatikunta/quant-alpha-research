"""Constrained, market-neutral long-short portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _neutralize(weights: pd.Series, sectors: pd.Series | None, betas: pd.Series | None) -> pd.Series:
    """Project candidate weights onto dollar, sector, and beta-neutral space."""
    constraints = [np.ones(len(weights))]
    if sectors is not None:
        for sector in sorted(sectors.dropna().unique()):
            constraints.append((sectors == sector).astype(float).to_numpy())
    if betas is not None and betas.notna().all():
        constraints.append(betas.to_numpy(dtype=float))
    matrix = np.column_stack(constraints)
    # w - A(A'A)^+A'w is the least-squares projection onto null(A').
    projected = weights.to_numpy(dtype=float) - matrix @ np.linalg.pinv(matrix.T @ matrix) @ matrix.T @ weights.to_numpy(dtype=float)
    return pd.Series(projected, index=weights.index)


def construct_portfolio(
    predictions: pd.DataFrame,
    prediction_column: str = "prediction",
    quantile: float = 0.10,
    sector_column: str = "sector",
    beta_column: str = "beta",
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Select top/bottom quantiles and neutralize their equal-weight exposures.

    Dates with too few names or degenerate constraints are skipped. The result
    is rescaled to one unit of gross exposure and includes diagnostics users
    should review before interpreting performance.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    required = {"date", "ticker", prediction_column}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Predictions missing: {sorted(missing)}")
    results = []
    for date, daily in predictions.groupby("date"):
        daily = daily.dropna(subset=[prediction_column]).copy()
        # Neutrality is a hard constraint, not a best-effort diagnostic. A name
        # without a contemporaneous sector or beta estimate cannot be traded.
        required_exposures = [column for column in (sector_column, beta_column) if column in daily]
        daily = daily.dropna(subset=required_exposures)
        names_per_leg = int(np.floor(len(daily) * quantile))
        if names_per_leg < 1:
            continue
        chosen = pd.concat([daily.nlargest(names_per_leg, prediction_column), daily.nsmallest(names_per_leg, prediction_column)]).drop_duplicates("ticker")
        chosen["raw_weight"] = np.where(chosen[prediction_column].rank(method="first", ascending=False) <= names_per_leg, 1.0, -1.0)
        sectors = chosen[sector_column] if sector_column in chosen else None
        betas = chosen[beta_column] if beta_column in chosen else None
        weights = _neutralize(chosen["raw_weight"], sectors, betas)
        gross = weights.abs().sum()
        if gross == 0:
            continue
        weights = weights / gross
        if weights.abs().max() > max_weight:
            # A proper constrained optimizer is required for exact neutralization with a binding cap.
            # Exclude this date rather than silently breaking neutrality by clipping.
            continue
        chosen["weight"] = weights
        chosen["gross_exposure"] = chosen["weight"].abs().sum()
        chosen["net_exposure"] = chosen["weight"].sum()
        chosen["beta_exposure"] = (chosen["weight"] * chosen[beta_column]).sum() if beta_column in chosen else np.nan
        results.append(chosen.drop(columns="raw_weight"))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def exposure_diagnostics(weights: pd.DataFrame, sector_column: str = "sector", beta_column: str = "beta") -> pd.DataFrame:
    """Summarize gross, net, beta, and sector exposures per rebalance."""
    rows = []
    for date, daily in weights.groupby("date"):
        row = {"date": date, "gross": daily["weight"].abs().sum(), "net": daily["weight"].sum()}
        if beta_column in daily:
            row["beta"] = (daily["weight"] * daily[beta_column]).sum()
        if sector_column in daily:
            row["max_abs_sector_exposure"] = daily.groupby(sector_column)["weight"].sum().abs().max()
        rows.append(row)
    return pd.DataFrame(rows)


def _constraint_matrix(
    daily: pd.DataFrame, sector_column: str, factor_columns: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    """Build dollar, sector, and style-factor equality constraints."""
    columns = [np.ones(len(daily))]
    names = ["dollar"]
    if sector_column in daily:
        for sector in sorted(daily[sector_column].dropna().unique()):
            columns.append((daily[sector_column] == sector).astype(float).to_numpy())
            names.append(f"sector:{sector}")
    for factor in factor_columns:
        if factor not in daily:
            raise ValueError(f"Missing factor exposure: {factor}")
        columns.append(daily[factor].to_numpy(dtype=float))
        names.append(f"factor:{factor}")
    return np.column_stack(columns), names


def construct_optimized_portfolios(
    predictions: pd.DataFrame,
    prediction_column: str = "prediction",
    quantile: float = 0.10,
    sector_column: str = "sector",
    factor_columns: tuple[str, ...] = ("beta",),
    max_weight: float = 0.05,
    gross_target: float = 1.0,
    risk_aversion: float = 0.01,
    turnover_penalty: float = 0.0,
    max_turnover: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize market-neutral weights with auditable hard constraints.

    The optimizer maximizes forecasts while penalizing concentration and changes
    from the preceding portfolio.  It enforces dollar, sector, and arbitrary
    factor neutrality; gross exposure and position limits; and an optional
    turnover cap.  Infeasible dates are reported in diagnostics and excluded
    from weights rather than being silently clipped.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError("Install scipy to use constrained portfolio optimization.") from error
    if not 0 < quantile < 0.5 or max_weight <= 0 or gross_target <= 0:
        raise ValueError("quantile, max_weight, and gross_target must be positive and valid")
    if risk_aversion < 0 or turnover_penalty < 0 or (max_turnover is not None and max_turnover <= 0):
        raise ValueError("penalties must be non-negative and max_turnover positive")
    required = {"date", "ticker", prediction_column, *factor_columns}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Predictions missing: {sorted(missing)}")
    portfolios: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    previous = pd.Series(dtype=float)
    for date, original in predictions.groupby("date"):
        daily = original.dropna(subset=[prediction_column, *factor_columns]).copy()
        if sector_column in daily:
            daily = daily.dropna(subset=[sector_column])
        names_per_leg = int(np.floor(len(daily) * quantile))
        if names_per_leg < 1:
            diagnostic_rows.append({"date": date, "feasible": False, "status": "insufficient_names"})
            continue
        daily = pd.concat([
            daily.nlargest(names_per_leg, prediction_column),
            daily.nsmallest(names_per_leg, prediction_column),
        ]).drop_duplicates("ticker").reset_index(drop=True)
        constraints_matrix, constraint_names = _constraint_matrix(daily, sector_column, factor_columns)
        scores = daily[prediction_column].to_numpy(dtype=float)
        score_scale = np.std(scores)
        scores = scores / score_scale if score_scale else scores
        prior = previous.reindex(daily["ticker"], fill_value=0.0).to_numpy(dtype=float)
        raw_initial = np.where(
            daily[prediction_column].rank(method="first", ascending=False) <= names_per_leg, 1.0, -1.0
        )
        # Start on the null space of every active equality constraint.  The
        # previous initializer only neutralized a single factor, which made
        # SLSQP's convergence platform-dependent whenever additional style
        # factors were requested.
        initial = raw_initial - constraints_matrix @ np.linalg.pinv(
            constraints_matrix.T @ constraints_matrix
        ) @ constraints_matrix.T @ raw_initial
        if not np.any(initial):
            diagnostic_rows.append({"date": date, "feasible": False, "status": "degenerate_constraints"})
            continue
        initial = initial / np.abs(initial).sum() * gross_target
        objective = lambda weights: -np.dot(scores, weights) + risk_aversion * np.dot(weights, weights) + turnover_penalty * np.dot(weights - prior, weights - prior)
        scipy_constraints = [
            {"type": "eq", "fun": lambda weights, matrix=constraints_matrix: matrix.T @ weights},
            {"type": "eq", "fun": lambda weights: np.abs(weights).sum() - gross_target},
        ]
        if max_turnover is not None:
            scipy_constraints.append({"type": "ineq", "fun": lambda weights: max_turnover - np.abs(weights - prior).sum() / 2})
        solved = minimize(
            objective, initial, method="SLSQP", bounds=[(-max_weight, max_weight)] * len(daily),
            constraints=scipy_constraints, options={"ftol": 1e-10, "maxiter": 1_000},
        )
        weights = solved.x
        residuals = constraints_matrix.T @ weights
        gross_residual = abs(np.abs(weights).sum() - gross_target)
        turnover = np.abs(weights - prior).sum() / 2
        # SLSQP's success flag varies across SciPy versions for this non-smooth
        # gross-exposure equality.  Feasibility is a mathematical property of
        # the returned weights, not the optimizer's status text. Keep both
        # fields: a constraint-satisfying but non-optimal solution is usable
        # for a baseline comparison, while reports can still flag it for review.
        tolerance = 1e-6
        bound_violation = max(0.0, float(np.abs(weights).max() - max_weight))
        feasible = bool(
            np.isfinite(weights).all()
            and np.max(np.abs(residuals)) < tolerance
            and gross_residual < tolerance
            and bound_violation < tolerance
            and (max_turnover is None or turnover <= max_turnover + tolerance)
        )
        diagnostic_rows.append({
            "date": date, "feasible": feasible, "solver_success": bool(solved.success),
            "status": str(solved.message), "objective": solved.fun,
            "gross": np.abs(weights).sum(), "turnover": turnover,
            "max_abs_constraint_residual": np.max(np.abs(residuals)), "gross_residual": gross_residual,
            "bound_violation": bound_violation,
            **{f"constraint_{name}": residual for name, residual in zip(constraint_names, residuals, strict=True)},
        })
        if not feasible:
            continue
        daily["weight"] = weights
        daily["post_optimization_sign_changed"] = np.sign(weights) != np.sign(scores)
        daily["gross_exposure"] = np.abs(weights).sum()
        daily["net_exposure"] = weights.sum()
        portfolios.append(daily)
        previous = daily.set_index("ticker")["weight"]
    weights_frame = pd.concat(portfolios, ignore_index=True) if portfolios else pd.DataFrame()
    return weights_frame, pd.DataFrame(diagnostic_rows)
