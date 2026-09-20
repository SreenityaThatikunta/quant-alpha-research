"""Cost and capacity-assumption sensitivity for completed OOS portfolios."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.backtest import run_weekly_backtest
from src.metrics import performance_metrics


def execution_cost_capacity_sensitivity(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    cost_model: str = "flat",
    cost_bps_values: Iterable[float] = (10.0,),
    portfolio_notionals: Iterable[float] = (1_000_000.0,),
    half_spread_bps: float = 2.0,
    impact_coefficient: float = 0.1,
    annual_borrow_bps: float = 0.0,
) -> pd.DataFrame:
    """Revalue fixed OOS weights under declared cost/capacity assumptions.

    This is a sensitivity analysis, not a new model-selection exercise: the
    forecasts and weights are held fixed. Notional affects the liquidity model
    through participation; it is included for flat costs too so reports make
    that invariance visible rather than implying a capacity estimate. For the
    liquidity model, ``cost_bps_values`` represents half-spread assumptions;
    for the flat model it represents all-in turnover cost assumptions.
    """
    costs = list(cost_bps_values)
    notionals = list(portfolio_notionals)
    if not costs or not notionals or any(value < 0 for value in costs) or any(value <= 0 for value in notionals):
        raise ValueError("Cost values must be non-negative and notionals must be positive.")
    rows: list[dict[str, float | str]] = []
    for cost_bps in costs:
        for notional in notionals:
            backtest = run_weekly_backtest(
                weights,
                realized_returns,
                transaction_cost_bps=cost_bps,
                cost_model=cost_model,
                half_spread_bps=cost_bps if cost_model == "liquidity" else half_spread_bps,
                impact_coefficient=impact_coefficient,
                portfolio_notional=notional,
                annual_borrow_bps=annual_borrow_bps,
            )
            metrics = performance_metrics(backtest["net_return"]) if not backtest.empty else pd.Series(dtype=float)
            rows.append({
                "cost_model": cost_model,
                "cost_assumption_bps": cost_bps,
                "cost_assumption_type": "half_spread" if cost_model == "liquidity" else "flat_turnover_cost",
                "portfolio_notional": notional,
                "mean_turnover": backtest["turnover"].mean() if not backtest.empty else float("nan"),
                "mean_trading_cost": backtest["trading_cost"].mean() if not backtest.empty else float("nan"),
                **metrics.to_dict(),
            })
    return pd.DataFrame(rows)
