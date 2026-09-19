import pandas as pd
import numpy as np

from src.backtest import estimated_transaction_cost, run_weekly_backtest


def test_backtest_deducts_turnover_costs():
    date = pd.Timestamp("2024-01-05")
    weights = pd.DataFrame({"date": [date, date], "ticker": ["A", "B"], "weight": [0.5, -0.5]})
    returns = pd.DataFrame({"date": [date, date], "ticker": ["A", "B"], "stock_forward_return": [0.02, -0.02]})
    result = run_weekly_backtest(weights, returns, transaction_cost_bps=10)
    assert result.loc[0, "gross_return"] == 0.02
    assert result.loc[0, "net_return"] < result.loc[0, "gross_return"]


def test_liquidity_cost_increases_with_participation_and_borrow():
    current = pd.DataFrame({
        "ticker": ["A", "B"], "weight": [0.5, -0.5],
        "dollar_volume_20d": [1_000_000.0, 1_000_000.0], "volatility_20d": [0.02, 0.02],
    })
    small_cost, turnover = estimated_transaction_cost(
        current, None, cost_model="liquidity", portfolio_notional=100_000, half_spread_bps=1, impact_coefficient=1,
    )
    large_cost, _ = estimated_transaction_cost(
        current, None, cost_model="liquidity", portfolio_notional=10_000_000, half_spread_bps=1, impact_coefficient=1,
    )
    assert turnover == 0.5
    assert large_cost > small_cost > 0

    date = pd.Timestamp("2024-01-05")
    weights = current.assign(date=date)
    returns = pd.DataFrame({"date": [date, date], "ticker": ["A", "B"], "stock_forward_return": [0.02, -0.02]})
    result = run_weekly_backtest(weights, returns, annual_borrow_bps=52)
    assert np.isclose(result.loc[0, "borrow_cost"], 0.00005)
