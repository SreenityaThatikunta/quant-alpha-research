import pandas as pd

from src.backtest import run_weekly_backtest


def test_backtest_deducts_turnover_costs():
    date = pd.Timestamp("2024-01-05")
    weights = pd.DataFrame({"date": [date, date], "ticker": ["A", "B"], "weight": [0.5, -0.5]})
    returns = pd.DataFrame({"date": [date, date], "ticker": ["A", "B"], "stock_forward_return": [0.02, -0.02]})
    result = run_weekly_backtest(weights, returns, transaction_cost_bps=10)
    assert result.loc[0, "gross_return"] == 0.02
    assert result.loc[0, "net_return"] < result.loc[0, "gross_return"]
