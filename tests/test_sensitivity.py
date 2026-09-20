import pandas as pd

from src.sensitivity import execution_cost_capacity_sensitivity


def test_flat_cost_sensitivity_holds_weights_fixed_and_reports_each_assumption():
    dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
    weights = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1]],
        "ticker": ["A", "B", "A", "B"],
        "weight": [0.5, -0.5, 0.5, -0.5],
        "stock_forward_return": [0.02, -0.01, 0.01, 0.00],
    })
    sensitivity = execution_cost_capacity_sensitivity(
        weights, weights, cost_bps_values=[0, 10], portfolio_notionals=[1_000_000, 2_000_000]
    )
    assert len(sensitivity) == 4
    assert set(sensitivity["cost_assumption_type"]) == {"flat_turnover_cost"}
    zero_cost = sensitivity.loc[sensitivity["cost_assumption_bps"] == 0, "annualized_return"].iloc[0]
    positive_cost = sensitivity.loc[sensitivity["cost_assumption_bps"] == 10, "annualized_return"].iloc[0]
    assert zero_cost > positive_cost


def test_liquidity_sensitivity_varies_half_spread_assumption():
    date = pd.Timestamp("2024-01-05")
    weights = pd.DataFrame({
        "date": [date, date], "ticker": ["A", "B"], "weight": [0.5, -0.5],
        "stock_forward_return": [0.02, -0.01],
        "dollar_volume_20d": [1_000_000.0, 1_000_000.0], "volatility_20d": [0.02, 0.02],
    })
    sensitivity = execution_cost_capacity_sensitivity(
        weights, weights, cost_model="liquidity", cost_bps_values=[1, 5], portfolio_notionals=[1_000_000]
    )
    assert set(sensitivity["cost_assumption_type"]) == {"half_spread"}
    assert sensitivity.sort_values("cost_assumption_bps")["mean_trading_cost"].iloc[1] > sensitivity.sort_values("cost_assumption_bps")["mean_trading_cost"].iloc[0]
