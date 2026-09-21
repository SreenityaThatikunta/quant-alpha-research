# Point-in-time proxy research dashboard

Completed nested out-of-sample Ridge study using a public historical-membership
proxy. This is an educational research artifact, not investment advice.

| Metric | Result | Context |
| --- | ---: | --- |
| Baseline rank IC | 1.84% | Mean OOS rank correlation |
| Baseline Sharpe | 0.57 | 2 bps half-spread baseline |
| Delayed Sharpe | 0.24 | Two-session entry delay |
| Historical members | 729 | 592 with free price history |

Rejected: performance is fragile to two-session execution delay and to 10–20 bps half-spread assumptions. This dashboard presents a research control, not an alpha claim.

## Net cumulative return

![Net cumulative return comparison](figures/equity_curve.svg)

## Forecast stability

![Rolling rank IC comparison](figures/rank_ic_decay.svg)

## Cost sensitivity

| Half-spread assumption | Baseline Sharpe | Two-session-delay Sharpe |
| --- | ---: | ---: |
| 5 bps | 0.34 | 0.02 |
| 10 bps | -0.02 | -0.33 |
| 20 bps | -0.73 | -1.02 |

![Liquidity-cost sensitivity](figures/liquidity_sensitivity.svg)

For an interactive local view, open [dashboard.html](dashboard.html).
