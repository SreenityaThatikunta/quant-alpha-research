# Final research report

## Status

Completed baseline run on 19 September 2026. This is an educational, public-data benchmark—not investment advice or a production trading result.

## Methodology

- Universe: liquid US equities screened using trailing median dollar volume and prior-day price.
- Target: five-trading-day residual return using trailing market beta.
- Validation: chronological walk-forward folds with a label-overlap embargo.
- Portfolio: weekly rebalancing with next-session-open entry and five-session-close marking; decile long/short selections with dollar, sector, and beta exposure controls.
- Costs: turnover-proportional transaction costs reported alongside gross performance.

## Required results

Report only out-of-sample metrics: Pearson/rank IC, IC IR, directional accuracy, annualized return/volatility, Sharpe, drawdown, turnover, gross/net returns, exposure diagnostics, yearly results, and market-regime splits.

## Baseline run: current S&P 500 public-data snapshot

| Item | Result |
| --- | ---: |
| Universe | 300 current S&P 500 constituents, liquidity/price screened |
| Raw observations | 641,074 daily equity-price rows |
| Model | Ridge regression on cross-sectional technical features |
| Validation | Expanding walk-forward; 504 training days, 5-day embargo, 252-day test blocks |
| OOS prediction observations | 101,267 |
| Mean Pearson IC / rank IC | 0.0129 / 0.0079 |
| Rank-IC information ratio | 0.0480 |
| Directional accuracy | 50.05% |
| Rebalance observations | 149 weekly portfolios, 28 Feb 2020–11 Sep 2026 |
| Gross annualized return / volatility / Sharpe | 3.43% / 7.56% / 0.45 |
| Net annualized return / volatility / Sharpe | -0.72% / 7.56% / -0.10 |
| Net maximum drawdown | -9.35% |
| Mean one-way weekly turnover | 78.84% |
| Assumed transaction cost | 10 bps per unit of turnover |
| Max absolute dollar / beta / sector exposure | < 2e-14 / < 4e-14 / < 9e-15 |

The net result is weak after costs. The correct conclusion is that this baseline does not establish a tradable alpha; future work should compare feature groups, tune only within walk-forward validation, and test sensitivity to costs and selection size.

## Limitations

The run uses a snapshot of **current** S&P 500 members and current sector classifications. It therefore has material survivorship bias and is not a point-in-time constituent backtest. Yahoo Finance is a convenient public adjusted-price source, not an institutional data feed. The project is educational research, not investment advice.
