# Technical signal candidate screen — 20 September 2026

## Scope and guardrail

This is an exploratory screen of pre-specified technical signals across the
public 300-name current-constituent universe. The signals themselves do not fit
parameters on these returns, but this screen is **not** a final untouched OOS
test. No candidate may be promoted or combined based on these results alone.

All studies use next-session entry, five-session holding returns, weekly
rebalance dates, dollar/sector/beta-neutral projection, and 10 bps per unit of
turnover. Generated source data, backtests, and summaries are reproducible
under `data/processed/*_signal_study` and excluded from Git.

## Results

| Candidate | Mean rank IC | Net Sharpe | Mean turnover | Portfolios | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Intermediate momentum | -0.0013 | -0.77 | 1.11 | 186 | Reject |
| Intermediate momentum, neutralized | 0.0088 | -0.50 | 1.01 | 442 | Reject |
| Short-horizon reversal | 0.0219 | -0.67 | 1.27 | 149 | Reject: turnover dominates |
| Short-horizon reversal, neutralized | 0.0180 | 0.03 | 1.25 | 442 | Hold only as a hypothesis; not promotable |
| Low risk | 0.0048 | 0.28 | 0.81 | 38 | Invalid screen: insufficient feasible portfolios |
| Low risk, neutralized | -0.0035 | -1.18 | 0.55 | 442 | Reject |
| Liquidity quality | 0.0054 | -0.46 | 1.07 | 383 | Reject |
| Liquidity quality, neutralized | 0.0053 | -0.53 | 0.95 | 442 | Reject |

## Research decisions

1. Do not combine any candidate into an alpha ensemble yet. A combined
   in-sample score would compound selection bias without evidence of net value.
2. Retain neutralized short-horizon reversal only for a pre-registered nested
   OOS experiment with turnover control, execution delay, and liquidity-aware
   costs.
3. Treat the low-risk raw result as a portfolio-construction diagnostic, not a
   performance result: only 38 of 442 eligible weekly dates satisfied the
   projection and maximum-weight constraints. Re-run it through the constrained
   optimizer before interpreting economics.
4. Prioritize SEC filing-timestamped accounting features next. They add a
   distinct information set instead of another variation of price history.
