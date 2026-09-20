# Research decision log

This log records research decisions, including negative findings. It is not a
performance leaderboard. A candidate can be promoted only after its
pre-specified validation design is completed and it contributes incremental,
net-of-cost out-of-sample value.

| Study ID | Hypothesis | Data / universe | Validation | Key OOS evidence | Decision | Rationale / next action |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline-ridge-20260919` | Cross-sectional technical, volatility, liquidity, and market-sensitivity features forecast five-session residual returns. | 300 current S&P 500 constituents; public adjusted daily prices; survivorship-biased benchmark. | Expanding walk-forward, 504 training days, five-session embargo, 252 test days; weekly next-open entry. | 101,267 predictions; mean rank IC 0.0079; gross Sharpe 0.45; net Sharpe -0.10 at 10 bps per unit turnover; mean turnover 78.84%. | `reject` | Gross performance did not survive stated transaction costs. Preserve as a reproducible baseline; do not represent it as tradable alpha. |
| `technical-library-20260920` | Transparent momentum, reversal, low-risk, and liquidity signals add implementable standalone alpha. | Same public 300-name benchmark. | Pre-specified candidates; weekly next-session entry, five-session holding period, sector/beta-neutral projection, 10 bps turnover cost. | Neutralized short-horizon reversal: rank IC 0.0180, net Sharpe 0.03, turnover 1.25. Other evaluated candidates had negative net Sharpe; raw low-risk had only 38 feasible dates. | `reject` / `invalid` | Do not combine candidates. Re-test neutralized short-horizon reversal only through a pre-registered nested OOS study with turnover controls; diagnose low-risk feasibility through constrained optimization. |
| `pit-proxy-ridge-20260920` | A leakage-controlled technical Ridge model retains usable residual-return forecasts after replacing current constituents with a historical membership proxy. | Public `pitindex` S&P 500 membership reconstruction, Yahoo adjusted prices, Fama–French daily factors; 729 members identified, 592 with price histories. | Nested expanding walk-forward; 504 training days, five-session embargo, pre-specified Ridge penalties 1/10/100, 252-day outer blocks. | Mean rank IC 0.0184; net Sharpe 0.57 at the baseline 2 bps half-spread/\$1m assumption. A two-session delay reduced rank IC to 0.0111 and Sharpe to 0.24. Baseline Sharpe was 0.34 at 5 bps half-spread and negative at 10 bps and 20 bps. | `reject` | Results are sensitive to delay and economically plausible spread assumptions. The free source is also incomplete for delistings. Preserve the manifests and artifacts; do not combine or market the signal as alpha. |
| `sec-fundamentals-next` | Filing-timestamped accounting quality features add incremental OOS value beyond price-derived signals. | SEC EDGAR XBRL facts joined as-of recorded availability; current-universe limitation remains. | To be pre-registered before final OOS evaluation. | Pending. | `pending` | Define availability rules, feature set, trial count, and rejection thresholds before evaluating results. |

## Decision rules

- `promote`: Survives pre-specified OOS, cost, stability, and incremental-value checks.
- `hold`: Evidence is incomplete; no combination or performance claim is allowed.
- `revise`: The economic hypothesis remains plausible, but implementation or data diagnostics require change.
- `reject`: The candidate failed its stated OOS or economic criteria.
- `invalid`: A data, feasibility, or validation failure prevents performance interpretation.

Every new row must link to its run manifest and retained output artifacts.
