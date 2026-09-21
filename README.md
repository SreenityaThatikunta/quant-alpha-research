# Market-Neutral Equity Alpha Research

An educational, end-to-end research platform for predicting five-session
residual equity returns and constructing sector- and beta-neutral long-short
portfolios. It is **not investment advice**.

```text
point-in-time eligibility → leakage-safe features → nested OOS forecasts
→ neutral portfolio construction → execution and cost diagnostics
```

## What this demonstrates

- Reproducible public-data and point-in-time-proxy pipelines
- Chronological, embargoed nested walk-forward model selection
- Market-neutral weekly portfolios with transaction-cost and capacity analysis
- Explicit research decisions: fragile signals are rejected rather than promoted

## Key finding

The completed public point-in-time membership-proxy study found positive
baseline forecasts but insufficient robustness to claim alpha.

| OOS metric | Baseline entry | Two-session delay |
| --- | ---: | ---: |
| Mean rank IC | 1.84% | 1.11% |
| Net Sharpe | 0.57 | 0.24 |
| Maximum drawdown | -7.02% | -8.93% |
| Mean weekly turnover | 76.98% | 77.63% |

At a $1m notional, baseline Sharpe falls from 0.34 at a 5 bps half-spread
assumption to -0.02 at 10 bps and -0.73 at 20 bps. The study is therefore
recorded as **rejected**, not presented as a tradable strategy.

## Dashboard and evidence

- [Research dashboard](reports/dashboard.md)
- [Final report](reports/final_report.md)
- [Research decision log](reports/research_decision_log.md)

![Net cumulative return comparison](reports/figures/equity_curve.svg)

![Liquidity sensitivity](reports/figures/liquidity_sensitivity.svg)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Reproduce the completed point-in-time proxy study

```bash
python -m scripts.data.download_pit_data --start 2016-01-01 --end 2025-12-31
python -m scripts.data.download_factors --output data/raw/fama_french_5_daily.parquet

python -m scripts.research.run_research \
  --panel data/raw/sp500_pit_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --universe-history data/raw/sp500_pit_universe.parquet \
  --factors data/raw/fama_french_5_daily.parquet \
  --output data/processed/sp500_pit_proxy \
  --model ridge --nested-validation --ridge-alphas 1,10,100 --test-days 252 \
  --cost-model liquidity --cost-sensitivity-bps 5,10,20 \
  --notional-sensitivity 1000000,5000000,10000000

python -m scripts.reporting.generate_report_assets
```

The free `pitindex`/Yahoo workflow is a point-in-time membership proxy, not a
CRSP-quality data set: delisting returns, ticker history, corporate actions,
and price availability remain limitations. The generated run manifest records
parameters, input fingerprints, and environment information.

## Repository map

| Folder | Purpose |
| --- | --- |
| `src/` | Research, modeling, portfolio, and backtest components |
| `scripts/data/` | Public data, factor, and PIT-proxy download commands |
| `scripts/research/` | Full-model and signal-library experiment commands |
| `scripts/reporting/` | Figure and dashboard generation |
| `reports/` | Dashboard, final report, data provenance, and decision log |
| `docs/` | Research plan |
| `tests/` | Leakage, alignment, neutrality, and cost controls |

For the detailed methodology, alternative workflows (SEC fundamentals and
signal-library studies), and assumptions, start with the [final report](reports/final_report.md).
