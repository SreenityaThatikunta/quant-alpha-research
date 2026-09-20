# Market-Neutral Equity Alpha Research

An educational research platform for forecasting next-week residual returns across a liquid US-equity universe and testing sector- and beta-neutral long-short portfolios. It is **not investment advice**.

## Research question

Can cross-sectional technical, volatility, liquidity, and market-sensitivity features forecast five-session residual equity returns well enough to support a market-neutral long-short portfolio after realistic turnover costs?

```
public data snapshot -> point-in-time eligibility -> leakage-safe features
    -> walk-forward OOS forecasts -> sector/beta-neutral portfolio -> net performance
```

## Completed baseline

The current public-data Ridge baseline uses 300 current S&P 500 constituents, a five-day label-overlap embargo, weekly rebalancing, next-session-open entry, five-session-close marking, and 10 bps of cost per unit of turnover.

| Out-of-sample metric | Result |
| --- | ---: |
| Mean rank IC | 0.0079 |
| Net annualized return | -0.72% |
| Net Sharpe | -0.10 |
| Net maximum drawdown | -9.35% |
| Mean weekly turnover | 78.84% |

This weak net baseline does **not** establish a tradable alpha. See [the full report](reports/final_report.md) for methodology, constraints, and limitations.

## Research decisions and evidence

This repository records negative findings rather than promoting a signal based
on an attractive gross backtest. The current candidates, OOS evidence, and
promotion/rejection decisions are maintained in the
[research decision log](reports/research_decision_log.md). The public-data
baseline remains a survivorship-biased benchmark while the project builds a
point-in-time universe study.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Reproduce the public-data baseline

```bash
python download_data.py --limit 300
python run_research.py \
  --panel data/raw/sp500_current_constituents_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --output data/processed/sp500_ridge_next_open \
  --model ridge --test-days 252
```

To select the Ridge penalty without evaluating candidate penalties on the
outer test blocks, use nested walk-forward selection. The listed penalties
must be chosen before reviewing the resulting final OOS metrics.

```bash
python run_research.py \
  --panel data/raw/sp500_current_constituents_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --output data/processed/sp500_ridge_nested \
  --model ridge --nested-validation --ridge-alphas 1,10,100
```

This writes `nested_model_selections.csv` alongside the ordinary run manifest
and OOS artifacts. It does not make the current-constituent benchmark
survivorship-free.

For a fixed OOS portfolio, publish cost/capacity assumptions rather than a
single cost estimate:

```bash
python run_research.py ... --cost-model liquidity \
  --cost-sensitivity-bps 5,10,20 --notional-sensitivity 1000000,5000000,10000000
```

This writes `execution_cost_capacity_sensitivity.csv`. It revalues fixed OOS
weights; it is not another round of signal selection.

For a point-in-time study, supply a historical membership change log with
`ticker`, `effective_date`, `metadata_available_date`, `in_universe`, and a
historical `sector` classification. The runner joins only records available by
the signal date and records the source file hash in its manifest:

```bash
python run_research.py \
  --panel data/raw/vendor_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --universe-history data/raw/vendor_universe_history.parquet \
  --output data/processed/point_in_time_study
```

The runner writes the labeled panel, out-of-sample predictions, portfolio weights, daily IC, exposure diagnostics, backtest, and summary to the requested output folder. Raw and generated data are intentionally excluded from version control.

## Add point-in-time SEC fundamentals

The optional SEC EDGAR XBRL path adds filing-timestamped fundamentals to the
public-price baseline. It uses a conservative next-business-day availability
lag; provide an identifying contact string when retrieving SEC data.

```bash
python download_fundamentals.py \
  --tickers AAPL,MSFT,NVDA \
  --user-agent "Your Name your-email@example.com" \
  --output data/raw/sec_fundamentals.parquet
```

The research runner joins the downloaded facts to the price panel by ticker and
performs an as-of merge, so a filing is never available before its recorded
availability date. It then creates accounting
features such as return on assets, cash-flow-to-assets, equity-to-assets, and
asset growth, and cross-sectionally normalizes them alongside technical inputs.
This improves feature breadth but does not remove the current-constituent and
delisting limitations of the public-price universe.

For a large universe, download bounded batches and store each result separately
before concatenating the Parquet files:

```bash
python download_fundamentals.py ... --tickers-file data/raw/sp500_current_constituents_sectors.csv \
  --offset 0 --limit 25 --output data/raw/sec_fundamentals_000.parquet
```

## Build a free point-in-time S&P 500 proxy

The public `pitindex` source reconstructs S&P 500 membership from free public
records. It is a substantial improvement over using today's constituents, but
not a replacement for CRSP: its upstream event dates, ticker history, and
corporate-action coverage are documented limitations. The downloader saves both
the membership change log and its provenance.

```bash
python download_pit_universe.py --start 2016-01-01 --end 2025-12-31 \
  --output data/raw/sp500_pit_universe.parquet
```

Use the output during a research run:

```bash
python run_research.py \
  --panel data/raw/sp500_pit_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --universe-history data/raw/sp500_pit_universe.parquet \
  --output data/processed/sp500_pit_proxy
```

The project records a missing-ticker audit before interpreting performance;
public price sources can lack fully delisted histories, so this is a
point-in-time membership proxy—not a claim of fully survivorship-free data.

## Add public factor-risk data

Download the official daily Fama–French five-factor series for risk attribution
and factor-neutrality research:

```bash
python download_factors.py --output data/raw/fama_french_5_daily.parquet
```

Factor attribution requires factor returns aggregated to the same holding
period as the portfolio return; the code intentionally refuses to infer that
aggregation automatically.

## Evaluate the transparent alpha library

Evaluate momentum, reversal, low-risk, liquidity, and their beta/sector-neutral
versions before combining any candidates:

```bash
python run_signal_research.py \
  --panel data/raw/sp500_current_constituents_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --output data/processed/technical_signal_study
```

The output separates each signal's rank IC, net backtest, turnover, and
multiple-testing-aware Sharpe diagnostic, plus rank-IC stability by calendar
year and sector. A signal is a research candidate—not an accepted alpha—until
it survives those independent checks.

When two or more candidates have aligned backtests, the runner also writes a
combinatorially symmetric probability-of-backtest-overfitting diagnostic. It
is valid only for the pre-specified candidate set recorded in the run manifest;
it cannot correct for ideas discarded before the run.

For a targeted, resumable run, select an individual candidate (and optionally
its neutralized version):

```bash
python run_signal_research.py ... --signals short_horizon_reversal --include-neutralized
```

## Project layout

- `src/`: data validation, labels, features, walk-forward models, portfolio construction, backtesting, and metrics.
- `notebooks/`: audit, feature, model, and backtest entry points.
- `tests/`: leakage, alignment, neutrality, and cost checks.
- `reports/`: data dictionary and research report.

## Data and limitations

The baseline downloads adjusted daily Yahoo Finance data and a snapshot of current S&P 500 constituents with sector classifications. This is reproducible but not a point-in-time constituent history, so it has material survivorship bias. Data availability, corporate-action quality, delistings, and execution assumptions also limit interpretation. The project is educational research—not investment advice.

## Research-integrity contract

- Panel key is unique `(date, ticker)`.
- Eligibility uses trailing median dollar volume and a closing-price floor, shifted one day before use.
- The target enters at the next session's open, holds for five sessions, and is the resulting stock return minus trailing-beta times the benchmark return over the same window.
- Walk-forward preprocessing is fit only within each historical fold, with a five-session embargo before each test block.
- Portfolios are dollar-, sector-, and beta-neutral; a name without a contemporaneous sector or beta estimate is not traded.
