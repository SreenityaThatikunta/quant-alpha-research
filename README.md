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

Before supplying `--fundamentals`, add the SEC `cik` identifier to every price
panel row. The research runner performs an as-of merge, so a filing is never
available before its recorded availability date. It then creates accounting
features such as return on assets, cash-flow-to-assets, equity-to-assets, and
asset growth, and cross-sectionally normalizes them alongside technical inputs.
This improves feature breadth but does not remove the current-constituent and
delisting limitations of the public-price universe.

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
multiple-testing-aware Sharpe diagnostic. A signal is a research candidate—not
an accepted alpha—until it survives those independent checks.

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
