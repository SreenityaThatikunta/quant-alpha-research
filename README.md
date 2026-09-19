# Market-Neutral Equity Alpha Research

An educational research platform for forecasting next-week residual returns across a liquid US-equity universe and testing sector- and beta-neutral long-short portfolios. It is **not investment advice**.

## Current scope

Phase 1 creates an auditable daily panel, reproducible liquidity universe, and five-trading-day residual-return label. Features and trades must be based only on information available at the signal date.

```
price history -> clean daily panel -> eligible universe -> forward residual-return target
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data and limitations

`src.data.download_price_history` uses Yahoo Finance adjusted daily data for convenient reproducibility. Yahoo data and a manually supplied ticker/sector mapping do not constitute a point-in-time constituent universe; results can therefore have survivorship bias and should be treated as research only. The raw data cache is excluded from version control.

## Phase 1 contract

- Panel key is unique `(date, ticker)`.
- Eligibility uses trailing median dollar volume and a closing-price floor, shifted one day before use.
- The target enters at the next session's open, holds for five sessions, and is the resulting stock return minus trailing-beta times the benchmark return over the same window.
- Labels are stored separately from input features and are never contemporaneous inputs.
