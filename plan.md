# Market-Neutral Equity Alpha Research Platform

> Implementation status, 19 September 2026: a public-data Ridge baseline, leakage tests, walk-forward evaluation, and a sector/beta-neutral weekly portfolio are implemented. The documented baseline has weak net performance after transaction costs. That result remains a benchmark, not an alpha claim: the project is being extended into a statistically disciplined multi-alpha research platform.

## Objective

Build an end-to-end quantitative research project that predicts next-week **residual equity returns** and converts those forecasts into a sector- and beta-neutral long-short portfolio. The finished project should demonstrate feature research, machine-learning experimentation, realistic backtesting, and rigorous validation.

## Scope

- **Universe:** 300-500 liquid US equities, with survivorship-bias limitations documented.
- **Frequency:** Daily data; weekly portfolio rebalancing.
- **Target:** Forward five-trading-day residual return.
- **Models:** An interpretable baseline (Ridge or Elastic Net) and one nonlinear model (LightGBM or XGBoost).
- **Portfolio:** Long top-decile stocks and short bottom-decile stocks, dollar-neutral with sector and market-beta controls.

## Standout research agenda

The objective is not to maximize one backtest statistic. It is to demonstrate a repeatable process for producing a diversified, implementable market-neutral alpha portfolio and for rejecting ideas that do not survive realistic controls.

### 1. Institutional-grade data controls

- Replace the current-constituent snapshot with point-in-time universe membership, sector classifications, corporate actions, and delisting returns when an appropriate source is available.
- Keep the public-data benchmark as a clearly labelled, survivorship-biased proxy; never compare its output directly with a point-in-time production study.
- Store source metadata, timestamps, schema validation, and immutable input hashes with every experiment.

### 2. Diversified alpha library

- Research separately motivated signal families: residual momentum, short-horizon reversal, volatility/idiosyncratic-risk, liquidity, fundamentals/earnings revisions, and market-regime-conditioned variants.
- For every candidate, report coverage, rank IC, IC decay, turnover, exposure, costs, capacity proxy, and correlations to the accepted signal library.
- Combine only signals that add incremental out-of-sample value after correlation and factor-exposure controls.

### 3. Statistical validation designed to reject false discoveries

- Retain chronological walk-forward evaluation with label-overlap embargoes.
- Add purged/embargoed cross-validation utilities for tuning and nested evaluation for model selection.
- Report multiple-testing diagnostics, including deflated Sharpe ratio and probability-of-backtest-overfitting estimates where sample size permits.
- Require stability tables by year, sector, market-volatility regime, liquidity regime, and sub-universe.

### 4. Risk-aware portfolio optimization

- Replace projection-only weighting with a constrained optimizer that jointly handles gross/net exposure, sector and factor neutrality, beta, position caps, turnover, and liquidity participation.
- Persist feasibility status, realized constraint residuals, and post-optimization sign changes. A skipped or infeasible portfolio is a result to investigate, never a silently altered portfolio.
- Extend risk controls from sector/beta to transparent style-factor exposures (size, value, momentum, volatility, and liquidity) when data supports them.

### 5. Execution and capacity economics

- Replace the single flat-cost assumption with a configurable model that can include spread, volatility, dollar volume, participation, borrow, and execution-delay assumptions.
- Measure forecast decay against delayed execution and show gross/net performance across costs and capacity levels.
- Separate research return, estimated trading cost, and financing/borrow assumptions in every report.

### 6. Reproducible research operations

- Version experiment parameters, input-data fingerprints, code revision, package versions, and output schemas in a run manifest.
- Use deterministic configurations and one-command report reproduction.
- Maintain a compact experiment table that records hypotheses, rejection criteria, OOS results, and the decision to promote, revise, or reject each alpha.

## Milestone acceptance criteria

| Milestone | Evidence required before advancing |
| --- | --- |
| Data integrity | Point-in-time source/proxy documented; membership, classifications, and delisting assumptions tested |
| Alpha research | Per-signal IC/decay/correlation diagnostics and OOS ablations committed |
| Model selection | Purged nested validation and multiple-testing outputs; no final OOS set used for tuning |
| Portfolio | Feasible constrained weights with audited risk, turnover, and liquidity constraints |
| Economics | Cost, delay, and capacity sensitivity results reported net of assumptions |
| Reproducibility | A run manifest and deterministic command reproduce all published result tables |

## Repository Structure

```text
quant-alpha-research/
├── data/
│   ├── raw/                 # Downloaded source data; do not commit large files
│   └── processed/           # Cleaned panels and feature matrices
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_research.ipynb
│   ├── 03_model_research.ipynb
│   └── 04_backtest_analysis.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── targets.py
│   ├── models.py
│   ├── portfolio.py
│   ├── backtest.py
│   └── metrics.py
├── tests/
├── reports/
│   ├── figures/
│   └── final_report.md
├── requirements.txt
└── README.md
```

## Phase 1 - Data and Research Design

### Tasks

1. Select a reproducible data source for adjusted daily OHLCV prices, volumes, market benchmark data, and sector classifications.
2. Define the investable universe using liquidity and price filters, such as median daily dollar volume and a minimum price threshold.
3. Create a clean daily panel keyed by `date` and `ticker`.
4. Document data availability, missing values, corporate-action treatment, universe limitations, and all assumptions.
5. Define the target before modeling:

   ```text
   target = stock forward 5-day return - beta * benchmark forward 5-day return
   ```

### Deliverables

- `notebooks/01_data_audit.ipynb`
- A data dictionary and a reproducible universe-construction function.
- A short note describing potential survivorship bias and any source-data limitations.

### Acceptance checks

- No duplicate `(date, ticker)` rows.
- Every feature uses information available no later than the signal date.
- The target is shifted forward and cannot appear in the model inputs.

## Phase 2 - Feature Engineering

### Initial Feature Families

| Family | Examples |
| --- | --- |
| Momentum | 5-, 20-, 60-, and 120-day returns; skip-period momentum |
| Reversal | 1- and 5-day returns; distance from short moving average |
| Volatility | Rolling standard deviation, downside volatility, ATR proxy |
| Volume and liquidity | Relative volume, volume z-score, dollar-volume rank |
| Market sensitivity | Rolling beta, correlation with benchmark, idiosyncratic volatility |
| Cross-sectional ranks | Sector-relative ranks and winsorized z-scores for all continuous features |

### Tasks

1. Compute rolling features with explicit lookback windows and no forward filling across unavailable dates.
2. Winsorize extreme values and convert features into date-wise cross-sectional z-scores or ranks.
3. Neutralize each signal against sector and market-beta exposures where appropriate.
4. Create feature-quality diagnostics: coverage, missingness, correlations, turnover, and rank information coefficient (IC).
5. Add unit tests for rolling-window and shifting logic.

### Deliverables

- `src/features.py`
- `notebooks/02_feature_research.ipynb`
- Feature-correlation heatmap and rolling rank-IC charts.

## Phase 3 - Walk-Forward Model Research

### Validation Design

Use expanding or rolling windows. Never randomly shuffle time-series observations.

```text
Train: historical window ending at T
Validate / tune: period immediately after T
Test: next out-of-sample period
Advance T and repeat
```

Use a small embargo between training and test windows if overlapping forward-return labels could leak information.

### Tasks

1. Establish baselines: individual feature ranks, equal-weight composite, and Ridge/Elastic Net regression.
2. Train a gradient-boosted tree model with a deliberately small, documented hyperparameter search.
3. Save out-of-sample predictions only; concatenate them for portfolio construction.
4. Evaluate prediction quality using Pearson IC, rank IC, IC information ratio, and directional accuracy.
5. Perform feature-group ablations: remove momentum, reversal, volatility, and volume groups one at a time.

### Deliverables

- `src/models.py`
- `notebooks/03_model_research.ipynb`
- A table comparing all models only on out-of-sample data.

## Phase 4 - Portfolio Construction and Backtesting

### Portfolio Rules

1. Rebalance weekly after the signal is available, with execution beginning on the next trading day.
2. Rank the eligible universe by model prediction.
3. Long the highest decile and short the lowest decile.
4. Start with equal-weighted positions, then neutralize dollar, sector, and market-beta exposures.
5. Impose constraints: maximum position size, maximum sector exposure, turnover cap, and liquidity screen.
6. Deduct transaction costs and slippage proportional to turnover.

### Required Metrics

- Annualized return and volatility
- Sharpe ratio
- Maximum drawdown
- Average weekly turnover
- Gross and net performance after costs
- Long-leg, short-leg, and long-short performance
- Rolling beta and sector exposure
- Performance by market regime

### Deliverables

- `src/portfolio.py` and `src/backtest.py`
- `notebooks/04_backtest_analysis.ipynb`
- Cumulative-return, drawdown, rolling-Sharpe, turnover, exposure, and regime-performance charts.

## Phase 5 - Robustness and Research Integrity

### Required checks

- **No look-ahead bias:** all signals are lagged; trades occur after signal generation.
- **No leakage:** preprocessing and normalization are fit only on training data when model-based.
- **Transaction costs:** report gross and net returns.
- **Survivorship bias:** state whether the universe uses present-day constituents and how this affects conclusions.
- **Sensitivity:** vary rebalance frequency, transaction costs, number of selected stocks, and feature windows.
- **Stability:** report results by year and by market regime, not only the aggregate result.

### Deliverables

- `tests/` covering target alignment, feature lagging, and next-day execution.
- `reports/final_report.md` with methodology, results, limitations, and next steps.

## Four-Week Schedule

| Week | Focus | Output |
| --- | --- | --- |
| 1 | Data pipeline, universe, target, audit | Clean panel and reproducible data notebook |
| 2 | Features and baseline signals | Feature library, IC analysis, tests |
| 3 | Walk-forward model experiments | Out-of-sample prediction comparison and ablations |
| 4 | Portfolio simulation, robustness, documentation | Final backtest report, polished README, resume bullets |

## Final README Checklist

- One-paragraph research question and investment hypothesis.
- Data source, sample period, and known limitations.
- Clear statement that the project is educational research, not investment advice.
- Setup and reproduction instructions.
- A concise methodology diagram.
- Out-of-sample results with costs and constraints clearly stated.
- Link or screenshots of the main research report and figures.

## Resume-Ready Result

Only use numerical outcomes after validating them out of sample. A final bullet template:

> Built a market-neutral equity alpha research platform for [N] liquid US equities, engineering [N] cross-sectional features and evaluating walk-forward ML forecasts with rank IC, transaction costs, sector/beta neutralization, and regime-level robustness tests.

## Stretch Goals

- Add Fama-French-style factor-exposure analysis.
- Optimize portfolio weights with constrained convex optimization.
- Add alternative data only if its timestamping and licensing are well documented.
- Package the backtest as a small Streamlit dashboard for inspecting signals, exposures, and performance.
