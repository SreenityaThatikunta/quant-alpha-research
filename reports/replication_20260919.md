# Public-data replication — 19 September 2026

## Status

This is a fresh, reproducible run of the price-only Ridge baseline after the
research-system upgrades. It is a **negative result**: the technical-feature
model does not establish a tradable alpha after the stated costs. The result is
recorded to preserve the research decision, not to imply investment advice.

## Reproduction

```bash
.venv/bin/python download_data.py --limit 300
.venv/bin/python run_research.py \
  --panel data/raw/sp500_current_constituents_prices.parquet \
  --benchmark data/raw/spy_benchmark.parquet \
  --output data/processed/sp500_ridge_upgraded \
  --model ridge --test-days 252
```

The generated `run_manifest.json` records the exact input hashes, parameters,
package versions, and code revision. Raw inputs and generated outputs remain
excluded from Git because they are reproducible artifacts.

## Out-of-sample results

| Metric | Result |
| --- | ---: |
| OOS prediction observations | 101,267 |
| Mean Pearson IC | 0.0129 |
| Mean rank IC | 0.0079 |
| Rank-IC IR | 0.0480 |
| Directional accuracy | 50.05% |
| Weekly portfolio periods | 149 |
| Net annualized return | -2.31% |
| Annualized volatility | 7.57% |
| Net Sharpe | -0.31 |
| Maximum drawdown | -11.14% |
| Deflated-Sharpe probability, one trial | 0.32 |

## Interpretation and decision

The weak IC and negative net performance fail the promotion criterion. Do not
optimize this baseline further on the same final OOS window. The next research
stage is to test the independently defined technical signals and then the SEC
filing-timestamped fundamental features using nested selection, correlation
controls, liquidity-aware costs, and the constrained portfolio optimizer.

This public-universe study remains exposed to current-constituent survivorship
bias and incomplete delisting treatment. See `reports/data_provenance.md`.
