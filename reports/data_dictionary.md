# Data dictionary

| Field | Meaning | Availability |
| --- | --- | --- |
| `date`, `ticker` | Daily panel key | Signal date |
| `open`, `high`, `low`, `close`, `volume` | Adjusted daily OHLCV | End of signal date |
| `eligible` | Price/liquidity universe flag | Computed solely from data through prior close |
| `beta` | Trailing market beta | Computed through signal date |
| `stock_forward_return` | Stock return for the next five trading days | Label only; unavailable at signal date |
| `benchmark_forward_return` | Benchmark return for the next five trading days | Label only; unavailable at signal date |
| `target_residual_return_5d` | Stock forward return less beta × benchmark forward return | Label only; unavailable at signal date |
| `*_rank`, `*_zscore` | Date-wise winsorized cross-sectional feature transforms | Signal date |

The raw source uses adjusted Yahoo Finance history for reproducibility. It does not provide a point-in-time index membership history. Ticker selection and sector classifications must be versioned with each run; otherwise results can contain survivorship bias.
