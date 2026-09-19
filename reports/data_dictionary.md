# Data dictionary

| Field | Meaning | Availability |
| --- | --- | --- |
| `date`, `ticker` | Daily panel key | Signal date |
| `open`, `high`, `low`, `close`, `volume` | Adjusted daily OHLCV | End of signal date |
| `sector` | Current constituent snapshot's GICS sector | Required for sector-neutrality constraint; not point-in-time historical data |
| `eligible` | Price/liquidity universe flag | Computed solely from data through prior close |
| `beta` | Trailing market beta | Computed through signal date |
| `stock_forward_return` | Next-open to fifth-session-close stock return | Label only; unavailable at signal date |
| `benchmark_forward_return` | Next-open to fifth-session-close benchmark return | Label only; unavailable at signal date |
| `target_residual_return_5d` | Stock forward return less beta × benchmark forward return | Label only; unavailable at signal date |
| `*_rank`, `*_zscore` | Date-wise winsorized cross-sectional feature transforms | Signal date |

The raw source uses adjusted Yahoo Finance history and a downloaded current-constituent sector snapshot for reproducibility. It does not provide a point-in-time index membership history. Ticker selection and sector classifications are saved with each run; the current snapshot still creates material survivorship bias.
