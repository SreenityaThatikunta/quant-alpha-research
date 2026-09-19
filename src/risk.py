"""Public factor-risk data and attribution utilities."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


FAMA_FRENCH_5_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FAMA_FRENCH_FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")


def parse_fama_french_daily_zip(content: bytes) -> pd.DataFrame:
    """Parse the official daily five-factor ZIP, converting percentage returns to decimals."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(csv_name).decode("latin-1").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.strip().startswith("DATE,"))
    data_lines = []
    for line in lines[header_index:]:
        if not line.strip():
            break
        data_lines.append(line)
    factors = pd.read_csv(io.StringIO("\n".join(data_lines)))
    factors = factors.rename(columns={factors.columns[0]: "date"})
    factors["date"] = pd.to_datetime(factors["date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    factors = factors.dropna(subset=["date"])
    for column in FAMA_FRENCH_FACTOR_COLUMNS:
        factors[column] = pd.to_numeric(factors[column], errors="coerce") / 100
    return factors[["date", *FAMA_FRENCH_FACTOR_COLUMNS]].sort_values("date").reset_index(drop=True)


def download_fama_french_daily(timeout_seconds: int = 30) -> pd.DataFrame:
    """Download and parse the official Ken French daily five-factor archive."""
    with urlopen(FAMA_FRENCH_5_DAILY_URL, timeout=timeout_seconds) as response:  # noqa: S310
        return parse_fama_french_daily_zip(response.read())


def write_factor_data(factors: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(output_path, index=False)


def factor_attribution(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    return_column: str = "net_return",
    factor_columns: tuple[str, ...] = FAMA_FRENCH_FACTOR_COLUMNS[:-1],
) -> pd.Series:
    """Estimate OLS alpha and factor exposures for same-frequency returns.

    Inputs must already share the same holding-period convention. This function
    intentionally does not turn daily factors into weekly returns implicitly.
    """
    if missing := {"date", return_column}.difference(returns.columns):
        raise ValueError(f"Returns missing: {sorted(missing)}")
    if missing := {"date", *factor_columns, "RF"}.difference(factors.columns):
        raise ValueError(f"Factors missing: {sorted(missing)}")
    merged = returns[["date", return_column]].merge(factors[["date", *factor_columns, "RF"]], on="date", how="inner").dropna()
    if len(merged) <= len(factor_columns) + 1:
        raise ValueError("Insufficient aligned observations for factor attribution")
    excess = merged[return_column].to_numpy(dtype=float) - merged["RF"].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(merged)), merged.loc[:, factor_columns].to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(matrix, excess, rcond=None)
    residuals = excess - matrix @ coefficients
    return pd.Series({
        "observations": len(merged), "alpha_per_period": coefficients[0],
        "residual_volatility": residuals.std(ddof=1),
        **{f"beta_{factor}": coefficient for factor, coefficient in zip(factor_columns, coefficients[1:], strict=True)},
    })
