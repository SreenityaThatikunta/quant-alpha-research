"""Point-in-time fundamental data utilities for SEC EDGAR XBRL facts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FUNDAMENTAL_CONCEPTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "revenue": (("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"), ("us-gaap", "SalesRevenueNet", "USD"), ("us-gaap", "Revenues", "USD")),
    "net_income": (("us-gaap", "NetIncomeLoss", "USD"),),
    "assets": (("us-gaap", "Assets", "USD"),),
    "equity": (("us-gaap", "StockholdersEquity", "USD"),),
    "operating_cash_flow": (("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),),
    "shares_outstanding": (("dei", "EntityCommonStockSharesOutstanding", "shares"),),
}
FUNDAMENTAL_FEATURE_COLUMNS = ("return_on_assets", "cashflow_to_assets", "equity_to_assets", "asset_growth")


def fetch_sec_json(url: str, user_agent: str, timeout_seconds: int = 30) -> dict[str, object]:
    """Fetch one SEC JSON endpoint using an identifying User-Agent string."""
    if len(user_agent.strip()) < 8:
        raise ValueError("Provide an identifying SEC User-Agent, for example 'Name contact@example.com'.")
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def ticker_cik_map(payload: Mapping[str, object]) -> dict[str, int]:
    """Create an upper-case ticker-to-CIK map from SEC company_tickers JSON."""
    mapping: dict[str, int] = {}
    for record in payload.values():
        if isinstance(record, Mapping) and "ticker" in record and "cik_str" in record:
            mapping[str(record["ticker"]).upper()] = int(record["cik_str"])
    return mapping


def company_facts_observations(payload: Mapping[str, object], cik: int, availability_lag_days: int = 1) -> pd.DataFrame:
    """Extract standardized 10-K/10-Q facts with a conservative next-day availability lag."""
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be non-negative")
    facts = payload.get("facts", {})
    if not isinstance(facts, Mapping):
        raise ValueError("SEC company-facts payload does not contain a facts mapping")
    rows: list[dict[str, object]] = []
    for feature, alternatives in FUNDAMENTAL_CONCEPTS.items():
        for taxonomy, tag, unit in alternatives:
            taxonomy_facts = facts.get(taxonomy, {})
            concept = taxonomy_facts.get(tag, {}) if isinstance(taxonomy_facts, Mapping) else {}
            units = concept.get("units", {}) if isinstance(concept, Mapping) else {}
            observations = units.get(unit, []) if isinstance(units, Mapping) else []
            if not observations:
                continue
            for observation in observations:
                if not isinstance(observation, Mapping) or observation.get("form") not in {"10-K", "10-Q"}:
                    continue
                filed = pd.to_datetime(observation.get("filed"), errors="coerce")
                report_end = pd.to_datetime(observation.get("end"), errors="coerce")
                value = observation.get("val")
                if pd.isna(filed) or pd.isna(report_end) or value is None:
                    continue
                rows.append({
                    "cik": cik, "feature": feature, "value": float(value), "report_end": report_end,
                    "available_date": filed.normalize() + pd.offsets.BDay(availability_lag_days),
                    "filed_date": filed.normalize(), "form": observation["form"], "source_tag": f"{taxonomy}:{tag}",
                    "accession": observation.get("accn"),
                })
            break
    if not rows:
        return pd.DataFrame(columns=["cik", "feature", "value", "report_end", "available_date", "filed_date", "form", "source_tag", "accession"])
    result = pd.DataFrame(rows).sort_values(["cik", "feature", "available_date", "report_end", "filed_date"])
    return result.drop_duplicates(["cik", "feature", "available_date", "report_end"], keep="last").reset_index(drop=True)


def align_fundamentals_asof(panel: pd.DataFrame, observations: pd.DataFrame, cik_column: str = "cik") -> pd.DataFrame:
    """Attach only the most recently available SEC fact to each security/date."""
    required_panel = {"date", "ticker", cik_column}
    required_observations = {"cik", "feature", "value", "available_date"}
    if missing := required_panel.difference(panel.columns):
        raise ValueError(f"Panel missing: {sorted(missing)}")
    if missing := required_observations.difference(observations.columns):
        raise ValueError(f"Fundamental observations missing: {sorted(missing)}")
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    facts = observations.copy()
    facts["available_date"] = pd.to_datetime(facts["available_date"])
    outputs = []
    for cik, securities in work.groupby(cik_column, dropna=False):
        security = securities.sort_values("date")
        if pd.isna(cik):
            outputs.append(security)
            continue
        history = facts.loc[facts["cik"] == int(cik)].pivot_table(index="available_date", columns="feature", values="value", aggfunc="last").reset_index().sort_values("available_date")
        if history.empty:
            outputs.append(security)
            continue
        aligned = pd.merge_asof(security, history, left_on="date", right_on="available_date", direction="backward")
        outputs.append(aligned.drop(columns="available_date"))
    return pd.concat(outputs, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def add_fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute accounting features only from already point-in-time aligned facts."""
    result = panel.copy()
    if {"net_income", "assets"}.issubset(result.columns):
        result["return_on_assets"] = result["net_income"].div(result["assets"].where(result["assets"].ne(0)))
    if {"operating_cash_flow", "assets"}.issubset(result.columns):
        result["cashflow_to_assets"] = result["operating_cash_flow"].div(result["assets"].where(result["assets"].ne(0)))
    if {"equity", "assets"}.issubset(result.columns):
        result["equity_to_assets"] = result["equity"].div(result["assets"].where(result["assets"].ne(0)))
    if "assets" in result.columns:
        result["asset_growth"] = result.groupby("ticker")["assets"].pct_change(fill_method=None)
    return result


def write_fundamentals(observations: pd.DataFrame, output_path: Path) -> None:
    """Persist the raw point-in-time fact table without altering its timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(output_path, index=False)
