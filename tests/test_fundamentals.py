import pandas as pd

from src.fundamentals import add_fundamental_features, align_fundamentals_asof, company_facts_observations, ticker_cik_map


def test_ticker_mapping_and_company_fact_availability_lag():
    mapping = ticker_cik_map({"0": {"ticker": "ABC", "cik_str": 123}})
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [{"form": "10-Q", "filed": "2024-02-01", "end": "2023-12-31", "val": 100, "accn": "x"}]}}}}}
    observations = company_facts_observations(payload, mapping["ABC"], availability_lag_days=1)
    assert observations.loc[0, "available_date"] == pd.Timestamp("2024-02-02")
    assert observations.loc[0, "feature"] == "assets"


def test_fundamentals_are_aligned_only_after_availability_date():
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05"]), "ticker": ["ABC"] * 3, "cik": [123] * 3})
    observations = pd.DataFrame({"cik": [123, 123, 123], "feature": ["assets", "net_income", "operating_cash_flow"], "value": [100, 10, 15], "available_date": pd.to_datetime(["2024-02-02"] * 3)})
    aligned = add_fundamental_features(align_fundamentals_asof(panel, observations))
    assert pd.isna(aligned.loc[0, "assets"])
    assert aligned.loc[1, "return_on_assets"] == 0.1
    assert aligned.loc[2, "cashflow_to_assets"] == 0.15
