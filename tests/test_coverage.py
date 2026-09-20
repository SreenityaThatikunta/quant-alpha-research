import pandas as pd

from src.coverage import universe_price_coverage_audit


def test_price_coverage_audit_preserves_missing_historical_members():
    history = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "effective_date": ["2020-01-01", "2020-01-01"], "in_universe": [True, True],
    })
    prices = pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-03"]})
    audit = universe_price_coverage_audit(history, prices).set_index("ticker")
    assert audit.loc["BBB", "missing_all_prices"]
    assert audit.loc["AAA", "price_starts_after_membership"]
