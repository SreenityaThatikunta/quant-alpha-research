import pandas as pd

from src.pit_universe import snapshots_to_change_log


def test_snapshots_become_as_of_add_remove_and_sector_change_events():
    snapshots = pd.DataFrame({
        "as_of": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
        "ticker": ["AAA", "BBB", "AAA", "CCC"],
        "gics_sector": ["Technology", "Energy", "Technology", "Health Care"],
        "cik": ["1", "2", "1", "3"],
    })
    history = snapshots_to_change_log(snapshots)
    additions = history.loc[history["in_universe"]]
    removal = history.loc[(history["ticker"] == "BBB") & ~history["in_universe"]].iloc[0]
    assert set(additions.loc[additions["effective_date"] == pd.Timestamp("2020-01-01"), "ticker"]) == {"AAA", "BBB"}
    assert removal["effective_date"] == pd.Timestamp("2020-02-01")
    assert removal["sector"] == "Energy"
    assert (history["effective_date"] == history["metadata_available_date"]).all()


def test_missing_metadata_does_not_create_spurious_membership_events():
    snapshots = pd.DataFrame({
        "as_of": ["2020-01-01", "2020-02-01"], "ticker": ["AAA", "AAA"],
        "gics_sector": [None, None], "cik": [None, None],
    })
    history = snapshots_to_change_log(snapshots)
    assert len(history) == 1
