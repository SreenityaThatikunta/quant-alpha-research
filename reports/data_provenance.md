# Data provenance contract

The public Yahoo Finance benchmark is a reproducibility convenience, not a
point-in-time institutional dataset. Its current index-constituent snapshot,
sector assignments, corporate-action handling, and absent delisting returns
make it unsuitable for a production alpha claim.

## Point-in-time production-study requirements

Every daily record must carry the following source-supplied fields before the
run can enable `require_point_in_time_metadata=True`:

| Field | Requirement |
| --- | --- |
| `in_universe` | Explicit historical membership as of the signal date, including securities later delisted |
| `metadata_available_date` | Earliest date the membership/classification was known; it must not be after `date` |
| Sector / industry | Historical classification with an availability timestamp and vendor/source identifier |
| Corporate actions | Vendor methodology and adjustment convention retained with the source snapshot |
| Delisting return | Captured or explicitly modelled for removed securities |
| Source version | Vendor dataset/version, extraction timestamp, license scope, schema, and file hash stored in the run manifest |

The ingestion validator rejects metadata dated after a signal date. It cannot
validate a vendor's historical reconstruction; the source version and data
methodology must therefore be reviewed separately before research results are
promoted.

## Vendor-agnostic historical-universe change log

`run_research.py --universe-history` accepts a CSV or Parquet change log. A
record contains `ticker`, `effective_date`, `metadata_available_date`,
`in_universe`, and the contemporaneous `sector` (with optional vendor fields).
The latest record effective on or before a price date is joined only when its
availability date is also on or before that price date. This interface does
not supply vendor data; it prevents a current membership snapshot from being
mistaken for one.
