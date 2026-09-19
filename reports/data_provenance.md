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
