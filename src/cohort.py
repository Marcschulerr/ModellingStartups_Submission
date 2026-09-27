"""
Firm-level base table: identity, circularity and founding-time variables.

Built from companies_lean plus the whitelisted slice of companies_fat.
The two core dimensions: `land` is the sampling-frame country from the
cb_source_file prefix (NOT the HQ location) and `circ`/`circ_subtype`
come from the classifier field cbm_type.

Recodes happen here exactly once:
  - cbm_type == "parse_error" (24 rows) -> circ = 0, circ_subtype = "None"
    (raw cbm_type preserved, `parse_error` flag set)
  - year-precision founding dates -> founding_month_idx imputed to January
    of the founding year (`founding_month_imputed` = 1)

Convention: all boolean flags are stored as int8 0/1 so that parquet, the
CSV mirror and R see identical values.
"""
from __future__ import annotations

import pandas as pd

from .data_loading import load_companies_lean, load_companies_fat_selective

LAND_SET = {
    "UK", "DE", "FR", "ES", "NL", "IT", "SE", "PL", "DK", "CH",
    "FI", "EE", "BE", "NO", "PT", "AT", "LT", "LV", "LU", "IE",
}
CIRC_TYPES = ("Design-based", "Platform-based", "Nature-based", "Service-based", "Waste-based")
COHORT_MIN, COHORT_MAX = 2016, 2022


def month_index(dates: pd.Series) -> pd.Series:
    """Integer month index: year*12 + (month - 1).

    The single month-arithmetic definition used across the pipeline (no
    Period arithmetic). NaT maps to <NA> (nullable Int32).
    """
    dt = pd.to_datetime(dates, errors="raise")
    idx = dt.dt.year.astype("Int32") * 12 + (dt.dt.month.astype("Int32") - 1)
    return idx


def month_index_scalar(date_str: str) -> int:
    """month_index for a single ISO date string."""
    return int(month_index(pd.Series([pd.Timestamp(date_str)])).iloc[0])


# The export batch DE4.csv holds 995 firms with HQ in Switzerland and none in
# Germany, while all other 45 export batches are 100 % HQ-consistent with their
# file prefix (and DE4's .ch-domain share, 20 %, equals the CH batches'): a
# mislabelled CH export. Relabelled surgically by file name; the prefix rule
# stays the definition everywhere else.
SOURCE_FILE_LAND_OVERRIDE = {"DE4.csv": "CH"}


def derive_land(cb_source_file: pd.Series) -> pd.Series:
    """Country code = leading letters of cb_source_file (e.g. 'UK_2018_1' -> 'UK'),
    except the documented mislabelled batches in SOURCE_FILE_LAND_OVERRIDE."""
    land = cb_source_file.str.extract(r"^([A-Za-z]+)", expand=False)
    assert land.notna().all(), "cb_source_file rows without a letter prefix"
    for fname, cc in SOURCE_FILE_LAND_OVERRIDE.items():
        land = land.mask(cb_source_file == fname, cc)
    unknown = set(land.unique()) - LAND_SET
    assert not unknown, f"unexpected land codes: {unknown}"
    return land


def build_firm_base() -> pd.DataFrame:
    """One row per firm (25,421 before any row drop) with identity, circularity,
    founding and team/sector source columns. Everything else is dropped."""
    lean = load_companies_lean()
    fat = load_companies_fat_selective()
    # cbcsv_number_of_founders exists in BOTH exports; keep the fat copy as
    # the single source so the merge stays suffix-free.
    overlap = (set(lean.columns) & set(fat.columns)) - {"crunchbase_url"}
    df = lean.drop(columns=list(overlap)).merge(
        fat, on="crunchbase_url", how="left", validate="1:1")
    assert len(df) == len(lean), "lean/fat merge changed the row count"

    out = pd.DataFrame({"firm_id": df["crunchbase_url"]})
    assert out["firm_id"].is_unique, "crunchbase_url is not unique"
    out["cb_company_name"] = df["cb_company_name"]
    out["cb_source_file"] = df["cb_source_file"]
    out["land"] = derive_land(df["cb_source_file"])

    out["cbm_type"] = df["cbm_type"]
    out["circ"] = df["cbm_type"].isin(CIRC_TYPES).astype("int8")
    out["circ_subtype"] = (df["cbm_type"].str.replace("-based", "", regex=False)
                             .where(out["circ"] == 1, "None").astype("string"))
    out["parse_error"] = (df["cbm_type"] == "parse_error").astype("int8")
    out["cbm_confidence"] = pd.to_numeric(df["cbm_confidence"], errors="coerce").astype("float32")

    founded = pd.to_datetime(df["cbcsv_founded_date"], errors="raise")
    assert founded.notna().all(), "cbcsv_founded_date must be 100% filled"
    out["founded_date"] = founded
    out["founded_precision"] = df["cbcsv_founded_date_precision"]
    assert out["founded_precision"].isin(["day", "month", "year"]).all()

    t_year = founded.dt.year.astype("int16")
    cb_year = pd.to_numeric(df["cb_founded_year"], errors="coerce")
    n_mismatch = int((cb_year.notna() & (cb_year != t_year)).sum())
    assert n_mismatch == 0, f"{n_mismatch} rows: year(cbcsv_founded_date) != cb_founded_year"
    out["t_year"] = t_year

    imputed = out["founded_precision"] == "year"
    fidx = month_index(founded)
    jan_idx = (t_year.astype("Int32") * 12)
    out["founding_month_idx"] = fidx.where(~imputed, jan_idx).astype("int32")
    out["founding_month_imputed"] = imputed.astype("int8")

    out["n_founders"] = pd.to_numeric(df["cbcsv_number_of_founders"], errors="coerce").astype("Int16")
    out["industries"] = df["cbcsv_industries"]
    return out
