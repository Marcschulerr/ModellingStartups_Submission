"""
Survival and exit block: firm status at the data cutoff (export 2026-03).

Scoped exception to the end-state leakage ban: the columns built here are
usable ONLY as survival/exit outcomes or for validation, never as covariates
or outcomes in the funding models. Template: Kennedy et al. (2024, Nature
Energy) exit-vs-failure outcomes on a funded-only frame. Documented deviation:
status-at-cutoff flags instead of dated Cox, because 62 % of Crunchbase
closures carry no closed date.

Dated M&A events (ma_any_dated / first_ma_offset_m) come from panel_events
via src/outcomes.py; only the undated snapshot flags live here.
"""
from __future__ import annotations

import pandas as pd

from .data_loading import load_companies_lean, load_companies_fat_selective, SURVIVAL_COLS


def attach_survival(base: pd.DataFrame) -> pd.DataFrame:
    """Merge the three snapshot survival columns onto the firm base.

    status_closed      1 = cb_operating_status 'Closed' at export (100% filled)
    exit_any           1 = 'Was Acquired' or IPO status Public/Delisted
    site_dead_2026_03  1 = website scrape failed on 2026-03-27 (jina), 0 = ok,
                       <NA> = no website on file / skipped; validation proxy:
                       failed-share is 53.6 % among Closed vs 21.4 % among Active
    """
    lean = load_companies_lean()[["crunchbase_url", "cb_operating_status"]]
    fat = load_companies_fat_selective(SURVIVAL_COLS)
    snap = lean.merge(fat, on="crunchbase_url", how="left", validate="1:1")

    status = snap["cb_operating_status"]
    assert status.isin(["Active", "Closed"]).all(), \
        f"unexpected cb_operating_status values: {set(status.unique()) - {'Active', 'Closed'}}"
    snap["status_closed"] = (status == "Closed").astype("int8")

    ipo = snap["cbcsv_ipo_status"]
    assert ipo.isin(["Private", "Public", "Delisted"]).all(), \
        f"unexpected cbcsv_ipo_status values: {set(ipo.unique()) - {'Private', 'Public', 'Delisted'}}"
    acquired = snap["cbcsv_acquisition_status"].fillna("").str.contains("Was Acquired")
    snap["exit_any"] = (acquired | ipo.isin(["Public", "Delisted"])).astype("int8")

    scrape = snap["jina_scrape_status"]
    site_dead = pd.Series(pd.NA, index=snap.index, dtype="Int8")
    site_dead[scrape == "failed"] = 1
    site_dead[scrape == "done"] = 0
    snap["site_dead_2026_03"] = site_dead

    out = base.merge(
        snap[["crunchbase_url", "status_closed", "exit_any", "site_dead_2026_03"]],
        left_on="firm_id", right_on="crunchbase_url", how="left", validate="1:1",
    ).drop(columns=["crunchbase_url"])
    assert len(out) == len(base), "survival merge changed the row count"
    assert out["status_closed"].notna().all(), "status_closed must be 100% filled"
    return out
