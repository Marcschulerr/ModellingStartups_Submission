"""
Build data/processed/firms_analysis.parquet, the single firm-level analysis
dataframe behind both research questions.

Rows: analysis firms only, founding cohorts 2016-2022 AND present in
panel_fat. Both drops are recorded in outputs/tables/cleaning_funnel.csv and
outputs/tables/dropped_panel_absent_firms.csv; data/raw/ keeps everything.

Also writes:
  - data/processed/firms_analysis.csv    (1:1 mirror for R)
  - outputs/tables/cleaning_funnel.csv
  - docs/variables.md                    (data dictionary from src/schema.py)
and finishes with the frozen-artifact reconciliation (comparable outcomes must
equal the frozen reference values firm-by-firm) and the sanity gate.

Pipeline order: scripts/match_cordis.py -> scripts/validate_grant_match.py
finalize -> this build. data/processed/panel_events.parquet is auto-extracted
from the 31 GB panel_fat.csv when missing (force with --extract, ~4 min).

Usage:  conda run -n ml python scripts/build_analysis.py [--extract]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cohort, enrich, grants, outcomes, schema, sectors, survival  # noqa: E402
import sanity_checks  # noqa: E402  (same directory)

PARQUET_PATH = ROOT / "data" / "processed" / "firms_analysis.parquet"
CSV_PATH = ROOT / "data" / "processed" / "firms_analysis.csv"
FUNNEL_PATH = ROOT / "outputs" / "tables" / "cleaning_funnel.csv"
DROPPED_PATH = ROOT / "outputs" / "tables" / "dropped_panel_absent_firms.csv"
VARIABLES_MD_PATH = ROOT / "docs" / "variables.md"
# Frozen artifact (reconciliation reference): a sealed copy of the dataframe
# built from the independent panel_lean source, kept in data/frozen_input/.
FROZEN_PATH = ROOT / "data" / "frozen_input" / "firms_analysis.csv"

# Outcomes whose definition is unchanged vs the frozen artifact: they must match
# firm-by-firm for every retained firm (panel_fat vs panel_lean cross-check).
RECONCILE_COLS = ["n_fund_events_36", "sum_fund_usd_36", "fund36", "success36"]


def cleaning_funnel(base: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Row counts through the two drops + treated-group sizes, circ split."""
    rows = []

    def add(stage, frame, mask=None):
        m = mask if mask is not None else pd.Series(True, index=frame.index)
        rows.append({"stage": stage, "total": int(m.sum()),
                     "circ": int((m & (frame["circ"] == 1)).sum()),
                     "non_circ": int((m & (frame["circ"] == 0)).sum())})

    add("all firms (raw)", base)
    coh = base["t_year"].between(cohort.COHORT_MIN, cohort.COHORT_MAX)
    add("founding cohorts 2016-2022", base, coh)
    add("+ present in panel_fat  (= analysis dataframe)", base, coh & base["in_panel"])
    add("      of which verified EU grant (has_grant)", df, df["has_grant"] == 1)
    sme = df["first_grant_scheme_group"].isin(["SME-Instrument", "EIC"])
    add("      of which SME-Instrument/EIC first grant", df, (df["has_grant"] == 1) & sme)
    return pd.DataFrame(rows)


def reconcile_with_frozen(df: pd.DataFrame) -> None:
    """Retained firms must reproduce the frozen artifact's outcome values
    exactly, which proves the panel_fat source agrees with the panel_lean one."""
    frozen = pd.read_csv(FROZEN_PATH, usecols=["firm_id"] + RECONCILE_COLS)
    m = df[["firm_id"] + RECONCILE_COLS].merge(frozen, on="firm_id",
                                               suffixes=("", "_frozen"), validate="1:1")
    assert len(m) == len(df), "retained firm missing from the frozen artifact"
    problems = []
    for c in RECONCILE_COLS:
        new, old = m[c].astype("float64"), m[f"{c}_frozen"].astype("float64")
        bad = ~np.isclose(new, old, rtol=0, atol=0.01)
        if bad.any():
            problems.append(f"{c}: {int(bad.sum()):,} firms differ "
                            f"(e.g. {m.loc[bad, 'firm_id'].iloc[0]})")
    assert not problems, "frozen-artifact reconciliation FAILED, panel_fat and " \
        f"panel_lean disagree: {problems}"
    print(f"      [ok ] reconciliation vs frozen artifact: "
          f"{', '.join(RECONCILE_COLS)} identical for all {len(m):,} retained firms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true",
                    help="force re-extraction of panel_events.parquet from panel_fat.csv")
    args = ap.parse_args()

    t0 = time.time()
    print("[1/6] firm base + sector controls ...")
    base = cohort.build_firm_base()
    base = sectors.attach_sector_controls(base)
    print(f"      {len(base):,} firms  ({time.time()-t0:.0f}s)")

    print("[2/6] panel events (panel_fat extraction + 36m window aggregation) ...")
    outcomes.extract_panel_events(force=args.extract)
    panel = outcomes.load_panel_events()
    agg = outcomes.aggregate_panel_windows(base, panel)
    base = outcomes.attach_outcomes(base, agg)
    print(f"      panel rows {len(panel):,}  ({time.time()-t0:.0f}s)")

    print("[3/6] EU grant block (verified CORDIS match) + survival snapshot ...")
    base = grants.attach_grants(base)
    base = survival.attach_survival(base)  # outcome-only block, never covariates

    print("[3b/6] founding v2 enrichment (registry/web) ...")
    base = enrich.attach_founding_v2(base)
    base = enrich.attach_v2_outcomes(base, panel)

    print("[4/6] row policy: cohorts 2016-2022 & present in panel_fat ...")
    coh = base["t_year"].between(cohort.COHORT_MIN, cohort.COHORT_MAX)
    absent = coh & ~base["in_panel"]
    dropped = base.loc[absent, ["firm_id", "cb_company_name", "land", "t_year", "circ"]].copy()
    if FROZEN_PATH.exists():
        old = pd.read_csv(FROZEN_PATH, usecols=["firm_id", "fund36", "success36"])
        dropped = dropped.merge(old, on="firm_id", how="left")
    DROPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(DROPPED_PATH, index=False)
    print(f"      dropped: {int((~coh).sum()):,} outside cohorts, "
          f"{len(dropped):,} in-cohort firms absent from panel_fat "
          f"(of which {int(dropped.get('fund36', pd.Series(dtype=float)).fillna(0).sum()):,} "
          f"had fund36=1 in the frozen artifact) -> {DROPPED_PATH.name}")
    df = base[coh & base["in_panel"]].copy()

    last_idx = cohort.month_index_scalar(outcomes.PANEL_LAST_MONTH)
    assert ((df["founding_month_idx"] + outcomes.MAIN_WINDOW - 1) <= last_idx).all(), \
        "incomplete 36m window inside analysis cohorts, impossible by design"
    for c in ("n_fund_events_36", "fund36", "fund36_private", "grant36_cb", "success36"):
        df[c] = df[c].astype("int16" if c == "n_fund_events_36" else "int8")

    print("[5/6] schema order + validation, write artifacts ...")
    funnel = cleaning_funnel(base, df)
    df = df.drop(columns=["in_panel"])
    extra = set(df.columns) - set(schema.expected_columns())
    assert not extra, f"undeclared columns (declare in src/schema.py): {sorted(extra)}"
    df = df[schema.expected_columns()]
    schema.validate(df)
    if os.environ.get("REBASE") == "1":
        print("      !! REBASE=1: frozen-artifact reconciliation SKIPPED "
              "(new data export). A candidate for a new frozen artifact is "
              "written to data/frozen_input/firms_analysis_NEW.csv; rename it "
              "DELIBERATELY after review (procedure in scripts/sanity_checks.py).")
        (ROOT / "data/frozen_input").mkdir(parents=True, exist_ok=True)
        df.to_csv(ROOT / "data/frozen_input/firms_analysis_NEW.csv", index=False)
    else:
        reconcile_with_frozen(df)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    df.to_csv(CSV_PATH, index=False)
    print(f"      wrote {PARQUET_PATH}  ({len(df):,} rows × {len(df.columns)} cols)")
    print(f"      wrote {CSV_PATH}")

    FUNNEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(FUNNEL_PATH, index=False)
    print("\nCleaning funnel:")
    print(funnel.to_string(index=False))

    VARIABLES_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    VARIABLES_MD_PATH.write_text(schema.render_markdown(df))
    print(f"      wrote {VARIABLES_MD_PATH}")

    print("\n[6/6] sanity gate ...")
    sanity_checks.run_all(df, panel=panel)
    print(f"\nBUILD OK  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
