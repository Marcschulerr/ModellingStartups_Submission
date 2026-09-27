"""
Hard sanity gate for data/processed/firms_analysis.parquet.

Every number is pinned against independently computed values: the frozen
reference artifact, a streaming audit of the raw panel and the verified grant
match. A red gate means the pipeline changed behaviour. Fix the pipeline or
investigate; never loosen a pin silently.

Anchor note: the reference anchors (jan-window fund36 64.5/70.8, success36
13.6/12.9; precision-aware 66.0/72.2 and 15.6/14.5) were computed on all
25,421 firms. The build drops non-analysis rows (cohorts and panel-absent,
funnel-recorded), so the pins below are re-based on the 22,039 analysis firms.
The full-population values are printed alongside for reference; the deltas
stay within 0.2 pp throughout.

Three layers of outcome verification:
  1. internal consistency (nesting, log identities),
  2. full recompute from panel_events.parquet must reproduce every outcome column,
  3. cross-source: a seeded 2,000-firm sample recomputed from RAW panel_lean.csv
     must match (guards against panel_fat/panel_lean divergence),
plus the build-time reconciliation against the frozen reference artifact.

Loading a NEW data export: run REBASE=1 SANITY_REPORT=1 sh run_pipeline.sh.
REBASE=1 lifts the frozen-artifact reconciliation in the build and the
stopifnot pins in the R scripts (loud banner); SANITY_REPORT=1 prints every
deviation from the pins here instead of aborting. Review the deviations,
update the constants at the top of this file and the stopifnot lines in the
analysis/*.R headers, then promote data/frozen_input/firms_analysis_NEW.csv
to firms_analysis.csv. After that everything runs without flags again,
hard-pinned to the new state. If panel_fat.csv changed, re-extract the event
slice once: conda run -n ml python scripts/build_analysis.py --extract

Usage:  conda run -n ml python scripts/sanity_checks.py       (re-reads parquet)
        sanity_checks.run_all(df, panel=...)                  (from build)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import outcomes, schema  # noqa: E402

PARQUET_PATH = ROOT / "data" / "processed" / "firms_analysis.parquet"

EXPECTED_N = 22_039
# DE4.csv is a mislabelled CH export batch: 835 analysis firms relabelled
# DE to CH (DE 3179 to 2344, CH 515 to 1350). A documented pin update, not a
# loosening; all other lands unchanged.
EXPECTED_LAND = {
    "UK": 6606, "DE": 2344, "FR": 2795, "ES": 1712, "NL": 1146, "IT": 1078,
    "SE": 1045, "PL": 602, "DK": 537, "CH": 1350, "FI": 459, "EE": 451,
    "BE": 433, "NO": 408, "PT": 302, "AT": 282, "LT": 211, "LV": 123,
    "LU": 116, "IE": 39,
}
EXPECTED_SUBTYPES = {"Design": 1067, "Platform": 970, "Nature": 783,
                     "Service": 727, "Waste": 450}
EXPECTED_CIRC = 3_997
EXPECTED_PARSE_ERROR = 22
EXPECTED_PRECISION = {"year": 12217, "day": 7385, "month": 2437}

ANCHOR_YEARS = (2016, 2021)
# re-based pins (round-4 exact) + the full-population reference for the printout
ANCHORS_JAN = {"fund36_circ": 0.6443, "fund36_noncirc": 0.7079,
               "success36_circ": 0.1362, "success36_noncirc": 0.1293}
ANCHORS_MAIN = {"fund36_circ": 0.6588, "fund36_noncirc": 0.7225,
                "success36_circ": 0.1558, "success36_noncirc": 0.1450}
ANCHORS_FULL_JAN = {"fund36_circ": 0.645, "fund36_noncirc": 0.708,
                      "success36_circ": 0.136, "success36_noncirc": 0.129}
ANCHORS_FULL_MAIN = {"fund36_circ": 0.660, "fund36_noncirc": 0.722,
                       "success36_circ": 0.156, "success36_noncirc": 0.145}

EXPECTED_TTFR_PRIVATE = (18, 35, 54)   # P50/P75/P90 months to first PRIVATE round
# 267 pre-founding funding events among analysis firms (CB artifact, all within
# the founding year, excluded by the window definition). Full 25,421-firm base: 336.
EXPECTED_PRE_FOUNDING_EVENTS = 267

# Two documented pin updates behind the current values, never silent loosenings:
# 1. Musa <-> MUSA scarl rejected as a wrong entity (append-only verdict in
#    the llm store): treated 1,554 to 1,553, consortium 770 to 769.
# 2. The DE4.csv relabel (DE to CH) makes the 835 Swiss firms matchable against
#    CORDIS country CH (+60 verified pairs); matcher extensions (accent folding,
#    UAB/SIA/OU/LDA legal forms, SME=true beyond activityType PRC) and the
#    short-name review round (26 pairs: 24 confirmed, Caura/Nido rejected)
#    lift treated 1,553 to 1,661 (circ 523 to 559), scheme groups 769/488/296
#    to 827/533/301, mono 777 to 823. All 82 new verdicts are append-only in
#    the llm store.
EXPECTED_GRANT_FIRMS = 1_661
EXPECTED_GRANT_CIRC = 559
EXPECTED_GRANT_UNCERTAIN = 16
EXPECTED_GRANT_SCHEME = {"consortium-R&D": 827, "SME-Instrument": 533, "EIC": 301}
EXPECTED_GRANT_MONO_FIRST = 823

LEAN_SAMPLE_SEED = 20260711
LEAN_SAMPLE_N = 2_000
EXPECTED_LEAN_ROWS = 2_073_807


class GateFailure(AssertionError):
    pass


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def _rates(n_events, max_month, n_ma, circ, t_year) -> dict[str, float]:
    m = t_year.between(*ANCHOR_YEARS)
    fund = (n_events >= 1)
    success = ((n_events >= outcomes.SUCCESS_MIN_EVENTS)
               | (max_month > outcomes.SUCCESS_MIN_MAX_MONTH_USD)
               | (n_ma >= 1))
    return {
        "fund36_circ": float(fund[m & (circ == 1)].mean()),
        "fund36_noncirc": float(fund[m & (circ == 0)].mean()),
        "success36_circ": float(success[m & (circ == 1)].mean()),
        "success36_noncirc": float(success[m & (circ == 0)].mean()),
    }


def check_recompute(df: pd.DataFrame, panel: pd.DataFrame, failures: list[str]) -> None:
    """Layer 2: every outcome column must reproduce from the events parquet."""
    agg = outcomes.aggregate_panel_windows(df[["firm_id", "founding_month_idx"]], panel)
    r = outcomes.attach_outcomes(df[["firm_id", "founding_month_idx"]].copy(), agg)
    ok = True
    for c in ["n_fund_events_36", "n_private_months_36", "fund36", "fund36_private",
              "grant36_cb", "sum_fund_usd_36", "logfund36", "sum_fund_usd_private_36",
              "logfund36_private", "success36", "success36_bigmonth", "success36_ma"]:
        a = pd.to_numeric(df[c]).to_numpy(dtype="float64")
        b = pd.to_numeric(r[c]).to_numpy(dtype="float64")
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            ok = False
            failures.append(f"recompute mismatch in {c}")
    off_a = df["first_private_offset_m"].astype("Float64")
    off_b = r["first_private_offset_m"].astype("Float64")
    ok &= bool(off_a.fillna(-1).eq(off_b.fillna(-1)).all())
    _check(ok, "all outcome columns reproduce from panel_events.parquet", failures)

    # v2 layer: the v2 outcome columns reproduce under the v2 founding month;
    # the baseline pins above stay untouched by construction.
    if "founding_month_idx_v2" in df.columns:
        from src import enrich
        r2 = enrich.attach_v2_outcomes(
            df[["firm_id", "founding_month_idx", "founding_month_idx_v2",
                "fund36_private"]].copy(), panel)
        ok2 = True
        for c in ["fund36_private_v2", "grant36_cb_v2", "sum_fund_usd_private_36_v2",
                  "logfund36_private_v2", "success36_v2"]:
            a = pd.to_numeric(df[c]).fillna(-1).to_numpy(dtype="float64")
            b = pd.to_numeric(r2[c]).fillna(-1).to_numpy(dtype="float64")
            if not np.allclose(a, b, rtol=0, atol=1e-9):
                ok2 = False
                failures.append(f"v2 recompute mismatch in {c}")
        ok2 &= bool(df["first_private_offset_m_v2"].astype("Float64").fillna(-1)
                    .eq(r2["first_private_offset_m_v2"].astype("Float64").fillna(-1)).all())
        _check(ok2, "v2 outcome columns reproduce under founding_month_idx_v2", failures)


def check_lean_cross_source(df: pd.DataFrame, failures: list[str]) -> None:
    """Layer 3: seeded sample recomputed from RAW panel_lean must agree."""
    from src.data_loading import load_panel_lean
    lean = load_panel_lean(columns=["crunchbase_url", "month", "funding_event"])
    _check(len(lean) == EXPECTED_LEAN_ROWS,
           f"panel_lean rows == {EXPECTED_LEAN_ROWS:,} (raw drift guard)", failures)
    rng = np.random.default_rng(LEAN_SAMPLE_SEED)
    sample = df.sample(n=LEAN_SAMPLE_N, random_state=int(rng.integers(2**31)))
    fmap = sample.set_index("firm_id")["founding_month_idx"].astype("int32")
    sub = lean[lean["crunchbase_url"].isin(set(fmap.index))].copy()
    from src.cohort import month_index
    sub["midx"] = month_index(sub["month"]).astype("int32")
    off = sub["midx"] - sub["crunchbase_url"].map(fmap).astype("int32")
    win = sub[(off >= 0) & (off <= outcomes.MAIN_WINDOW - 1)]
    n_lean = (win.groupby("crunchbase_url", observed=True)["funding_event"]
              .sum().astype("int64"))
    got = sample.set_index("firm_id")["n_fund_events_36"].astype("int64")
    both = pd.concat([got, n_lean.reindex(got.index).fillna(0).astype("int64")], axis=1,
                     keys=["fat", "lean"])
    _check(bool(both["fat"].eq(both["lean"]).all()),
           f"cross-source check: {LEAN_SAMPLE_N:,}-firm seeded sample identical in "
           f"panel_lean (seed {LEAN_SAMPLE_SEED})", failures)


def check_anchors(df: pd.DataFrame, panel: pd.DataFrame, failures: list[str]) -> None:
    firms_jan = df[["firm_id", "t_year", "circ"]].copy()
    firms_jan["jan_idx"] = (firms_jan["t_year"].astype("int32") * 12)
    agg = outcomes.aggregate_panel_windows(firms_jan, panel, fidx_col="jan_idx")
    j = firms_jan.merge(agg, left_on="firm_id", right_index=True, how="left")
    for c in ("n_fund_events_36", "max_month_fund_usd_36", "n_ma_36"):
        j[c] = j[c].fillna(0)
    jan = _rates(j["n_fund_events_36"], j["max_month_fund_usd_36"], j["n_ma_36"],
                 j["circ"], j["t_year"])

    agg2 = outcomes.aggregate_panel_windows(df[["firm_id", "founding_month_idx"]], panel)
    k = df[["firm_id", "t_year", "circ"]].merge(agg2, left_on="firm_id",
                                                right_index=True, how="left")
    main = _rates(k["n_fund_events_36"], k["max_month_fund_usd_36"], k["n_ma_36"],
                  k["circ"], k["t_year"])

    print("\nAnchor check (cohorts 2016-21; pins re-based on analysis rows):")
    print(f"  {'rate':<18} {'full(25,421)':>14} {'pin(22,039)':>12} "
          f"{'jan-replication':>16} {'precision-aware':>16}")
    ok = True
    for kk in ANCHORS_JAN:
        print(f"  {kk:<18} {ANCHORS_FULL_JAN[kk]:>14.1%} {ANCHORS_JAN[kk]:>12.2%} "
              f"{jan[kk]:>16.2%} {main[kk]:>16.2%}")
        if round(jan[kk], 4) != ANCHORS_JAN[kk]:
            ok = False
            failures.append(f"jan anchor {kk}: {jan[kk]:.4f} != pin {ANCHORS_JAN[kk]:.4f}")
        if round(main[kk], 4) != ANCHORS_MAIN[kk]:
            ok = False
            failures.append(f"main anchor {kk}: {main[kk]:.4f} != pin {ANCHORS_MAIN[kk]:.4f}")
    if ok:
        print("  [ok ] all anchor rates equal their re-based pins (round-4 exact)")


def report_tables(df: pd.DataFrame) -> None:
    circ = df["circ"] == 1
    print("\nFunding rates (analysis rows, cohorts 2016-2022):")
    for c in ("fund36", "fund36_private", "grant36_cb", "success36", "has_grant"):
        print(f"  {c:<15} circ {df.loc[circ, c].mean():6.1%}   "
              f"non-circ {df.loc[~circ, c].mean():6.1%}")
    hg = df[df["has_grant"] == 1]
    print("\nFirst-grant scheme groups (verified match):")
    print(hg.groupby(["first_grant_scheme_group"])
            .agg(firms=("firm_id", "count"), circ=("circ", "sum"),
                 mono=("first_grant_mono", "mean"),
                 median_eur=("ec_total_eur", "median")).round(2).to_string())
    fp = df.loc[df["first_private_offset_m"].notna(), "first_private_offset_m"]
    q = fp.quantile([0.5, 0.75, 0.9]).astype(int)
    print(f"\nTime to first PRIVATE round (months from founding, uncensored, "
          f"n={len(fp):,}): P50={q[0.5]}  P75={q[0.75]}  P90={q[0.9]}")


def run_all(df: pd.DataFrame, panel: pd.DataFrame | None = None) -> None:
    failures: list[str] = []
    print("Hard checks:")

    try:
        schema.validate(df)
        _check(True, "schema: exact closed column list", failures)
    except AssertionError as e:
        _check(False, f"schema: {e}", failures)

    _check(len(df) == EXPECTED_N, f"N == {EXPECTED_N:,} (got {len(df):,})", failures)
    _check(df["firm_id"].is_unique, "firm_id unique", failures)
    _check(bool(df["t_year"].between(2016, 2022).all()), "t_year ∈ [2016, 2022]", failures)
    _check(df["land"].value_counts().to_dict() == EXPECTED_LAND,
           "land counts match pinned values exactly", failures)
    sub = df.loc[df["circ"] == 1, "circ_subtype"].value_counts().to_dict()
    _check(sub == EXPECTED_SUBTYPES, f"circ_subtype counts match (got {sub})", failures)
    _check(bool((df.loc[df["circ"] == 0, "circ_subtype"] == "None").all()),
           "non-circular ⇒ circ_subtype == 'None'", failures)
    _check(int((df["circ"] == 1).sum()) == EXPECTED_CIRC, f"circ == {EXPECTED_CIRC:,}", failures)
    _check(int((df["parse_error"] == 1).sum()) == EXPECTED_PARSE_ERROR,
           f"parse_error == {EXPECTED_PARSE_ERROR} (coded non-circular)", failures)
    _check(df["founded_precision"].value_counts().to_dict() == EXPECTED_PRECISION,
           "founded-date precision counts match", failures)

    tagged = df["industries"].notna()
    _check(bool((df.loc[tagged, "n_industries"] >= 1).all())
           and bool(df.loc[~tagged, "n_industries"].isna().all()),
           "n_industries consistent with raw industries string", failures)
    ind_cols = list(schema.TOP_INDUSTRY_TAGS.values())
    _check(bool(df[ind_cols].isin([0, 1]).all().all())
           and bool((df.loc[~tagged, ind_cols] == 0).all().all()),
           "industry dummies are 0/1 and 0 without tags", failures)

    _check(bool((df["fund36"] == (df["n_fund_events_36"] >= 1)).all()),
           "fund36 == 1[n_fund_events_36 ≥ 1]", failures)
    _check(bool((df["fund36_private"] <= df["fund36"]).all()),
           "fund36_private ⊆ fund36", failures)
    _check(bool((df["grant36_cb"] <= df["fund36"]).all()), "grant36_cb ⊆ fund36", failures)
    _check(bool(np.allclose(df["logfund36"], np.log1p(df["sum_fund_usd_36"])))
           and bool(np.allclose(df["logfund36_private"],
                                np.log1p(df["sum_fund_usd_private_36"]))),
           "log outcomes == log1p(sums)", failures)
    _check(bool((df["sum_fund_usd_private_36"] <= df["sum_fund_usd_36"] + 1e-9).all()),
           "private amounts ≤ total amounts", failures)
    priv = df["fund36_private"] == 1
    _check(bool((df.loc[priv, "first_private_offset_m"] <= 35).all()),
           "fund36_private == 1 ⇒ first private round inside the window", failures)
    _check(bool((df["success36"] == ((df["n_fund_events_36"] >= 3)
                                     | (df["success36_bigmonth"] == 1)
                                     | (df["success36_ma"] == 1)).astype("int8")).all()),
           "success36 == OR of its three components", failures)

    _check(int(df["has_grant"].sum()) == EXPECTED_GRANT_FIRMS,
           f"verified grant firms == {EXPECTED_GRANT_FIRMS:,}", failures)
    _check(int(df.loc[df["circ"] == 1, "has_grant"].sum()) == EXPECTED_GRANT_CIRC,
           f"circular grant firms == {EXPECTED_GRANT_CIRC}", failures)
    _check(int(df["has_grant_uncertain"].sum()) == EXPECTED_GRANT_UNCERTAIN,
           f"uncertain-only grant firms == {EXPECTED_GRANT_UNCERTAIN} (untreated)", failures)
    hg = df["has_grant"] == 1
    _check(bool((df.loc[hg, "grant_match_tier"].notna()).all())
           and bool((df.loc[~hg, "grant_match_tier"].isna()).all()),
           "grant_match_tier present exactly for treated firms", failures)
    _check(bool((~(df["has_grant"].astype(bool) & df["has_grant_uncertain"].astype(bool))).all()),
           "has_grant and has_grant_uncertain mutually exclusive", failures)
    _check(bool(df.loc[hg, "first_grant_date"].notna().all()),
           "every treated firm has a dated first grant", failures)
    got_scheme = df.loc[hg, "first_grant_scheme_group"].value_counts().to_dict()
    _check(got_scheme == EXPECTED_GRANT_SCHEME,
           f"first-grant scheme groups match (got {got_scheme})", failures)
    _check(int(df.loc[hg, "first_grant_mono"].sum()) == EXPECTED_GRANT_MONO_FIRST,
           f"mono-beneficiary first grants == {EXPECTED_GRANT_MONO_FIRST}", failures)

    fp = df.loc[df["first_private_offset_m"].notna(), "first_private_offset_m"]
    q = fp.quantile([0.5, 0.75, 0.9]).astype(int)
    got_ttfr = (int(q[0.5]), int(q[0.75]), int(q[0.9]))
    _check(got_ttfr == EXPECTED_TTFR_PRIVATE,
           f"time-to-first-private-round P50/P75/P90 == {EXPECTED_TTFR_PRIVATE} "
           f"(got {got_ttfr})", failures)

    if panel is None:
        print("\nLoading panel events (standalone mode) ...")
        panel = outcomes.load_panel_events()

    # Known Crunchbase artifact: funding events dated BEFORE the official founding
    # date (all within the founding year). The window definition excludes them.
    fmap = df.set_index("firm_id")["founding_month_idx"].astype("int32")
    sub_p = panel[panel["crunchbase_url"].isin(set(fmap.index))]
    off = sub_p["midx"].astype("int32") - sub_p["crunchbase_url"].map(fmap).astype("int32")
    n_pre = int(sub_p.loc[off < 0, "funding_event"].sum())
    _check(n_pre == EXPECTED_PRE_FOUNDING_EVENTS,
           f"pre-founding events == {EXPECTED_PRE_FOUNDING_EVENTS} "
           f"(CB-artifact pin, analysis firms; got {n_pre})", failures)

    check_recompute(df, panel, failures)
    check_anchors(df, panel, failures)
    check_lean_cross_source(df, failures)
    report_tables(df)

    import os
    if failures and os.environ.get("SANITY_REPORT") == "1":
        print("\n!! SANITY_REPORT=1: report mode, pins are NOT enforced "
              "(only for a new data export; normal runs must not set this flag).")
        print(f"   {len(failures)} deviation(s) from the pinned values:")
        for f in failures:
            print(f"   - {f}")
        print("   -> review the values and DELIBERATELY re-pin them in "
              "scripts/sanity_checks.py (procedure in this file's docstring).")
        return
    if failures:
        print(f"\nSANITY GATE RED: {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        raise GateFailure(f"{len(failures)} sanity check(s) failed")
    print("\nSANITY GATE GREEN: all checks passed.")


if __name__ == "__main__":
    run_all(pd.read_parquet(PARQUET_PATH), panel=None)
