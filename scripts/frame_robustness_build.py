"""
Frame-selection robustness inputs for analysis/8_frame_robustness.R.

Builds:
  1. frame_entry.csv        per grants_matched_mv row: month of the firm's FIRST
     recorded CB financing event (frame entry) and a follow-on-grant count that
     excludes months attributable to ANY verified CORDIS award (controls can
     never hold an EU follow-on grant by construction, so the reported
     comparison must run on non-EU grant months only).
  2. grants_future_winner.csv   future-winner sets: each pure SME/EIC recipient
     compared with LATER H2020 winners (award ≥ 24 months after the pseudo-award,
     same circ × prior-funding stratum × country group, founding within ±12
     months). Both sides passed the same application/selection filter and both
     entered the frame through an award, which addresses frame-entry selection
     AND grant-seeking self-selection at once. Deterministic (no RNG:
     candidates ranked by founding-month distance, ties by firm_id).
  3. Console counts for the set gate (report the arm only if at least 150
     sets survive).

Inputs:  data/processed/grants_matched_mv.csv   (pipeline step 5, --mv-only)
         data/processed/panel_events.parquet    (via src/outcomes loader)
         outputs/tables/cordis_firm_grants.csv + grant_match_final.csv
           (verified firm×org pairs -> all their project award months)
Outputs: data/processed/frame_entry.csv, data/processed/grants_future_winner.csv
Run:     conda run -n ml python scripts/frame_robustness_build.py
         (after step 5; before analysis/8_frame_robustness.R)

Circular Startup Financing. Master's thesis, 2026.
###############################################################################
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import outcomes  # noqa: E402

MV_CSV = ROOT / "data" / "processed" / "grants_matched_mv.csv"
ENTRY_CSV = ROOT / "data" / "processed" / "frame_entry.csv"
FW_CSV = ROOT / "data" / "processed" / "grants_future_winner.csv"

POST_M = 24          # outcome window [g, g+23], as in build_grants_sample.py
OWN_AWARD_EX = 4     # own-award CB recording window g..g+3 (34% within ±3m)
EU_WINDOW = 3        # CB grant month within ±3m of a CORDIS award = that award
FW_GAP_M = 24        # future winner's own award ≥ pseudo-award + 24m
FW_LAST_AWARD_YEAR = 2020   # H2020 winners only as controls (same programme era)
FW_FOUND_TOL = 12    # founding months within ±12m
FW_K = 4             # up to 4 future-winner controls per set (as in main design)
FW_GATE_SETS = 150   # report the arm only at 150 sets or more

# Country groups for the future-winner arm: exact-land matching leaves only
# 91/453 feasible sets, so lands pool into market regions.
COUNTRY_GROUPS = {
    "DACH": ["DE", "AT", "CH"], "NORDICS": ["SE", "DK", "FI", "NO"],
    "SOUTH": ["ES", "IT", "PT", "GR"], "UK_IE": ["UK", "IE"],
    "FR_BENELUX": ["FR", "BE", "NL", "LU"],
    "EAST": ["PL", "LT", "EE", "CZ", "HU", "SI", "SK", "RO", "BG", "HR", "LV"],
}
LAND_TO_GROUP = {c: g for g, cs in COUNTRY_GROUPS.items() for c in cs}


def month_idx(dates: pd.Series) -> pd.Series:
    """CORDIS date strings -> month index (single definition, src/cohort.py)."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year * 12 + d.dt.month - 1


def main() -> None:
    m = pd.read_csv(MV_CSV)  # pipeline step 5 output (mv treatment definition)

    ev = outcomes.load_panel_events()  # dated monthly events (never end-state cols)
    ev = ev[ev["funding_event"] == 1]
    first_ev = ev.groupby("crunchbase_url")["midx"].min()
    first_grant_ev = (ev[ev["grant_month"] == 1]
                      .groupby("crunchbase_url")["midx"].min())
    grant_arr = {k: g["midx"].to_numpy()
                 for k, g in ev[ev["grant_month"] == 1]
                 .sort_values(["crunchbase_url", "midx"])
                 .groupby("crunchbase_url", observed=True)}
    priv_arr = {k: g["midx"].to_numpy()
                for k, g in ev[ev["private_month"] == 1]
                .sort_values(["crunchbase_url", "midx"])
                .groupby("crunchbase_url", observed=True)}

    # ---- Verified CORDIS award months per firm (for the non-EU grant count) ----
    # cordis_firm_grants.csv is raw firm×project matches; keep only projects of
    # firm×org pairs that survived verification (grant_match_final.csv).
    raw = pd.read_csv(ROOT / "outputs" / "tables" / "cordis_firm_grants.csv",
                      usecols=["firm_id", "organisationID", "ecSignatureDate",
                               "startDate"])
    fin = pd.read_csv(ROOT / "outputs" / "tables" / "grant_match_final.csv",
                      usecols=["firm_id", "organisationID", "final_status"])
    fin = fin[fin["final_status"].isin(["verified_auto", "verified_llm"])]
    raw = raw.merge(fin[["firm_id", "organisationID"]],
                    on=["firm_id", "organisationID"], how="inner")
    # award month = signature month; startDate fallback where signature missing
    raw["award_midx"] = month_idx(raw["ecSignatureDate"])
    raw.loc[raw["award_midx"].isna(), "award_midx"] = month_idx(raw["startDate"])
    raw = raw.dropna(subset=["award_midx"])
    eu_months = {k: g["award_midx"].to_numpy()
                 for k, g in raw.groupby("firm_id", observed=True)}

    # ---- 1. frame_entry.csv: entry month + non-EU follow-on grant count ----
    rows = []
    for r in m.itertuples(index=False):
        g = r.g_midx
        gm = grant_arr.get(r.firm_id, np.empty(0))
        in_win = gm[(gm >= g + OWN_AWARD_EX) & (gm <= g + POST_M - 1)]
        eu = eu_months.get(r.firm_id, np.empty(0))
        if len(eu) and len(in_win):
            near_eu = np.abs(in_win[:, None] - eu[None, :]).min(axis=1) <= EU_WINDOW
            n_noneu = int((~near_eu).sum())
        else:
            n_noneu = len(in_win)
        rows.append((r.set_id, r.firm_id, r.treated,
                     first_ev.get(r.firm_id, np.nan),
                     first_grant_ev.get(r.firm_id, np.nan),
                     len(in_win), n_noneu))
    entry = pd.DataFrame(rows, columns=["set_id", "firm_id", "treated",
                                        "first_ev", "first_grant_ev",
                                        "n_grant_months_ex_check",
                                        "post24_n_grant_months_noneu"])
    # cross-check against the stored pipeline column (same window definition)
    chk = entry.merge(m[["set_id", "firm_id", "post24_n_grant_months_ex"]],
                      on=["set_id", "firm_id"])
    assert (chk["n_grant_months_ex_check"]
            == chk["post24_n_grant_months_ex"]).all(), "grant-month mismatch"
    entry = entry.drop(columns=["n_grant_months_ex_check"])
    entry.to_csv(ENTRY_CSV, index=False)
    print(f"frame_entry.csv: {len(entry):,} rows "
          f"({entry['first_ev'].isna().sum()} without any panel event)")

    # ---- 2. Future-winner sets ----
    s = m[(m["group"] == "sme_eic") & (m["bf_set"] == 0)].copy()
    tl = (s[s["treated"] == 1][["set_id", "land", "g_midx"]]
          .rename(columns={"land": "set_land", "g_midx": "set_g"}))
    s = s.merge(tl, on="set_id")
    # E4: UK/CH awards ≥2021 fall in the CORDIS-unobservable period -> excluded,
    # exactly as in 4_matched_grants.R
    s = s[~(s["set_land"].isin(["UK", "CH"]) & (s["set_g"] // 12 >= 2021))]
    t = s[s["treated"] == 1].copy()
    t["found_midx"] = t["g_midx"] - t["grant_age_m"]
    t["prior"] = (t["pre_n_private"] > 0).astype(int)
    t["grp"] = t["land"].map(LAND_TO_GROUP)
    assert t["grp"].notna().all()

    def pre_n_private_at(firm: str, midx: int) -> int:
        pm = priv_arr.get(firm, np.empty(0))
        return int((pm < midx).sum())

    def any_private_in(firm: str, g0: int) -> int:
        pm = priv_arr.get(firm, np.empty(0))
        return int(((pm >= g0) & (pm <= g0 + POST_M - 1)).any())

    def first_private_offset(firm: str, g0: int) -> float:
        # same >= g convention as build_grants_sample.window_stats (TRANS input)
        pm = priv_arr.get(firm, np.empty(0))
        pm = pm[pm >= g0]
        return float(pm.min() - g0) if len(pm) else np.nan

    fw_rows, n_no_cand = [], 0
    for r in t.itertuples(index=False):
        cand = t[(t["circ"] == r.circ) & (t["grp"] == r.grp)
                 & ((t["found_midx"] - r.found_midx).abs() <= FW_FOUND_TOL)
                 & (t["t_year"] <= FW_LAST_AWARD_YEAR)
                 & (t["firm_id"] != r.firm_id)].copy()
        # future winner: own award ≥ 24m after ITS pseudo-award (same firm age
        # as the focal recipient at award) -> outcome window untreated
        cand["pseudo"] = cand["found_midx"] + r.grant_age_m
        cand = cand[cand["g_midx"] >= cand["pseudo"] + FW_GAP_M]
        # prior-funding stratum evaluated AT the pseudo-award (not at the
        # candidate's own later award: rounds accumulate in between)
        cand["prior_at_pseudo"] = [
            int(pre_n_private_at(f, p) > 0)
            for f, p in zip(cand["firm_id"], cand["pseudo"])]
        cand = cand[cand["prior_at_pseudo"] == r.prior]
        if not len(cand):
            n_no_cand += 1
            continue
        cand["dist"] = (cand["found_midx"] - r.found_midx).abs()
        cand = cand.sort_values(["dist", "firm_id"]).head(FW_K)
        fw_id = f"fw_{r.set_id}"
        fw_rows.append(dict(fw_set_id=fw_id, firm_id=r.firm_id, treated=1,
                            land=r.land, grp=r.grp, circ=r.circ, prior=r.prior,
                            eval_midx=r.g_midx, own_award_midx=r.g_midx,
                            post24_any_private=r.post24_any_private,
                            pre_n_private=r.pre_n_private,
                            post_first_private_offset_m=r.post_first_private_offset_m))
        for c in cand.itertuples(index=False):
            fw_rows.append(dict(
                fw_set_id=fw_id, firm_id=c.firm_id, treated=0, land=c.land,
                grp=c.grp, circ=c.circ, prior=r.prior, eval_midx=int(c.pseudo),
                own_award_midx=c.g_midx,
                post24_any_private=any_private_in(c.firm_id, int(c.pseudo)),
                pre_n_private=pre_n_private_at(c.firm_id, int(c.pseudo)),
                post_first_private_offset_m=first_private_offset(
                    c.firm_id, int(c.pseudo))))
    fw = pd.DataFrame(fw_rows)
    n_sets = fw["fw_set_id"].nunique()
    fw.to_csv(FW_CSV, index=False)
    by_prior = fw[fw["treated"] == 1].groupby("prior").size().to_dict()
    by_circ = fw[fw["treated"] == 1].groupby("circ").size().to_dict()
    reuse = fw[fw["treated"] == 0].groupby("firm_id").size()
    print(f"grants_future_winner.csv: {n_sets} sets ({len(fw):,} rows) | "
          f"prior split {by_prior} | circ split {by_circ} | "
          f"controls reused: {(reuse > 1).sum()} of {reuse.size} serve >1 set")
    print(f"set gate (>= {FW_GATE_SETS}): "
          f"{'PASS - report the arm' if n_sets >= FW_GATE_SETS else 'FAIL - drop the arm'}")


if __name__ == "__main__":
    main()
