"""Display CSVs for thesis Appendix 3-12.

Builds every appendix table in its final display form (cells
'-2.42** [-4.14, -0.70]', unicode minus, stars *p<.05 **p<.01 ***p<.001,
differences from unrounded values) so the tables rebuild 1:1 in Word.
Numbers come ONLY from the canonical estimate CSVs (analysis/*.R,
scripts/sequencing_stats.py, scripts/grants_predynamics.py) and read-only
tabulations of processed files; nothing here estimates anything new.

Output: outputs/tables/appendix/appendix{03..12}*.csv
Run:    conda run -n ml python scripts/appendix_tables.py
        (pipeline step 9, AFTER the R scripts; reads only estimate CSVs
         + processed/registry tables)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/tables/appendix"
OUT.mkdir(parents=True, exist_ok=True)
M = "−"  # unicode minus, table convention


def num(x, dec=2):
    s = f"{x:.{dec}f}"
    return s.replace("-", M)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def snum(x, dec=2):
    # signed number: explicit + on positives (pp-difference convention)
    return ("+" if x > 0 else "") + num(x, dec)


def cell(est, se, p, scale=100, dec=2):
    lo, hi = est - 1.96 * se, est + 1.96 * se
    return (f"{snum(est * scale, dec)}{stars(p)} "
            f"[{snum(lo * scale, dec)}, {snum(hi * scale, dec)}]")


def hrcell(loghr, se, p, dec=3):
    return (f"{num(np.exp(loghr), dec)}{stars(p)} "
            f"[{num(np.exp(loghr - 1.96 * se), dec)}, "
            f"{num(np.exp(loghr + 1.96 * se), dec)}]")


def row_of(df, run_id, term=None):
    r = df[df.run_id == run_id]
    if term is not None:
        r = r[r.term == term]
    assert len(r) == 1, f"{run_id}/{term}: {len(r)} rows"
    return r.iloc[0]


def pct(x, dec=1):
    return num(100 * x, dec)


timing = pd.read_csv(ROOT / "outputs/estimates/timing_estimates.csv")
grants = pd.read_csv(ROOT / "outputs/estimates/grant_estimates.csv")
surv = pd.read_csv(ROOT / "outputs/estimates/survival_estimates.csv")
seq = pd.read_csv(ROOT / "outputs/estimates/sequencing_stats.csv")
pdyn = pd.read_csv(ROOT / "outputs/estimates/grants_predynamics_mv.csv")
fg = pd.read_csv(ROOT / "outputs/estimates/grants_fourgroup_km.csv")
frame = pd.read_csv(ROOT / "outputs/estimates/frame_robustness.csv")
fetab = pd.read_csv(ROOT / "outputs/estimates/frame_entry_tabulation.csv")

FIRM_COLS = ["land", "t_year", "circ", "circ_subtype", "founding_v2_source",
             "founding_month_imputed_v2", "fund36_private_v2", "grant36_cb_v2",
             "has_grant", "first_grant_scheme_group", "first_grant_scheme",
             "status_closed", "registry_dissolution_date", "n_founders",
             "n_industries", "firm_id", "ec_total_eur"]
firms = pd.read_parquet(ROOT / "data/processed/firms_analysis.parquet",
                        columns=FIRM_COLS)
assert len(firms) == 22039
mv = firms[firms.founding_month_imputed_v2 == 0]
assert len(mv) == 16615

# ---------------------------------------------------------------- Appendix 3
REGISTER = {
    "UK": "Companies House; Orbis (residuals)", "FR": "Sirene; Orbis (residuals)",
    "BE": "KBO/CBE Open Data", "DK": "CVR", "EE": "e-Business Register",
    "FI": "PRH", "LV": "Register of Enterprises (UR)",
    "NO": "Brønnøysund Register", "DE": "OpenCorporates; Orbis",
    "IE": "OpenCorporates", "LU": "OpenCorporates", "ES": "Orbis", "SE": "Orbis",
    "NL": "Orbis", "IT": "Orbis", "CH": "Orbis", "PT": "Orbis", "AT": "Orbis",
    "LT": "—", "PL": "—",
}
src = firms.founding_v2_source
kind = np.select(
    [src.isin(["cb_day", "cb_month"]), src.str.startswith("registry_"),
     src == "web", src == "january"],
    ["cb", "register", "web", "not_verified"], default="?")
a3 = (pd.DataFrame({"land": firms.land, "kind": kind})
      .pivot_table(index="land", columns="kind", aggfunc=len, fill_value=0))
a3["firms"] = a3.sum(axis=1)
a3 = a3.reset_index()
a3["register_name"] = a3.land.map(REGISTER)
a3["verified_pct"] = 100 * (1 - a3.not_verified / a3.firms)
tot = {"land": "Total", "register_name": "",
       "firms": a3.firms.sum(), "cb": a3.cb.sum(), "register": a3.register.sum(),
       "web": a3.web.sum(), "not_verified": a3.not_verified.sum()}
tot["verified_pct"] = 100 * (1 - tot["not_verified"] / tot["firms"])
assert tot["cb"] == 9822 and tot["register"] == 6161 and tot["web"] == 632
assert tot["not_verified"] == 5424
a3 = pd.concat([a3, pd.DataFrame([tot])], ignore_index=True)
a3out = pd.DataFrame({
    "Country": a3.land, "Register": a3.register_name,
    "Firms in panel": a3.firms.map("{:,}".format),
    "Month from Crunchbase": a3.cb.map("{:,}".format),
    "Month from register": a3.register.map("{:,}".format),
    "Month from website": a3.web.map("{:,}".format),
    "Not verified": a3.not_verified.map("{:,}".format),
    "Verified (%)": a3.verified_pct.map(lambda x: num(x, 1)),
})
a3out.to_csv(OUT / "appendix03_founding_sources.csv", index=False)

# ---------------------------------------------------------------- Appendix 4
gm = pd.read_csv(ROOT / "outputs/tables/grant_match_final.csv")
fs = gm.final_status.value_counts()
routes = gm.route.value_counts()
mv_grant = mv[mv.has_grant == 1]
sg = mv_grant.first_grant_scheme_group.value_counts()
comp = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                   usecols=["crunchbase_url", "cbcsv_closed_date"],
                   low_memory=False).rename(columns={"crunchbase_url": "firm_id"})
fc = firms.merge(comp, on="firm_id", how="left")
# register dissolution dates from the VERIFIED register-match store; the
# analysis column registry_dissolution_date carries only the adopted subset,
# the 3.2 closure-dating sentence (589 of 1,085) counts every verified match
reg = pd.read_csv(ROOT / "data/scrape/registry/registry_match_final.csv",
                  low_memory=False)
regd = (reg[reg.dissolution_date.notna()][["firm_id"]].drop_duplicates()
        .assign(reg_diss=1))
fc = fc.merge(regd, on="firm_id", how="left")
mc = fc[fc.founding_month_imputed_v2 == 0]


def closure_cells(d):
    cl = d.status_closed.astype(bool)
    cb_dated = int((cl & d.cbcsv_closed_date.notna()).sum())
    dated = int((cl & (d.cbcsv_closed_date.notna()
                       | (d.reg_diss == 1))).sum())
    return int(cl.sum()), cb_dated, dated, int((d.reg_diss == 1).sum())


closed_full, cbdated_full, regdated_full, regall_full = closure_cells(fc)
closed_mv, cbdated_mv, regdated_mv, regall_mv = closure_cells(mc)
# 3.2 pins: 589/1,085 dated in the estimation sample; 698 / 1,395 full panel
assert (regdated_mv, closed_mv, regdated_full, regall_full) == (
    589, 1085, 698, 1395)
a4a = pd.DataFrame([
    ("Candidate firm–organization pairs (five routes)", f"{len(gm):,}"),
    ("Confirmed by rule", f"{fs['verified_auto']:,}"),
    ("Confirmed in review", f"{fs['verified_llm']:,}"),
    ("Rejected", f"{fs['rejected']:,}"),
    ("Uncertain, excluded from both pools", f"{fs['uncertain']:,}"),
    ("Linked firms in the full panel", f"{int(firms.has_grant.sum()):,}"),
    ("Linked firms in the estimation sample", f"{int(mv.has_grant.sum()):,}"),
    ("   first grant SME Instrument", f"{int(sg['SME-Instrument']):,}"),
    ("   first grant EIC", f"{int(sg['EIC']):,}"),
    ("   first grant consortium-type action",
     f"{int(sg['consortium-R&D']):,}"),
    ("Audit: confirmed pairs re-checked", "30"),
    ("Audit: disagreements", "1"),
], columns=["Step", "Pairs or firms"])
a4a.to_csv(OUT / "appendix04a_linkage_funnel.csv", index=False)
a4b = pd.DataFrame([
    ("Firms listed as closed by Crunchbase", f"{closed_full:,}", f"{closed_mv:,}"),
    ("   dated by Crunchbase", f"{cbdated_full:,}", f"{cbdated_mv:,}"),
    ("   dated after register linkage", f"{regdated_full:,}", f"{regdated_mv:,}"),
    ("Register dissolution dates, all firms", f"{regall_full:,}", f"{regall_mv:,}"),
], columns=["", "Full panel", "Estimation sample"])
a4b.to_csv(OUT / "appendix04b_closure_dating.csv", index=False)
print("A4 routes:", dict(routes))
print(f"A4 scheme split mv: {dict(sg)} | grant firms full "
      f"{int(firms.has_grant.sum())} mv {int(mv.has_grant.sum())}")

# ---------------------------------------------------------------- Appendix 5
mm = pd.read_csv(ROOT / "data/processed/grants_matched_mv.csv")
tl = mm[mm.treated == 1][["set_id", "land", "g_midx"]].rename(
    columns={"land": "set_land"})
tl["set_gyear"] = tl.g_midx // 12
mm = mm.merge(tl[["set_id", "set_land", "set_gyear"]], on="set_id")
mm = mm[~((mm.set_land.isin(["UK", "CH"])) & (mm.set_gyear >= 2021))]
pure = mm[(mm.group == "sme_eic") & (mm.bf_set == 0)]
bf = mm[(mm.group == "sme_eic") & (mm.bf_set == 1)]
cons = mm[mm.group == "consortium"]
assert pure.set_id.nunique() == 453 and len(pure) == 2204
assert bf.set_id.nunique() == 147 and cons.set_id.nunique() == 439
size_fill = pure[pure.treated == 1].size_class.notna().mean()
a5 = pd.DataFrame([
    ("Industry tags", pct(mv.n_industries.notna().mean())),
    ("Number of founders", pct(mv.n_founders.notna().mean())),
    ("Award size (treated firms, matched sample)",
     f"{num(100 * size_fill, 0)} of treated"),
], columns=["Variable", "Fill rate (%)"])
a5.to_csv(OUT / "appendix05_fill_rates.csv", index=False)

# ---------------------------------------------------------------- Appendix 6
def lpm_hr_row(label, lpm_id, hr_id, n_label, lpm_term="circ",
               hr_term="circ_loghr", crit=1.96):
    d = ""
    if lpm_id:
        l = row_of(timing, lpm_id, lpm_term)
        # crit: 1.96 (normal) except country-clustered rows, whose fixest
        # p-values are t(19)-based -> CI must use the same distribution
        lo, hi = l.estimate - crit * l.se, l.estimate + crit * l.se
        d = (f"{snum(100 * l.estimate)}{stars(l.p)} "
             f"[{snum(100 * lo)}, {snum(100 * hi)}]")
    h = ""
    if hr_id:
        r = row_of(timing, hr_id, hr_term)
        h = hrcell(r.estimate, r.se, r.p)
    return (label, d, h, n_label)


ame = row_of(timing, "V2_LOGIT_MV_AME", "circ_AME")
a6a = pd.DataFrame([
    lpm_hr_row("Main specification", "V2_S2_fund36_private_v2",
               "V2_HAZ_S2_first_private", "16,615"),
    lpm_hr_row("Country-clustered standard errors", "V2_S2_CLAND",
               "V2_HAZ_S2_CLAND", "16,615", crit=2.093),  # t(19), 20 clusters
    ("Logit, average marginal effect",
     num(100 * ame.estimate, 1), "", f"{int(ame.n):,}"),
    lpm_hr_row("High-confidence classifier subsample (≥ 0.8)",
               "V2_CONF_MV_S2", "V2_CONF_MV_HAZ_S2", "15,772"),
    lpm_hr_row("Full panel, January-imputed months",
               "V2FULL_S2_fund36_private_v2", "V2FULL_HAZ_S2_first_private",
               "22,039"),
    lpm_hr_row("Random month within founding year", "V2_RANDMONTH_S2",
               "V2_RANDMONTH_HAZ_S2", "22,039"),
    lpm_hr_row("Pre-registration rounds as month-0 events", "V2_PREFOUND0_S2",
               "V2_PREFOUND0_HAZ_S2", "16,615"),
    lpm_hr_row("Censored at register dissolution", None,
               "V2_REGDISS_CENSOR_HAZ_S2", "16,615"),
], columns=["Variant", "Difference (pp)", "Hazard ratio", "n"])
# frame-entry check (analysis/8_frame_robustness.R): the 36m gap
# among firms with >= 1 private round by the panel end is a pure timing
# quantity, free of the funded-frame entry margin
ep = frame[(frame.run_id == "R1_everpriv_S2") & (frame.term == "circ")].iloc[0]
a6a.loc[len(a6a)] = ("Firms with ≥ 1 private round by 2026 only",
                     cell(ep.estimate, ep.se, ep.p), "", f"{int(ep.n):,}")
yr = row_of(timing, "V2_YEARLY_CLOGLOG_S2", "circ_loghr")
a6a.loc[len(a6a)] = ("Yearly discrete-time model (cloglog)", "",
                     hrcell(yr.estimate, yr.se, yr.p), "22,039")
a6a.to_csv(OUT / "appendix06a_gap_robustness.csv", index=False)

SUBT_N = {"Waste": 337, "Nature": 574, "Design": 801, "Service": 558,
          "Platform": 774}
a6b_rows = []
for st in ["Waste", "Nature", "Design", "Service", "Platform"]:
    s1 = row_of(timing, "TIME_COX_subtypes", f"circ_subtype{st}_loghr")
    s2 = row_of(timing, "TIME_COX_subtypes_S2", f"circ_subtype{st}_loghr")
    a6b_rows.append((f"{st}-based ({SUBT_N[st]})",
                     hrcell(s1.estimate, s1.se, s1.p),
                     hrcell(s2.estimate, s2.se, s2.p)))
a6b = pd.DataFrame(a6b_rows, columns=[
    "Subtype (n)", "S1: country × cohort", "S2: + industry, team"])
a6b.to_csv(OUT / "appendix06b_subtype_hrs.csv", index=False)

# ---------------------------------------------------------------- Appendix 7
a7_rows = []
for cy in range(2016, 2023):
    d = mv[mv.t_year == cy]
    s2 = row_of(timing, f"V2_COHORT_S2_{cy}", "circ")
    a7_rows.append((
        str(cy), f"{len(d):,}", pct(d.circ.mean()),
        pct(d[d.circ == 1].fund36_private_v2.mean()),
        pct(d[d.circ == 0].fund36_private_v2.mean()),
        cell(s2.estimate, s2.se, s2.p),
        pct(d[d.circ == 1].grant36_cb_v2.mean()),
        pct(d[d.circ == 0].grant36_cb_v2.mean())))
a7 = pd.DataFrame(a7_rows, columns=[
    "Cohort", "Firms", "Circular (%)",
    "Private round ≤ 36 months, circular (%)", "non-circular (%)",
    "S2 difference (pp)", "Grant round ≤ 36 months, circular (%)",
    "non-circular (%)"])
a7.to_csv(OUT / "appendix07_cohorts.csv", index=False)

# ---------------------------------------------------------------- Appendix 8
def bal_row(d, col, label):
    t, c = d[d.treated == 1][col], d[d.treated == 0][col]
    pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    sd = 0.0 if pooled == 0 else (t.mean() - c.mean()) / pooled
    fmt = "{:.3f}" if abs(t.mean()) < 10 else "{:.1f}"
    return (label, fmt.format(t.mean()).replace("-", M),
            fmt.format(c.mean()).replace("-", M),
            ("+" if sd >= 0.0005 else "") + num(sd, 3))


p8 = pure.copy()
p8["pre_any_private"] = (p8.pre_n_private > 0).astype(int)
p8["founders_missing"] = p8.n_founders.isna().astype(int)
p8["n_founders_obs"] = p8.n_founders
a8a = pd.DataFrame([
    bal_row(p8, "pre_any_private", "Any private round before award"),
    bal_row(p8, "pre12_any_private", "Any private round in prior 12 months"),
    bal_row(p8, "pre_n_private", "Pre-award private-round count"),
    bal_row(p8, "grant_age_m", "Age at award (months)"),
    bal_row(p8.dropna(subset=["n_founders_obs"]), "n_founders_obs",
            "Number of founders"),
    bal_row(p8, "founders_missing", "Founders missing"),
    bal_row(p8, "n_industries", "Number of industry tags"),
], columns=["Variable", "Treated mean", "Control mean",
            "Standardized difference"])
a8a.to_csv(OUT / "appendix08a_balance.csv", index=False)

def set_sizes(d):
    sz = d[d.treated == 0].groupby("set_id").size()
    all_sets = d.set_id.unique()
    sz = sz.reindex(all_sets, fill_value=0)
    coarse = d.groupby("set_id").land_coarsened.max()
    return {
        "sets": len(all_sets), "full": int((sz == 4).sum()),
        "c3": int((sz == 3).sum()), "c2": int((sz == 2).sum()),
        "c1": int((sz == 1).sum()), "controls": int(sz.sum()),
        "coarse": int(coarse.sum()),
    }


ss = {g: set_sizes(d) for g, d in
      [("pure", pure), ("bf", bf), ("cons", cons)]}
a8b = pd.DataFrame([
    ("Sets", *[f"{ss[g]['sets']:,}" for g in ("pure", "bf", "cons")]),
    ("Sets with 4 controls (%)",
     *[num(100 * ss[g]["full"] / ss[g]["sets"], 1)
       for g in ("pure", "bf", "cons")]),
    ("Sets with 3 / 2 / 1 controls",
     *[f"{ss[g]['c3']} / {ss[g]['c2']} / {ss[g]['c1']}"
       for g in ("pure", "bf", "cons")]),
    ("Controls", *[f"{ss[g]['controls']:,}" for g in ("pure", "bf", "cons")]),
    ("Sets coarsened to all of Europe",
     *[f"{ss[g]['coarse']} ({num(100 * ss[g]['coarse'] / ss[g]['sets'], 1)}%)"
       for g in ("pure", "bf", "cons")]),
], columns=["", "Firm-targeted pure grants", "Blended finance",
            "Consortium-type"])
assert ss["pure"]["controls"] == 2204 - 453
a8b.to_csv(OUT / "appendix08b_set_sizes.csv", index=False)

# --------------------------------------------------------------- Appendix 9
def bridge_row(bs, h):
    m1_id = ("V2_run_bridge_" + bs if h == 24
             else f"V2_run_bridge_{bs}_{h}m")
    m1 = row_of(grants, m1_id, "treated")
    m2 = row_of(grants, f"V2_run_bridge_{bs}_{h}m_M2", "treated")
    hr = ""
    if h == 24:
        t = row_of(grants, f"V2_TRANS_bridge_{bs}", "treated_loghr")
        hr = hrcell(t.estimate, t.se, t.p)
    return (f"≤ {h} months", cell(m1.estimate, m1.se, m1.p),
            cell(m2.estimate, m2.se, m2.p), hr, f"{int(m1.n_sets):,}")


for bs, fname in [("noprior", "appendix09a_bridge_noprior.csv"),
                  ("prior", "appendix09b_bridge_prior.csv")]:
    t = pd.DataFrame([bridge_row(bs, h) for h in (12, 24, 36, 48)],
                     columns=["Horizon", "Matched (pp)", "+ covariates (pp)",
                              "Hazard ratio", "Sets"])
    t.to_csv(OUT / fname, index=False)

# --------------------------------------------------------------- Appendix 10
def q2(split):
    r = pdyn[(pdyn.split == split) & (pdyn.q == -2)].iloc[0]
    return ("+" if r.diff_withinset >= 0 else "") + num(
        100 * r.diff_withinset, 1)


def att_row(label, m1_id, trans_id, split):
    m1 = row_of(grants, m1_id, "treated")
    hr = ""
    if trans_id:
        t = row_of(grants, trans_id, "treated_loghr")
        hr = hrcell(t.estimate, t.se, t.p)
    return (f"{label} ({int(m1.n_sets):,})", cell(m1.estimate, m1.se, m1.p),
            hr, q2(split) if split else "")


a10a = pd.DataFrame([
    att_row("Firm-targeted pure grants", "V2_M1_grant_only",
            "V2_TRANS_grant_only", "sme_eic_grant_only"),
    att_row("Blended finance", "V2_M1_bf", "V2_TRANS_bf", "sme_eic_bf"),
    att_row("Pooled SME Instrument and EIC", "V2_M1_sme_eic", None, None),
    att_row("Consortium-type R&D", "V2_M1_consortium", "V2_TRANS_consortium",
            "consortium"),
], columns=["Definition (sets)", "Difference (pp)", "Hazard ratio",
            "Pre-trend at quarter −2 (pp)"])
a10a.to_csv(OUT / "appendix10a_treatment_definitions.csv", index=False)
a10b = pd.DataFrame([
    att_row("Horizon 2020", "V2_ERA_M1_sme", "V2_TRANS_ERA_sme", None),
    att_row("Horizon Europe", "V2_ERA_M1_eic", "V2_TRANS_ERA_eic", None),
], columns=["Era (sets)", "Difference (pp)", "Hazard ratio", "drop"]
).drop(columns="drop")
a10b.to_csv(OUT / "appendix10b_era.csv", index=False)


def se_row(label, run_id):
    r = row_of(grants, run_id, "treated")
    lo, hi = r.estimate - 1.96 * r.se, r.estimate + 1.96 * r.se
    return (label, f"{snum(100 * r.estimate)}{stars(r.p)}",
            f"[{snum(100 * lo)}, {snum(100 * hi)}]")


a10c = pd.DataFrame([
    se_row("Clustered by set (main)", "V2_M1_grant_only"),
    se_row("Two-way, set and firm", "V2_M1_grant_only_tw"),
    se_row("Heteroskedasticity-robust", "V2_M1_grant_only_hc"),
], columns=["Standard errors", "Difference (pp)", "95% CI"])
a10c.to_csv(OUT / "appendix10c_inference.csv", index=False)

# --------------------------------------------------------------- Appendix 11
def sv_cells(y):
    r1 = row_of(surv, f"SV1_MV_{y}", "circ")
    r2 = row_of(surv, f"SV2_MV_{y}", "circ")
    c1 = row_of(surv, f"RATE_MV_{y}_circ1", "circ=1").estimate
    c0 = row_of(surv, f"RATE_MV_{y}_circ0", "circ=0").estimate
    return (pct(c1), pct(c0), cell(r1.estimate, r1.se, r1.p),
            cell(r2.estimate, r2.se, r2.p))


a11a = pd.DataFrame([
    ("Closed", *sv_cells("status_closed")),
    ("Exited (acquired or IPO)", *sv_cells("exit_any")),
    ("Dead by any signal", *sv_cells("dead_any")),
], columns=["Status", "Circular (%)", "Non-circular (%)",
            "S1 difference (pp)", "S2 difference (pp)"])
a11a.to_csv(OUT / "appendix11a_status.csv", index=False)

a11b_rows = []
for st in ["Waste", "Nature", "Design", "Service", "Platform"]:
    cl = row_of(surv, f"RATE_MV_status_closed_{st}", st).estimate
    da = row_of(surv, f"RATE_MV_dead_any_{st}", st).estimate
    s2 = row_of(surv, "SV2_MV_SUBTYPE_dead_any", f"circ_subtype{st}")
    a11b_rows.append((f"{st}-based ({SUBT_N[st]})", pct(cl), pct(da),
                      cell(s2.estimate, s2.se, s2.p)))
nc_cl = row_of(surv, "RATE_MV_status_closed_None").estimate
nc_da = row_of(surv, "RATE_MV_dead_any_None").estimate
a11b_rows.append(("Non-circular (13,571)", pct(nc_cl), pct(nc_da),
                  "(reference)"))
a11b = pd.DataFrame(a11b_rows, columns=[
    "Subtype (n)", "Closed (%)", "Dead by any signal (%)",
    "S2 difference vs. non-circular, dead by any signal (pp)"])
a11b.to_csv(OUT / "appendix11b_subtype_status.csv", index=False)


def seq_cell(grp, h):
    r = seq[(seq["sample"] == "mv") & (seq.group == grp)
            & (seq.horizon_m == h)].iloc[0]
    return (f"{pct(r.share)} [{pct(r.ci_lo)}, {pct(r.ci_hi)}]")


gf_c = seq[(seq["sample"] == "mv") & (seq.group == "share_circ")].iloc[0]
gf_n = seq[(seq["sample"] == "mv") & (seq.group == "share_noncirc")].iloc[0]
gf_all = (gf_c.k + gf_n.k) / (gf_c.n + gf_n.n)
a11c = pd.DataFrame([
    ("First recorded event is a pure grant (%)", pct(gf_c.share),
     pct(gf_n.share), pct(gf_all)),
    ("Private financing within 24 months, grant-first firms (%)",
     seq_cell("circular", 24), seq_cell("non-circular", 24),
     seq_cell("all", 24)),
    ("   within 36 months", seq_cell("circular", 36),
     seq_cell("non-circular", 36), seq_cell("all", 36)),
    ("   within 48 months", seq_cell("circular", 48),
     seq_cell("non-circular", 48), seq_cell("all", 48)),
], columns=["", "Circular", "Non-circular", "All"])
a11c.to_csv(OUT / "appendix11c_sequencing.csv", index=False)

GROUPS = {
    "circ=1, treated=1": "Circular recipients (116)",
    "circ=1, treated=0": "Circular controls",
    "circ=0, treated=1": "Non-circular recipients (337)",
    "circ=0, treated=0": "Non-circular controls",
}
a11d_rows = []
for key in ["circ=1, treated=1", "circ=1, treated=0",
            "circ=0, treated=1", "circ=0, treated=0"]:
    g = fg[fg.group == key].set_index("horizon_m")
    a11d_rows.append((GROUPS[key],
                      *[pct(g.loc[h].incidence) for h in (12, 24, 36, 48)]))
a11d = pd.DataFrame(a11d_rows, columns=[
    "Group", "≤ 12 months (%)", "≤ 24 months (%)",
    "≤ 36 months (%)", "≤ 48 months (%)"])
a11d.to_csv(OUT / "appendix11d_fourgroup.csv", index=False)

n_t_circ = pure[(pure.treated == 1) & (pure.circ == 1)].set_id.nunique()
n_t_nc = pure[(pure.treated == 1) & (pure.circ == 0)].set_id.nunique()
print(f"A11d treated sets: circ {n_t_circ} | non-circ {n_t_nc}")

# --------------------------------------------------------------- Appendix 12
# Frame-entry selection check. Source: analysis/8_frame_robustness.R.


def fe_row(stratum, group):
    r = fetab[(fetab.stratum == stratum) & (fetab.group == group)].iloc[0]
    return r


a12a_rows = []
for grp, label in (("treated", "Recipients"), ("control", "Matched controls")):
    np_, pr = fe_row("noprior", grp), fe_row("prior", grp)
    a12a_rows.append((f"{label}: frame entry only after the (pseudo-)award (%)",
                      f"{pct(np_.share_late_entry)} (of {int(np_.n)})",
                      f"{pct(pr.share_late_entry)} (of {int(pr.n)})"))
late = fe_row("noprior", "control_late_entry")
early = fe_row("noprior", "control_early_entry")
via_g = fe_row("noprior", "control_early_via_cb_grant")
a12a_rows += [
    ("Controls, private round within 24 months (%):", "", ""),
    ("   frame entry after the pseudo-award",
     f"{pct(late.p_private_24m)} (n={int(late.n)})", "—"),
    ("   frame entry before the pseudo-award",
     f"{pct(early.p_private_24m)} (n={int(early.n)})", "—"),
    ("   of which entered via a CB grant month",
     f"{pct(via_g.share_late_entry)} ({int(via_g.n)} of {int(early.n)})", "—"),
]
a12a = pd.DataFrame(a12a_rows,
                    columns=["", "No prior private round", "Prior private round"])
a12a.to_csv(OUT / "appendix12a_frame_entry.csv", index=False)


def frow(run_id, arm, term="treated"):
    r = frame[(frame.run_id == run_id) & (frame.arm == arm)
              & (frame.term == term)]
    assert len(r) == 1, f"{run_id}/{arm}: {len(r)} rows"
    return r.iloc[0]


def fcell(run_id, arm, term="treated"):
    r = frow(run_id, arm, term)
    if r.scale == "HR":
        return hrcell(np.log(r.estimate), r.se, r.p)
    return cell(r.estimate, r.se, r.p)


def gcell(run_id, term="treated"):
    r = grants[grants.run_id == run_id]
    r = r[r.term == term] if term in set(r.term) else r
    r = r.iloc[0]
    if str(r.term).endswith("_loghr"):
        return hrcell(r.estimate, r.se, r.p)
    return cell(r.estimate, r.se, r.p)


A, B = "A_control_entry", "B_symmetric_entry"
a12b = pd.DataFrame([
    ("ATT, any private round ≤ 24 months (M1)",
     gcell("V2_M1_grant_only"), fcell("M1", A), fcell("M1", B)),
    ("ATT + founder controls (M2)",
     gcell("V2_M2_grant_only"), fcell("M2", A), fcell("M2", B)),
    ("Transition hazard ratio",
     gcell("V2_TRANS_grant_only", "treated_loghr"),
     fcell("TRANS_pooled", A, "treated_HR"), fcell("TRANS_pooled", B, "treated_HR")),
    ("No prior private round",
     gcell("V2_run_bridge_noprior"), fcell("bridge_noprior", A),
     fcell("bridge_noprior", B)),
    ("Prior private round",
     gcell("V2_run_bridge_prior"), fcell("bridge_prior", A),
     fcell("bridge_prior", B)),
    ("Difference (prior − no prior)",
     gcell("V2_run_bridge_diff", "treated:pre_any"),
     fcell("bridge_diff", A, "treated:pre_any"),
     fcell("bridge_diff", B, "treated:pre_any")),
    ("Award-size difference (small − large)",
     gcell("V2_run_dose_diff", "treated:small"),
     fcell("dose_diff", A, "treated:small"), fcell("dose_diff", B, "treated:small")),
    ("Matched sets", "453",
     str(int(frow("M1", A).n_sets)), str(int(frow("M1", B).n_sets))),
    ("   of which no prior private round", "223",
     str(int(frow("bridge_noprior", A).n_sets)),
     str(int(frow("bridge_noprior", B).n_sets))),
], columns=["", "Canonical", "Controls in frame before award",
            "Both sides in frame before award"])
a12b.to_csv(OUT / "appendix12b_frame_restricted.csv", index=False)

pure_t = pure[pure.treated == 1]
pure_c = pure[pure.treated == 0]
sh_t_all = (pure_t.post24_n_grant_months_ex > 0).mean()
sh_c_all = (pure_c.post24_n_grant_months_ex > 0).mean()
a12c = pd.DataFrame([
    ("Recipients with a further grant ≤ 24 months (%)", pct(sh_t_all),
     pct(frow("share_treated", "G_noneu_grants", "share").estimate)),
    ("Controls with a further grant ≤ 24 months (%)", pct(sh_c_all),
     pct(frow("share_control", "G_noneu_grants", "share").estimate)),
    ("Difference (pp, set FE)", gcell("V2_run_anygrant24_ex"),
     fcell("anygrant24_noneu", "G_noneu_grants")),
], columns=["", "All CB grant months", "Non-EU grant months only"])
a12c.to_csv(OUT / "appendix12c_noneu_grants.csv", index=False)

print("wrote", len(list(OUT.glob("appendix*.csv"))), "tables to", OUT)
