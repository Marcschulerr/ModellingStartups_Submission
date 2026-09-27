"""
Founding-month enrichment v2: the registry-verified founding month columns.

Baseline columns stay untouched (the frozen-artifact reconciliation and all
pins keep passing); v2 lives in parallel columns. Precedence for
founding_month_idx_v2:

  cb_day / cb_month   CB's own precise month, kept as-is (registry-consistent
                      for UK: 48.5 % same-day, 71.4 % exact month in the bulk QA)
  registry_uk / _fr   verified register incorporation month (auto-confirmed via
                      two independent signals, scripts/match_registry.py), adopted
                      ONLY where CB is year-only AND registry year == t_year, so
                      the v2 month provably stays inside the founding year
  web                 verified web month (grade A/B, modal across sources,
                      dealroom-first; adopt flag = web year == t_year)
  january             unchanged January fallback (founding_month_imputed_v2 = 1)

Definition precedent: Guzman & Kacperczyk (2019, Research Policy) define their
population via business REGISTRATION records, so "founding = registry
incorporation where verified" has a named template.

Registry status columns (registry_status, registry_dissolution_date) are
collect-only: a scoped end-state exception like the survival block, never
covariates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import outcomes

ROOT = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "data/scrape/registry"


def attach_founding_v2(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    src = np.select(
        [df["founded_precision"] == "day", df["founded_precision"] == "month"],
        ["cb_day", "cb_month"], default="january")
    v2 = df["founding_month_idx"].astype("int32").copy()
    year_only = df["founded_precision"].eq("year")

    reg_p = REG_DIR / "registry_match_final.csv"
    if reg_p.exists():
        reg = pd.read_csv(reg_p)
        reg["dt"] = pd.to_datetime(reg["creation_date"], errors="coerce")
        reg = reg.dropna(subset=["dt"]).set_index("firm_id")
        m = df["firm_id"].map(reg["dt"])
        my = df["firm_id"].map(reg["source"])
        # Q1 lag rule: register creation in Q1 of t_year+1 is the registration
        # lag signature (founded late in the year, registered weeks later,
        # median +21 days in the DE QA), so those months are adopted too; the
        # cohort variable t_year stays CB-based and untouched
        lag_q1 = (m.dt.year == df["t_year"] + 1) & (m.dt.month <= 3)
        adopt = year_only & m.notna() & ((m.dt.year == df["t_year"]) | lag_q1)
        v2[adopt] = (m.dt.year * 12 + m.dt.month - 1)[adopt].astype("int32")
        cc = my.str.extract(r"registry_([a-z]{2})")[0].fillna("uk")
        src = np.where(adopt, ("registry_" + cc).where(adopt, ""), src)
        # collect-only registry columns, ALL auto-confirmed matches, all lands
        df["registry_id"] = df["firm_id"].map(reg["registry_id"]).astype("string")
        df["registry_status"] = df["firm_id"].map(reg["registry_status"]).astype("string")
        df["registry_dissolution_date"] = pd.to_datetime(
            df["firm_id"].map(reg["dissolution_date"]), errors="coerce")
    else:
        df["registry_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["registry_status"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["registry_dissolution_date"] = pd.NaT

    webs = [pd.read_csv(p) for p in (REG_DIR / "web_month_final.csv",
                                     REG_DIR / "web_site_final.csv") if p.exists()]
    if webs:
        web = pd.concat(webs, ignore_index=True)
        web = (web[web["adopt"] == 1]
               .drop_duplicates("firm_id", keep="first").set_index("firm_id"))
        wy = df["firm_id"].map(web["year"])
        wm = df["firm_id"].map(web["month"])
        still_jan = year_only & (pd.Series(src) == "january")
        adopt_w = still_jan & wy.notna() & wm.notna() & (wy == df["t_year"])
        v2[adopt_w] = (wy * 12 + wm - 1)[adopt_w].astype("int32")
        src = np.where(adopt_w, "web", src)

    df["founding_month_idx_v2"] = v2.astype("int32")
    df["founding_v2_source"] = pd.Series(src, index=df.index, dtype="string")
    df["founding_month_imputed_v2"] = (df["founding_v2_source"] == "january"
                                       ).astype("int8")
    n0, n1 = int(df["founding_month_imputed"].sum()), int(
        df["founding_month_imputed_v2"].sum())
    counts = df["founding_v2_source"].value_counts().to_dict()
    print(f"      founding v2 sources: {counts}")
    print(f"      imputed share: {n0/len(df):.1%} (v1) -> {n1/len(df):.1%} (v2); "
          f"months gained: {n0 - n1:,}")
    assert (df.loc[df['founding_v2_source'].isin(['cb_day', 'cb_month', 'january']),
                   'founding_month_idx_v2']
            == df.loc[df['founding_v2_source'].isin(['cb_day', 'cb_month', 'january']),
                      'founding_month_idx']).all(), "v2 must equal v1 outside adoptions"
    v2y = df["founding_month_idx_v2"] // 12
    v2m = df["founding_month_idx_v2"] % 12  # 0 = January
    assert ((v2y == df["t_year"])
            | ((v2y == df["t_year"] + 1) & (v2m <= 2))).all(), \
        "v2 month left the founding year (+Q1 lag window)"
    return df


def attach_v2_outcomes(base: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Second window pass under the v2 founding month (parallel outcome columns)."""
    agg = outcomes.aggregate_panel_windows(
        base, panel, fidx_col="founding_month_idx_v2").add_suffix("_v2agg")
    df = base.merge(agg, left_on="firm_id", right_index=True, how="left",
                    validate="1:1")
    w = outcomes.MAIN_WINDOW
    df["fund36_private_v2"] = (df[f"n_private_months_{w}_v2agg"] >= 1).astype("Int8")
    df["grant36_cb_v2"] = (df[f"n_grant_months_{w}_v2agg"] >= 1).astype("Int8")
    df["sum_fund_usd_private_36_v2"] = df[f"sum_fund_usd_private_{w}_v2agg"]
    df["logfund36_private_v2"] = np.log1p(df["sum_fund_usd_private_36_v2"])
    df["success36_v2"] = (
        (df[f"n_fund_events_{w}_v2agg"] >= outcomes.SUCCESS_MIN_EVENTS)
        | (df[f"max_month_fund_usd_{w}_v2agg"] > outcomes.SUCCESS_MIN_MAX_MONTH_USD)
        | (df[f"n_ma_{w}_v2agg"] >= 1)).astype("Int8")
    df["first_private_offset_m_v2"] = df["first_private_offset_m_v2agg"].astype("Int32")
    df = df.drop(columns=[c for c in df.columns if c.endswith("_v2agg")])
    same = df["founding_month_idx_v2"] == df["founding_month_idx"]
    assert (df.loc[same, "fund36_private_v2"].astype("Int64")
            == df.loc[same, "fund36_private"].astype("Int64")).all(), \
        "v2 outcome differs from v1 where the month is unchanged"
    return df
