"""
Firm-level EU-grant block from the verified CORDIS match.

Inputs (both generated; pipeline order: scripts/match_cordis.py ->
scripts/validate_grant_match.py finalize -> this module inside the build):
  outputs/tables/cordis_firm_grants.csv   raw firm×project participations
  outputs/tables/grant_match_final.csv    pair-level verdicts (verified_* / uncertain / rejected)

Rules:
  - treated = verified pairs only (auto or LLM-reviewed); rejected pairs never enter;
    firms whose ONLY pairs are unresolved get has_grant_uncertain = 1 and stay untreated.
  - first grant = earliest by ecSignatureDate (fallback: project startDate);
    undated participations cannot be "first" but count in totals.
  - scheme groups: SME* -> SME-Instrument, *EIC* -> EIC, else consortium-R&D.
    Raw fundingScheme kept alongside ('-BF' = EIC blended finance incl. equity).
  - mono-beneficiary = the first grant's project has exactly one organisation
    (consortium size from the raw CORDIS organization.csv files).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRANTS_CSV = ROOT / "outputs/tables/cordis_firm_grants.csv"
FINAL_CSV = ROOT / "outputs/tables/grant_match_final.csv"
CORDIS_DIRS = (ROOT / "data/external/cordis-h2020projects-csv",
               ROOT / "data/external/cordis-HORIZONprojects-csv")

GRANT_COLS = ["has_grant", "has_grant_uncertain", "first_grant_date", "first_grant_scheme",
              "first_grant_scheme_group", "first_grant_mono", "ec_total_eur",
              "n_grant_participations", "grant_match_tier"]


def eurnum(s) -> float:
    """CORDIS numbers come as '1234.56' or '1234,56': normalize either."""
    if pd.isna(s) or str(s).strip() == "":
        return float("nan")
    t = str(s).strip()
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def scheme_group(scheme: pd.Series) -> pd.Series:
    """SME* to SME-Instrument, *EIC* to EIC, else consortium-R&D (SME wins on overlap)."""
    s = scheme.fillna("")
    out = pd.Series("consortium-R&D", index=scheme.index, dtype="string")
    out[s.str.contains("EIC", case=False)] = "EIC"
    out[s.str.contains("SME", case=False)] = "SME-Instrument"
    return out


def _project_n_orgs() -> pd.Series:
    parts = [pd.read_csv(d / "organization.csv", sep=";", dtype=str,
                         usecols=["projectID", "organisationID"]) for d in CORDIS_DIRS]
    o = pd.concat(parts, ignore_index=True)
    return o.groupby("projectID")["organisationID"].nunique().rename("n_orgs")


def build_grant_block() -> pd.DataFrame:
    g = pd.read_csv(GRANTS_CSV, dtype=str)
    pairs = pd.read_csv(FINAL_CSV, dtype=str,
                        usecols=["firm_id", "organisationID", "final_status"])
    assert not (pairs.final_status == "needs_review").any(), \
        "unreviewed match pairs, run validate_grant_match.py finalize and review first"
    g = g.merge(pairs, on=["firm_id", "organisationID"], how="left")
    assert g.final_status.notna().all(), "participation without a pair verdict"

    # verified pairs only: rejected and uncertain never enter the treated set
    v = g[g.final_status.str.startswith("verified")].copy()
    v["ec"] = v.ecContribution.map(eurnum)
    v["date"] = pd.to_datetime(v.ecSignatureDate.fillna(v.startDate), errors="coerce")
    v["scheme_group"] = scheme_group(v.fundingScheme)
    v = v.merge(_project_n_orgs(), on="projectID", how="left")
    v["mono"] = (pd.to_numeric(v.n_orgs, errors="coerce") == 1).astype("int8")

    # firm totals over verified participations
    firm = (v.groupby("firm_id")
             .agg(ec_total_eur=("ec", lambda s: float(np.nansum(s.values))),
                  n_grant_participations=("projectID", "nunique"))
             .reset_index())
    # first grant = earliest dated participation; undated ones count in totals
    # but can never be "first"
    dated = v.dropna(subset=["date"]).sort_values("date")
    first = dated.drop_duplicates("firm_id")[
        ["firm_id", "date", "fundingScheme", "scheme_group", "mono"]].rename(columns={
            "date": "first_grant_date", "fundingScheme": "first_grant_scheme",
            "scheme_group": "first_grant_scheme_group", "mono": "first_grant_mono"})
    firm = firm.merge(first, on="firm_id", how="left")
    # a firm's match tier is verified_llm as soon as ANY of its pairs needed review
    tier = (g[g.final_status.str.startswith("verified")]
            .groupby("firm_id").final_status
            .agg(lambda s: "verified_llm" if (s == "verified_llm").any() else "verified_auto")
            .rename("grant_match_tier"))
    firm = firm.merge(tier, on="firm_id", how="left")
    firm["has_grant"] = 1

    # firms whose ONLY pairs are unresolved: flagged, never treated
    by_firm = g.groupby("firm_id").final_status
    uncertain_only = by_firm.agg(lambda s: bool((s == "uncertain").any()
                                                and not s.str.startswith("verified").any()))
    uncertain_firms = pd.DataFrame({"firm_id": uncertain_only[uncertain_only].index,
                                    "has_grant_uncertain": 1})
    firm = firm.merge(uncertain_firms, on="firm_id", how="outer")
    return firm


def attach_grants(firms: pd.DataFrame) -> pd.DataFrame:
    block = build_grant_block()
    df = firms.merge(block, on="firm_id", how="left", validate="1:1")
    df["has_grant"] = df["has_grant"].fillna(0).astype("int8")
    df["has_grant_uncertain"] = df["has_grant_uncertain"].fillna(0).astype("int8")
    df["first_grant_mono"] = df["first_grant_mono"].fillna(0).astype("int8")
    df["n_grant_participations"] = df["n_grant_participations"].astype("Int16")
    df["ec_total_eur"] = df["ec_total_eur"].astype("float64")
    df["first_grant_scheme"] = df["first_grant_scheme"].astype("string")
    df["first_grant_scheme_group"] = df["first_grant_scheme_group"].astype("string")
    df["grant_match_tier"] = df["grant_match_tier"].astype("string")
    n = int(df["has_grant"].sum())
    print(f"      grant block: {n:,} treated firms | "
          f"{int((df['has_grant'] & (df['circ'] == 1)).sum()):,} circular | "
          f"{int(df['has_grant_uncertain'].sum()):,} uncertain-only (untreated)")
    return df
