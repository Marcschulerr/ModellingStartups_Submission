"""
Match CORDIS beneficiaries (H2020 + Horizon Europe) to the Crunchbase firm table.

Writes outputs/tables/cordis_firm_grants.csv, one row per firm x project with
the match route, the matched CORDIS org name (for auditing), funding scheme,
dates and EC contribution. Raw candidate matches only: the pair verdicts
(verified/uncertain/rejected) are merged by `scripts/validate_grant_match.py
finalize` into grant_match_final.csv, which is what downstream code consumes.

Routes (priority order; CORDIS orgs restricted to activityType == "PRC" OR
SME == true):
  name       exact normalized legal name + country          (validated clean)
  shortname  CORDIS shortName + country                     (validated clean)
  prefix     one normalized name is a prefix of the other,
             min 6 chars, same country                      (every pair reviewed)
  domain     registrable website domain, shared domains
             (>2 organisationIDs) and social/blog hosts excluded

Known recall gap: legal-name changes (e.g. SEaB Energy vs SEAB POWER LTD).
Run:  conda run -n ml python scripts/match_cordis.py
Expected size (soft pin): 3,761 participations, 1,885 firms, 613 circular.
"""
from __future__ import annotations

import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.csv as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.names import SUFF, PRIVATE_REGISTRIES, dom, nname  # noqa: E402,F401
from src.grants import eurnum  # noqa: E402
LANDS = {"UK", "DE", "FR", "ES", "NL", "IT", "SE", "PL", "DK", "CH",
         "FI", "EE", "BE", "NO", "PT", "AT", "LT", "LV", "LU", "IE"}
CORDIS_DIRS = {"h2020": ROOT / "data/external/cordis-h2020projects-csv",
               "HORIZON": ROOT / "data/external/cordis-HORIZONprojects-csv"}
OUT = ROOT / "outputs/tables/cordis_firm_grants.csv"

BAD_HOSTS = {"gmail.com", "wixsite.com", "google.com", "facebook.com", "linkedin.com"}
OCOLS = ["projectID", "organisationID", "name", "country", "role", "SME",
         "ecContribution", "fp"]


# nname/dom/SUFF/PRIVATE_REGISTRIES live in src/names.py and are re-exported
# above so `from match_cordis import nname, dom` keeps working.


# Recall recoveries (fuzzy probe + city verification): pairs the name routes
# cannot find because of a CB-side typo or a legal-name variant.
MANUAL_MATCHES = [
    # (firm_id suffix, organisationID, why)
    ("mater-dyamics", "913842239", "CB name typo 'Dyamics'; Porto = Porto"),
    ("arrecife-systems", "909962918", "legal name adds 'Energy'; Bilbao = Bilbao"),
    ("sens-solutions", "915873225", "legal 'SENSING SOLUTIONS SL'; Barcelona metro"),
]


def load_cordis() -> tuple[pd.DataFrame, pd.DataFrame]:
    ocols = ["projectID", "organisationID", "name", "shortName", "SME", "activityType",
             "country", "organizationURL", "role", "ecContribution"]
    pcols = ["id", "acronym", "startDate", "endDate", "ecSignatureDate", "fundingScheme",
             "ecMaxContribution", "frameworkProgramme", "legalBasis"]
    orgs, projs = [], []
    for tag, d in CORDIS_DIRS.items():
        o = pd.read_csv(d / "organization.csv", sep=";", dtype=str, usecols=ocols)
        o = o[o.country.isin(LANDS)]
        o["fp"] = tag
        orgs.append(o)
        p = pd.read_csv(d / "project.csv", sep=";", dtype=str, usecols=pcols)
        p["fp_p"] = tag
        projs.append(p)
    return pd.concat(orgs, ignore_index=True), pd.concat(projs, ignore_index=True)


def load_firms() -> pd.DataFrame:
    # Match against the FULL raw frame (25,421 firms), not the trimmed analysis
    # df: the match universe must not depend on the build's row policy.
    sys.path.insert(0, str(ROOT))
    from src.cohort import build_firm_base
    f = build_firm_base()[["firm_id", "cb_company_name", "land", "circ", "t_year"]]
    ro = pv.ReadOptions(block_size=1 << 26)          # multiline text fields in fat
    po = pv.ParseOptions(newlines_in_values=True)
    co = pv.ConvertOptions(include_columns=["crunchbase_url", "cb_website", "cbcsv_website"])
    w = pv.read_csv(str(ROOT / "data/raw/companies_fat.csv"),
                    read_options=ro, parse_options=po, convert_options=co).to_pandas()
    return f.merge(w, left_on="firm_id", right_on="crunchbase_url", how="left")


def main() -> None:
    O, P = load_cordis()
    F = load_firms()
    F["nn"] = F.cb_company_name.map(nname)
    F = F[F.nn != ""].copy()      # a name that is only legal-form tokens must not cross-join
    F["dom1"] = F.cb_website.map(dom)
    F["dom2"] = F.cbcsv_website.map(dom)

    # SME=true organisations also run under activityType OTH/REC in CORDIS
    # (e.g. ASTRAVEUS SAS, NAVENTIK GmbH, both exact name matches); a PRC-only
    # gate would lose 8 verified treated firms.
    Opc = O[(O.activityType == "PRC") | (O.SME.astype(str).str.lower() == "true")].copy()
    Opc["nn"] = Opc.name.map(nname)
    Opc["nns"] = Opc.shortName.map(nname)
    Opc = Opc[Opc.nn != ""].copy()
    Opc["odom"] = Opc.organizationURL.map(dom)
    dcount = Opc.dropna(subset=["odom"]).groupby("odom")["organisationID"].nunique()
    bad = set(dcount[dcount > 2].index) | BAD_HOSTS

    m1 = F.merge(Opc[OCOLS + ["nn"]], left_on=["nn", "land"], right_on=["nn", "country"])
    m1["match"] = "name"
    m2 = F.merge(Opc[Opc.nns.notna() & (Opc.nns != "")][OCOLS + ["nns"]],
                 left_on=["nn", "land"], right_on=["nns", "country"])
    m2["match"] = "shortname"

    # prefix route, memory-safe: per-country sorted unique org names + bisect
    oun = (Opc[Opc.nn.str.len() >= 6].groupby("country")["nn"]
           .apply(lambda s: sorted(set(s))).to_dict())
    pairs = []
    for _, r in F[F.nn.str.len() >= 6].iterrows():
        names = oun.get(r.land)
        if not names:
            continue
        hits = set(names[bisect_left(names, r.nn):bisect_right(names, r.nn + "￿")])
        for L in range(6, len(r.nn)):                # firm name startswith org name
            i = bisect_left(names, r.nn[:L])
            if i < len(names) and names[i] == r.nn[:L]:
                hits.add(r.nn[:L])
        # sorted: set iteration is hash-random across runs and the (firm, project)
        # dedupe below keeps the first row, so the order must be stable
        pairs += [(r.firm_id, r.land, h) for h in sorted(hits)]
    pp = pd.DataFrame(pairs, columns=["firm_id", "land", "nn_o"]).drop_duplicates()
    m3 = (pp.merge(Opc[OCOLS + ["nn"]].rename(columns={"nn": "nn_o"}),
                   left_on=["nn_o", "land"], right_on=["nn_o", "country"])
            .merge(F.drop(columns=["nn"]), on=["firm_id", "land"]))
    m3["match"] = "prefix"

    dl = pd.concat([F[["firm_id", "dom1"]].rename(columns={"dom1": "d"}),
                    F[["firm_id", "dom2"]].rename(columns={"dom2": "d"})]
                   ).dropna().drop_duplicates()
    dl = dl[~dl.d.isin(bad)]
    m4 = (dl.merge(Opc[Opc.odom.notna() & ~Opc.odom.isin(bad)][OCOLS + ["odom"]],
                   left_on="d", right_on="odom")
            .merge(F.drop(columns=["nn"]), on="firm_id"))
    m4["match"] = "domain"

    manual_ids = {f"https://www.crunchbase.com/organization/{fid}": oid
                  for fid, oid, _ in MANUAL_MATCHES}
    m5 = (pd.DataFrame({"firm_id": list(manual_ids), "organisationID": list(manual_ids.values())})
            .merge(Opc[OCOLS], on="organisationID")
            .merge(F.drop(columns=["nn"]), on="firm_id"))
    m5["match"] = "manual"

    cols = ["firm_id", "cb_company_name", "land", "circ", "t_year", "match",
            "projectID", "organisationID", "name", "role", "SME", "ecContribution", "fp"]
    M = pd.concat([m1[cols], m2[cols], m3[cols], m4[cols], m5[cols]], ignore_index=True)
    M["prio"] = M.match.map({"name": 0, "shortname": 1, "prefix": 2, "domain": 3, "manual": 4})
    # explicit two-stage tie-break, stable sorted so the pick is reproducible across
    # environments (a plain sort on prio would leave it to quicksort tie order):
    # 1. an org listed twice in one project keeps its funding-carrying row
    #    (0-contribution duplicates are linked-entity records)
    # 2. across different orgs of one project the lowest organisationID wins
    M["ec_num"] = M.ecContribution.map(eurnum).fillna(0.0)
    M = (M.sort_values(["prio", "organisationID", "ec_num"],
                       ascending=[True, True, False], kind="mergesort")
          .drop_duplicates(subset=["firm_id", "projectID", "organisationID"], keep="first")
          .drop_duplicates(subset=["firm_id", "projectID"], keep="first")
          .drop(columns="ec_num")
          .drop(columns="prio")
          .merge(P.rename(columns={"id": "projectID"}), on="projectID", how="left"))

    g = M.drop_duplicates("firm_id")
    sme = M.fundingScheme.fillna("").str.contains("SME|EIC", case=False)
    print(f"participations {len(M):,} | firms {len(g):,} | circ {int(g.circ.sum()):,} "
          f"| routes {M.match.value_counts().to_dict()}")
    print(f"SME/EIC: {int(sme.sum()):,} participations | "
          f"{M[sme].firm_id.nunique():,} firms | "
          f"{int(M[sme].drop_duplicates('firm_id').circ.sum()):,} circular")
    if not (3650 <= len(M) <= 3850):                 # soft pin around the expected 3,761
        print("WARNING: participation count drifted from the expected 3,761, "
              "investigate before using downstream.", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
