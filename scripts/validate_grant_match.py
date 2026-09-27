"""
Validate the CORDIS grant match (outputs/tables/cordis_firm_grants.csv) toward
individually verified pairs.

Unit of validation: the identity claim (firm_id, organisationID). Signals, each
independent of the name-based match routes:
  geo     CORDIS city/postcode vs Crunchbase HQ city/postal code
  domain  CORDIS organizationURL vs CB website(s) + contact-email domain
  name    route ∈ {name, shortname} = exact normalized legal name + country
Auto-confirm rule: (exact-name route AND (geo OR domain)) OR (geo AND domain).
Everything else lands in a review queue for pair-by-pair (LLM/manual) judgment.
Non-exact routes with a project starting >1y before CB founding are forced to review.

Also writes: scheme structure (mono-beneficiary vs consortium, SME/EIC groups),
recall candidates (unmatched CORDIS SME/EIC orgs fuzzy-close to a CB firm name),
and a grant↔CB-funding-event overlap table (panel_lean, ±3 months around project
start), evidence for whether CB records the grant itself as a funding event.

Read-only over inputs; writes outputs/tables/grant_match_*.csv + grant_scheme_structure.csv
+ grant_cb_overlap.csv. Deterministic (no sampling).

Run:  conda run -n ml python scripts/validate_grant_match.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT / "scripts")])
from match_cordis import CORDIS_DIRS, LANDS, dom, nname  # noqa: E402
from src.grants import eurnum  # noqa: E402

OUT = ROOT / "outputs/tables"
GRANTS = OUT / "cordis_firm_grants.csv"


def fold(s) -> str:
    """lowercase, strip accents, collapse to alnum+space"""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


# CB uses English exonyms and CORDIS local names; canonicalize both sides before comparing
CITY_ALIASES = {
    "vienna": "wien", "copenhagen": "kobenhavn", "munich": "munchen", "muenchen": "munchen",
    "cologne": "koln", "nuremberg": "nurnberg", "zurich": "zurich", "zuerich": "zurich",
    "geneva": "geneve", "lucerne": "luzern", "milan": "milano", "rome": "roma",
    "turin": "torino", "florence": "firenze", "naples": "napoli", "venice": "venezia",
    "genoa": "genova", "lisbon": "lisboa", "seville": "sevilla", "warsaw": "warszawa",
    "cracow": "krakow", "brussels": "bruxelles", "ghent": "gent", "gand": "gent",
    "antwerp": "antwerpen", "bruges": "brugge", "gothenburg": "goteborg",
    "the hague": "den haag", "s gravenhage": "den haag", "bozen": "bolzano",
    "brunswick": "braunschweig", "arhus": "aarhus", "hanover": "hannover",
}


def city_eq(a: str, b: str) -> bool:
    a, b = fold(a), fold(b)
    a, b = CITY_ALIASES.get(a, a), CITY_ALIASES.get(b, b)
    if not a or not b:
        return False
    if a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a):
        return True
    a0, b0 = a.split()[0], b.split()[0]
    a0, b0 = CITY_ALIASES.get(a0, a0), CITY_ALIASES.get(b0, b0)
    return a0 == b0 and len(a0) >= 4


def pc_norm(s) -> str:
    if pd.isna(s):
        return ""
    s = re.sub(r"^[A-Za-z]{1,2}-", "", str(s).strip())      # 'DK-1220' -> '1220'
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def pc_eq(a: str, b: str) -> bool:
    a, b = pc_norm(a), pc_norm(b)
    return bool(a) and bool(b) and (a == b or (len(a) >= 3 and (a.startswith(b) or b.startswith(a))))


def load_orgs() -> pd.DataFrame:
    cols = ["projectID", "organisationID", "name", "shortName", "SME", "activityType",
            "country", "city", "postCode", "organizationURL"]
    parts = []
    for d in CORDIS_DIRS.values():
        parts.append(pd.read_csv(d / "organization.csv", sep=";", dtype=str, usecols=cols))
    return pd.concat(parts, ignore_index=True)


def load_projects() -> pd.DataFrame:
    cols = ["id", "startDate", "ecSignatureDate", "fundingScheme"]
    parts = []
    for d in CORDIS_DIRS.values():
        parts.append(pd.read_csv(d / "project.csv", sep=";", dtype=str, usecols=cols))
    return pd.concat(parts, ignore_index=True).rename(columns={"id": "projectID"})


def load_cb_side() -> pd.DataFrame:
    from src.data_loading import load_companies_fat_selective
    cb = load_companies_fat_selective([
        "crunchbase_url", "cbcsv_headquarters_location", "cbcsv_postal_code",
        "cbcsv_contact_email", "cb_website", "cbcsv_website"])
    cb["cb_city"] = cb.cbcsv_headquarters_location.astype(str).str.split(",").str[0]
    cb["email_dom"] = cb.cbcsv_contact_email.astype(str).str.extract(r"@([\w.\-]+)")[0]
    return cb


def build_evidence() -> pd.DataFrame:
    g = pd.read_csv(GRANTS, dtype=str)
    orgs = load_orgs()
    cb = load_cb_side()

    # org-level identity fields (an organisationID can appear with several cities, keep the set)
    o = (orgs.groupby("organisationID")
             .agg(org_name=("name", "first"), org_short=("shortName", "first"),
                  org_cities=("city", lambda s: "|".join(sorted({str(x) for x in s.dropna()}))),
                  org_pcs=("postCode", lambda s: "|".join(sorted({str(x) for x in s.dropna()}))),
                  org_url=("organizationURL", "first"), sme=("SME", "first"))
             .reset_index())

    pair_proj = (g.groupby(["firm_id", "organisationID"])
                   .agg(route=("match", "first"), n_projects=("projectID", "nunique"),
                        min_start=("startDate", "min"), max_start=("startDate", "max"),
                        schemes=("fundingScheme", lambda s: "|".join(sorted({str(x) for x in s.dropna()}))),
                        cb_name=("cb_company_name", "first"), land=("land", "first"),
                        circ=("circ", "first"), t_year=("t_year", "first"))
                   .reset_index())
    e = pair_proj.merge(o, on="organisationID", how="left").merge(
        cb, left_on="firm_id", right_on="crunchbase_url", how="left")

    e["city_match"] = e.apply(lambda r: any(
        city_eq(r.cb_city, c) for c in str(r.org_cities).split("|") if c), axis=1)
    e["postcode_match"] = e.apply(lambda r: any(
        pc_eq(r.cbcsv_postal_code, p) for p in str(r.org_pcs).split("|") if p), axis=1)
    org_dom = e.org_url.map(dom)
    cb_doms = e.apply(lambda r: {d for d in (dom(r.cb_website), dom(r.cbcsv_website),
                                             dom("http://" + r.email_dom) if pd.notna(r.email_dom) else None)
                                 if d}, axis=1)
    e["domain_match"] = [od in ds if od else False for od, ds in zip(org_dom, cb_doms)]
    e["org_domain"] = org_dom
    e["cb_domains"] = ["|".join(sorted(d)) for d in cb_doms]

    e["geo"] = e.city_match | e.postcode_match
    e["name_route"] = e.route.isin(["name", "shortname"])
    e["founding_conflict"] = (pd.to_datetime(e.min_start, errors="coerce").dt.year
                              < pd.to_numeric(e.t_year) - 1)
    auto = (e.name_route & (e.geo | e.domain_match)) | (e.geo & e.domain_match)
    force_review = e.founding_conflict          # a project older than the firm always gets eyes
    e["status"] = "review"
    e.loc[auto & ~force_review, "status"] = "auto_confirmed"

    keep = ["firm_id", "cb_name", "land", "circ", "t_year", "organisationID", "org_name",
            "org_short", "route", "n_projects", "min_start", "max_start", "schemes",
            "cb_city", "org_cities", "city_match", "cbcsv_postal_code", "org_pcs",
            "postcode_match", "cb_domains", "org_domain", "domain_match", "sme",
            "founding_conflict", "status"]
    e = e[keep].sort_values(["status", "route", "land", "cb_name"])
    e.to_csv(OUT / "grant_match_evidence.csv", index=False)
    e[e.status == "review"].to_csv(OUT / "grant_match_review_queue.csv", index=False)

    print("== evidence (unit = firm × CORDIS organisation)")
    print(e.groupby(["route", "status"]).size().unstack(fill_value=0).to_string())
    print(f"pairs {len(e):,} | auto-confirmed {(e.status == 'auto_confirmed').sum():,} "
          f"| review {(e.status == 'review').sum():,} "
          f"| founding conflicts {int(e.founding_conflict.sum()):,}")
    return e


def scheme_structure() -> None:
    g = pd.read_csv(GRANTS, dtype=str)
    orgs = load_orgs()
    proj = load_projects()
    norg = orgs.groupby("projectID")["organisationID"].nunique().rename("n_orgs")
    g = (g.merge(norg, on="projectID", how="left")
          .merge(proj[["projectID", "ecSignatureDate"]], on="projectID", how="left"))
    sch = g.fundingScheme.fillna("")
    g["scheme_group"] = "other"
    g.loc[sch.str.contains("SME", case=False), "scheme_group"] = "SME-Instrument"
    g.loc[sch.str.contains("EIC", case=False), "scheme_group"] = "EIC"
    g["year"] = pd.to_datetime(g.startDate, errors="coerce").dt.year
    g["ec"] = g.ecContribution.map(eurnum)
    g["mono"] = g.n_orgs == 1

    t = (g.groupby(["scheme_group", "fp_x"])
           .agg(participations=("projectID", "count"), firms=("firm_id", "nunique"),
                circ_firms=("firm_id", lambda s: g.loc[s.index].drop_duplicates("firm_id").circ.astype(int).sum()),
                mono_share=("mono", "mean"), median_ec=("ec", "median"),
                yr_min=("year", "min"), yr_max=("year", "max"),
                dated=("startDate", lambda s: s.notna().mean()),
                signed=("ecSignatureDate", lambda s: s.notna().mean()))
           .round(2).reset_index())
    t.to_csv(OUT / "grant_scheme_structure.csv", index=False)
    print("\n== scheme structure")
    print(t.to_string(index=False))
    detail = (g[g.scheme_group != "other"].groupby(["scheme_group", "fundingScheme", "mono"])
              .size().rename("n").reset_index())
    print(detail.to_string(index=False))


def recall_candidates() -> None:
    """Unmatched CORDIS SME/EIC PRC orgs (our 20 countries) whose name is fuzzy-close
    to a CB firm in the same country, the 'legal name changed' miss class."""
    g = pd.read_csv(GRANTS, dtype=str)
    orgs = load_orgs()
    proj = load_projects()
    orgs = orgs.merge(proj[["projectID", "fundingScheme"]], on="projectID", how="left")
    sel = (orgs.activityType.eq("PRC") & orgs.country.isin(LANDS)
           & orgs.fundingScheme.fillna("").str.contains("SME|EIC", case=False))
    o = orgs[sel & ~orgs.organisationID.isin(g.organisationID)].drop_duplicates("organisationID").copy()
    o["nn"] = o.name.map(nname)

    f = pd.read_csv(ROOT / "data/processed/firms_analysis.csv", dtype=str,
                    usecols=["firm_id", "cb_company_name", "land"])
    f = f[~f.firm_id.isin(g.firm_id)].copy()
    f["nn"] = f.cb_company_name.map(nname)

    cands = []
    for land, fl in f.groupby("land"):
        ol = o[o.country == land]
        if not len(ol):
            continue
        by_pref: dict[str, list] = {}
        for _, r in fl.iterrows():
            if len(r.nn) >= 4:
                by_pref.setdefault(r.nn[:4], []).append(r)
        for _, orow in ol.iterrows():
            if len(orow.nn) < 4:
                continue
            for frow in by_pref.get(orow.nn[:4], []):
                sim = SequenceMatcher(None, orow.nn, frow.nn).ratio()
                if sim >= 0.78 and orow.nn != frow.nn:
                    cands.append({"org_name": orow["name"], "org_city": orow.city,
                                  "country": land, "cb_name": frow.cb_company_name,
                                  "firm_id": frow.firm_id, "similarity": round(sim, 3)})
    c = pd.DataFrame(cands).sort_values("similarity", ascending=False) if cands else pd.DataFrame()
    c.to_csv(OUT / "grant_recall_candidates.csv", index=False)
    print(f"\n== recall probe: {len(o):,} unmatched SME/EIC PRC orgs in our countries; "
          f"{len(c):,} fuzzy-close CB name candidates (≥0.78) → grant_recall_candidates.csv")


def cb_overlap() -> None:
    """±3-month window around first project start: does CB record a funding event?"""
    from src.data_loading import load_panel_lean
    g = pd.read_csv(GRANTS, dtype=str)
    g["start"] = pd.to_datetime(g.startDate, errors="coerce")
    g["ec"] = g.ecContribution.map(eurnum)
    first = (g.dropna(subset=["start"]).sort_values("start")
              .drop_duplicates("firm_id")[["firm_id", "start", "ec", "fundingScheme"]])
    first["m_idx"] = first.start.dt.year * 12 + first.start.dt.month - 1

    p = load_panel_lean(columns=["crunchbase_url", "month", "funding_event", "funding_total_usd"])
    p = p[p.crunchbase_url.isin(set(first.firm_id))].copy()
    p["month"] = pd.to_datetime(p.month)
    p["m_idx"] = p.month.dt.year * 12 + p.month.dt.month - 1
    ev = p[p.funding_event.astype(int) > 0].merge(first, left_on="crunchbase_url",
                                                  right_on="firm_id")
    ev["lag"] = ev.m_idx_x - ev.m_idx_y
    win = ev[ev.lag.between(-3, 3)].copy()
    win["amt_ratio"] = win.funding_total_usd.astype(float) / win.ec
    near = (win[win.amt_ratio.between(0.7, 1.6)].groupby("firm_id").size()
            .rename("n_amount_matched"))

    hit_firms = win.firm_id.nunique()
    out = first.merge(win.groupby("firm_id").size().rename("events_pm3"), on="firm_id",
                      how="left").merge(near, on="firm_id", how="left").fillna(
                      {"events_pm3": 0, "n_amount_matched": 0})
    out.to_csv(OUT / "grant_cb_overlap.csv", index=False)
    print(f"\n== grant↔CB overlap (first dated grant per matched firm, n={len(first):,})")
    print(f"firms with ≥1 CB funding event within ±3m of project start: {hit_firms:,} "
          f"({100 * hit_firms / len(first):.1f}%)")
    print(f"…of which amount ≈ ecContribution (ratio 0.7–1.6): {int((out.n_amount_matched > 0).sum()):,} "
          f"({100 * (out.n_amount_matched > 0).mean():.1f}% of matched firms)")
    print("lag distribution of in-window events (months, event − grant start):")
    print(win.lag.value_counts().sort_index().to_string())


def finalize() -> pd.DataFrame:
    """Evidence + auto rule + stored review verdicts -> grant_match_final.csv.

    Precedence: stored reject/uncertain beat everything (the review had more
    information than the signals); then manual recall recoveries and the
    auto-confirm rule; then stored confirms; leftovers = needs_review (new
    pairs the matcher found after the last review round; review and append
    to grant_match_llm_verdicts.csv, then rerun).
    """
    e = build_evidence()
    v = pd.read_csv(OUT / "grant_match_llm_verdicts.csv", dtype=str)
    e = e.merge(v[["firm_id", "organisationID", "verdict", "verdict_reason", "verdict_note"]],
                on=["firm_id", "organisationID"], how="left")

    e["final_status"] = "needs_review"
    e.loc[e.verdict == "confirm", "final_status"] = "verified_llm"
    e.loc[e.status == "auto_confirmed", "final_status"] = "verified_auto"
    manual = e.route == "manual"
    e.loc[manual, "final_status"] = "verified_llm"
    e.loc[manual & e.verdict_reason.isna(), "verdict_reason"] = \
        "recall recovery (city-verified, see MANUAL_MATCHES)"
    e.loc[e.verdict == "uncertain", "final_status"] = "uncertain"
    e.loc[e.verdict == "reject", "final_status"] = "rejected"
    e.to_csv(OUT / "grant_match_final.csv", index=False)

    print("\n== final status (after verdict merge)")
    print(e.groupby(["route", "final_status"]).size().unstack(fill_value=0).to_string())
    nr = e[e.final_status == "needs_review"]
    if len(nr):
        print(f"\n{len(nr)} pair(s) NEED REVIEW:")
        for _, r in nr.iterrows():
            print(f"  {r.route:9s} {r.land} | {r.cb_name} <-> {r.org_name} | "
                  f"{r.cb_city} vs {r.org_cities} | {r.cb_domains} vs {r.org_domain} | "
                  f"t{r.t_year} start{str(r.min_start)[:4]}")
    ok = e[e.final_status.str.startswith("verified")]
    g = pd.read_csv(GRANTS, dtype=str)
    keep = g.merge(ok[["firm_id", "organisationID"]], on=["firm_id", "organisationID"])
    firms = keep.drop_duplicates("firm_id")
    sme = keep.fundingScheme.fillna("").str.contains("SME|EIC", case=False)
    print(f"\nverified: {len(keep):,} participations | {len(firms):,} firms | "
          f"{firms.circ.astype(int).sum():,} circular | SME/EIC "
          f"{keep[sme].firm_id.nunique():,} firms / "
          f"{keep[sme].drop_duplicates('firm_id').circ.astype(int).sum():,} circular")
    return e


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "finalize":
        finalize()
    else:
        build_evidence()
        scheme_structure()
        recall_candidates()
        cb_overlap()
    print("\ndone.")


if __name__ == "__main__":
    main()
