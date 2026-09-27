"""
Registry matching for the founding-date enrichment. Offline: all network I/O
lives in scripts/fetch_registry.py, this script only matches register bulks
and cached responses. It follows the CORDIS verified-match discipline:
candidates, then a two-independent-signal auto-confirm, and everything
uncertain is kept out or reviewed pair by pair.

Arms (run order):
  uk-bulk   Companies House bulk snapshot vs UK firms; also writes the API
            target list (closed firms are absent from the bulk by construction)
  fr-bulk   Sirene parquets vs FR firms
  uk-api    replays cached Companies House lookups for the API targets
  web       scores cached Jina results into verified web months (grade A/B)
  finalize  merges all decision files and the verdict store into
            registry_match_final.csv; uncertain pairs stay in their own file
  uncertain-evidence  extracts the candidate rows behind uncertain pairs for
            the pair-by-pair review (append-only registry_match_verdicts.csv,
            confirms promoted by finalize as decision='verified_llm')

Auto-confirm needs two independent signals: exact current-name equality
(nname_reg) plus a geo signal (postal code or city), or, without geo, a
globally unique register name with creation year == t_year. Previous-name and
prefix routes are never auto-confirmed. Month adoption additionally requires
creation year == t_year, so the v2 month provably stays inside the founding
year.

Run:  conda run -n ml python scripts/match_registry.py --arm uk-bulk|fr-bulk|uk-api|web|finalize|uncertain-evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib import parse as uparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.names import nname_reg, city_eq  # noqa: E402
from src import webfetch  # noqa: E402

REG_DIR = ROOT / "data/scrape/registry"
CH_CSV = ROOT / "data/external/ch-bulk/BasicCompanyDataAsOneFile-2026-08-01.csv"
SIR_UL = ROOT / "data/external/sirene/StockUniteLegale.parquet"
SIR_ET = ROOT / "data/external/sirene/StockEtablissement.parquet"
FA_PARQ = ROOT / "data/processed/firms_analysis.parquet"


def norm_pc(s) -> str:
    return "" if pd.isna(s) else str(s).upper().replace(" ", "").strip()


def pc_out(s) -> str:
    s = "" if pd.isna(s) else str(s).upper().strip()
    return s.split(" ")[0] if " " in s else s[:4]


def load_targets(lands: list[str]) -> pd.DataFrame:
    fa = pd.read_parquet(FA_PARQ, columns=[
        "firm_id", "cb_company_name", "land", "t_year", "founded_date",
        "founded_precision", "status_closed"])
    fa = fa[fa["land"].isin(lands)].copy()
    ids = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cbcsv_headquarters_location"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    pcs = pd.read_csv(ROOT / "data/raw/companies_lean.csv",
                      usecols=["crunchbase_url", "cbcsv_postal_code"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    fa = fa.merge(ids, on="firm_id", how="left").merge(pcs, on="firm_id", how="left")
    fa["city"] = (fa["cbcsv_headquarters_location"].astype(str)
                  .str.split(",").str[0].str.strip().replace({"nan": ""}))
    fa["pc"] = fa["cbcsv_postal_code"].map(norm_pc)
    fa["pco"] = fa["cbcsv_postal_code"].map(pc_out)
    fa["nn"] = fa["cb_company_name"].map(nname_reg)
    fa["ty"] = fa["t_year"].astype(int)
    fa["precise"] = fa["founded_precision"].isin(["day", "month"])
    dup = fa["nn"].duplicated(keep=False) & fa["nn"].ne("")
    fa["cb_dup_name"] = dup
    return fa.reset_index(drop=True)


def decide(cands: pd.DataFrame, targets: pd.DataFrame, name_counts: Counter,
           source: str) -> pd.DataFrame:
    """Per-firm decision from candidate rows (columns: nn, route, reg_id, reg_name,
    pc, city, creation, status, dissolution)."""
    rows = []
    cmap = {nn: g for nn, g in cands.groupby("nn")} if len(cands) else {}
    for _, t in targets.iterrows():
        g = cmap.get(t["nn"])
        if g is None or t["nn"] == "":
            rows.append(dict(firm_id=t["firm_id"], source=source, decision="miss"))
            continue
        g = g.copy()
        g["yr"] = pd.to_datetime(g["creation"], errors="coerce").dt.year
        g["geo"] = [
            (t["pc"] != "" and norm_pc(p) == t["pc"])
            or (t["pco"] != "" and pc_out(p) == t["pco"])
            or (t["city"] != "" and city_eq(c, t["city"]))
            for p, c in zip(g["pc"], g["city"])]
        g["yr_ok"] = g["yr"].between(t["ty"] - 1, t["ty"] + 1)
        cur = g[g["route"] == "current"]
        surv = cur[cur["geo"] & cur["yr_ok"]]
        pick, dec = None, None

        def _year_promote(pool: pd.DataFrame):
            # exact name + register year == founding year is accepted even
            # without a geo signal; the month is adopted only when it is
            # unambiguous across all same-year candidates
            ye = pool[pool["yr"] == t["ty"]]
            if not len(ye):
                return None
            months = pd.to_datetime(ye["creation"], errors="coerce").dt.month
            return ye.iloc[0] if months.dropna().nunique() == 1 else None

        if t["cb_dup_name"]:
            # two CB firms share the normalized name: structurally unresolvable
            dec = "uncertain_cb_dup"
        elif len(surv) == 1:
            # exactly one candidate passes geo + year gates: the auto-confirm case
            pick, dec = surv.iloc[0], "auto"
        elif len(surv) > 1:
            # several gated candidates: only a unanimous same-year month survives
            p = _year_promote(surv)
            pick, dec = (p, "promoted_year") if p is not None else (None, "uncertain_multi")
        else:
            # no candidate passes the geo gate: accept only a globally unique
            # register name founded in t_year, or a unanimous same-year month
            uniq = cur[(cur["yr"] == t["ty"])
                       & (cur["nn"].map(lambda n: name_counts.get(n, 99)) == 1)]
            if len(uniq) == 1:
                pick, dec = uniq.iloc[0], "auto_unique"
            elif len(cur):
                p = _year_promote(cur)
                pick, dec = (p, "promoted_year") if p is not None else (None, "uncertain_gates")
            else:
                dec = "uncertain_route"  # previous-name/prefix candidates only
        row = dict(firm_id=t["firm_id"], source=source, decision=dec,
                   n_candidates=len(g))
        if pick is not None:
            row.update(route=pick["route"], registry_id=pick["reg_id"],
                       registry_name=pick["reg_name"], creation_date=pick["creation"],
                       registry_status=pick.get("status"),
                       dissolution_date=pick.get("dissolution"),
                       geo_signal=bool(pick["geo"]),
                       creation_year_eq=bool(pick["yr"] == t["ty"]))
        rows.append(row)
    return pd.DataFrame(rows)


def qa_precise(dec: pd.DataFrame, targets: pd.DataFrame, label: str) -> pd.DataFrame:
    m = dec[dec["decision"].isin(["auto", "auto_unique", "promoted_year"])].merge(
        targets[["firm_id", "precise", "founded_date", "ty"]], on="firm_id")
    p = m[m["precise"]].copy()
    if not len(p):
        print(f"  QA {label}: no precise firms matched")
        return pd.DataFrame()
    p["reg_dt"] = pd.to_datetime(p["creation_date"], errors="coerce")
    p["cb_dt"] = pd.to_datetime(p["founded_date"])
    p["err_days"] = (p["reg_dt"] - p["cb_dt"]).dt.days
    a = p["err_days"].abs()
    exact_m = ((p["reg_dt"].dt.year == p["cb_dt"].dt.year)
               & (p["reg_dt"].dt.month == p["cb_dt"].dt.month)).mean()
    print(f"  QA {label} (n={len(p)} precise auto-matches): same-day "
          f"{(a == 0).mean():.1%} | ≤7d {(a <= 7).mean():.1%} | ≤31d "
          f"{(a <= 31).mean():.1%} | exact month {exact_m:.1%} | "
          f"|Δyear|>1 {((p['reg_dt'].dt.year - p['cb_dt'].dt.year).abs() > 1).mean():.2%}")
    p["qa_label"] = label
    return p[["firm_id", "qa_label", "creation_date", "founded_date", "err_days"]]


# ── UK bulk ──────────────────────────────────────────────────────────────────────
def arm_uk_bulk() -> None:
    t = load_targets(["UK"])
    tset = set(t["nn"]) - {""}
    print(f"UK targets: {len(t)} firms ({t['precise'].sum()} precise, "
          f"{int(t['status_closed'].sum())} closed); pc filled {(t['pc'] != '').mean():.1%}")
    usecols = ["CompanyName", "CompanyNumber", "RegAddress.PostTown",
               "RegAddress.PostCode", "CompanyStatus", "DissolutionDate",
               "IncorporationDate"] + \
              [f"PreviousName_{i}.CompanyName" for i in range(1, 11)]
    name_counts: Counter = Counter()
    parts = []
    reader = pd.read_csv(CH_CSV, chunksize=250_000, dtype=str,
                         usecols=lambda c: c.strip() in usecols)
    for i, ch in enumerate(reader):
        ch.columns = [c.strip() for c in ch.columns]
        # CH bulk dates are DD/MM/YYYY, normalized to ISO at the source so no
        # downstream pd.to_datetime can month/day-swap them
        inc_iso = pd.to_datetime(ch["IncorporationDate"], format="%d/%m/%Y",
                                 errors="coerce").dt.strftime("%Y-%m-%d")
        dis_iso = pd.to_datetime(ch["DissolutionDate"], format="%d/%m/%Y",
                                 errors="coerce").dt.strftime("%Y-%m-%d")
        base = dict(reg_id=ch["CompanyNumber"], pc=ch["RegAddress.PostCode"],
                    city=ch["RegAddress.PostTown"], creation=inc_iso,
                    status=ch["CompanyStatus"], dissolution=dis_iso)
        for route, col in [("current", "CompanyName")] + \
                [("previous", f"PreviousName_{i}.CompanyName") for i in range(1, 11)]:
            if col not in ch:
                continue
            names = ch[col]
            mask = names.notna()
            if route == "previous" and not mask.any():
                continue
            nn = names[mask].map(nname_reg)
            hit = nn.isin(tset)
            if route == "current":
                name_counts.update(nn[hit].tolist())
            if hit.any():
                idx = nn[hit].index
                parts.append(pd.DataFrame({
                    "nn": nn[hit], "route": route, "reg_name": names[idx],
                    **{k: v[idx] for k, v in base.items()}}))
        if (i + 1) % 8 == 0:
            print(f"  chunk {i+1}, candidates so far "
                  f"{sum(len(p) for p in parts):,}", flush=True)
    cands = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["nn", "route"]))
    multi = int(sum(1 for v in name_counts.values() if v > 1))
    print(f"candidate rows {len(cands):,}; matched distinct names "
          f"{len(name_counts):,} (with registry duplicates: {multi:,})")
    dec = decide(cands, t, name_counts, "registry_uk_bulk")
    dec.to_csv(REG_DIR / "registry_decisions_uk_bulk.csv", index=False)
    qa = qa_precise(dec, t, "uk_bulk")
    qa.to_csv(REG_DIR / "registry_qa_uk_bulk.csv", index=False)
    auto = dec["decision"].isin(["auto", "auto_unique"])
    matched = set(dec.loc[auto, "firm_id"])
    api = t[(t["status_closed"] == 1) | (~t["precise"] & ~t["firm_id"].isin(matched))]
    api[["firm_id", "cb_company_name", "city", "pc", "pco", "ty",
         "status_closed", "precise", "nn", "cb_dup_name"]].to_csv(
        REG_DIR / "registry_uk_api_targets.csv", index=False)
    print(f"decisions: {dec['decision'].value_counts().to_dict()}")
    print(f"api targets written: {len(api)} "
          f"(closed {int(api['status_closed'].sum())}, misses {int((~api['precise'] & ~api['firm_id'].isin(matched)).sum())})")


# ── FR bulk ──────────────────────────────────────────────────────────────────────
def arm_fr_bulk() -> None:
    import pyarrow.parquet as pq
    t = load_targets(["FR"])
    tset = set(t["nn"]) - {""}
    print(f"FR targets: {len(t)} firms ({t['precise'].sum()} precise); "
          f"pc filled {(t['pc'] != '').mean():.1%}")
    name_counts: Counter = Counter()
    parts = []
    pf = pq.ParquetFile(SIR_UL)
    cols = ["siren", "dateCreationUniteLegale", "denominationUniteLegale",
            "denominationUsuelle1UniteLegale", "etatAdministratifUniteLegale",
            "dateDebut"]  # dateDebut of the current C-period = cessation date
    for bi, batch in enumerate(pf.iter_batches(batch_size=1_000_000, columns=cols)):
        d = batch.to_pandas()
        for route, col in [("current", "denominationUniteLegale"),
                           ("usuelle", "denominationUsuelle1UniteLegale")]:
            names = d[col]
            mask = names.notna()
            if not mask.any():
                continue
            nn = names[mask].map(nname_reg)
            hit = nn.isin(tset)
            if route == "current":
                name_counts.update(nn[hit].tolist())
            if hit.any():
                idx = nn[hit].index
                etat = d["etatAdministratifUniteLegale"][idx]
                parts.append(pd.DataFrame({
                    "nn": nn[hit], "route": route, "reg_name": names[idx],
                    "reg_id": d["siren"][idx],
                    "creation": d["dateCreationUniteLegale"][idx],
                    "status": etat,
                    "dissolution": d["dateDebut"][idx].where(etat == "C")}))
        if (bi + 1) % 6 == 0:
            print(f"  UL batch {bi+1}, candidates {sum(len(p) for p in parts):,}",
                  flush=True)
    cands = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"UL candidate rows {len(cands):,}")
    # siège postal/commune lookup for candidate sirens
    sirens = set(cands["reg_id"]) if len(cands) else set()
    pcmap, ctmap = {}, {}
    pf2 = pq.ParquetFile(SIR_ET)
    cols2 = ["siren", "etablissementSiege", "codePostalEtablissement",
             "libelleCommuneEtablissement"]
    for bi, batch in enumerate(pf2.iter_batches(batch_size=2_000_000, columns=cols2)):
        d = batch.to_pandas()
        d = d[d["etablissementSiege"].astype(str).str.lower().eq("true")
              & d["siren"].isin(sirens)]
        for s, p, c in zip(d["siren"], d["codePostalEtablissement"],
                           d["libelleCommuneEtablissement"]):
            pcmap.setdefault(s, p)
            ctmap.setdefault(s, c)
    cands["pc"] = cands["reg_id"].map(pcmap)
    cands["city"] = cands["reg_id"].map(ctmap)
    # FR usuelle route counts as current-quality only with geo; keep route label
    cands["route"] = cands["route"].replace({"usuelle": "current"})
    dec = decide(cands, t, name_counts, "registry_fr_bulk")
    dec.to_csv(REG_DIR / "registry_decisions_fr_bulk.csv", index=False)
    qa = qa_precise(dec, t, "fr_bulk")
    qa.to_csv(REG_DIR / "registry_qa_fr_bulk.csv", index=False)
    print(f"decisions: {dec['decision'].value_counts().to_dict()}")


# ── UK api replay ────────────────────────────────────────────────────────────────
def arm_uk_api() -> None:
    t = pd.read_csv(REG_DIR / "registry_uk_api_targets.csv", dtype={"ty": int})
    t["precise"] = t["precise"].astype(bool)
    t["cb_dup_name"] = t["cb_dup_name"].astype(bool)
    full = load_targets(["UK"])
    t = t.merge(full[["firm_id", "founded_date", "status_closed"]].drop_duplicates(
        "firm_id"), on="firm_id", how="left", suffixes=("", "_y"))
    rows = []
    for _, r in t.iterrows():
        rec = webfetch.ch_get("/search/companies?q="
                              + uparse.quote(str(r["cb_company_name"]))
                              + "&items_per_page=10")
        items = []
        if rec["status"] == 200:
            try:
                items = json.loads(rec["body"]).get("items", [])
            except json.JSONDecodeError:
                pass
        for it in items:
            nn = nname_reg(it.get("title", ""))
            route = "current"
            if nn != r["nn"]:
                # rename hunt: CH search matches previous names and returns
                # them in `snippet`; verified via the profile's
                # previous_company_names (cached by fetch stage uk-renames)
                sn = it.get("snippet") or ""
                if not any(nname_reg(p) == r["nn"] for p in sn.split("·")):
                    continue
                route = "previous"
            num = it.get("company_number")
            prec = webfetch.ch_get(f"/company/{num}")
            if prec["status"] != 200:
                continue
            prof = json.loads(prec["body"])
            if route == "previous":
                prevs = [p.get("name", "")
                         for p in prof.get("previous_company_names") or []]
                if not any(nname_reg(p) == r["nn"] for p in prevs):
                    continue
            addr = prof.get("registered_office_address") or {}
            rows.append(dict(
                nn=r["nn"], route=route, reg_id=num,
                reg_name=prof.get("company_name"), pc=addr.get("postal_code"),
                city=addr.get("locality"), creation=prof.get("date_of_creation"),
                status=prof.get("company_status"),
                dissolution=prof.get("date_of_cessation"), firm_hint=r["firm_id"]))
    cands = pd.DataFrame(rows)
    print(f"api candidate rows {len(cands)} "
          f"(previous-name route: {int((cands['route'] == 'previous').sum()) if len(cands) else 0})")
    counts = Counter(cands["nn"].tolist()) if len(cands) else Counter()
    tt = t.rename(columns={"status_closed": "status_closed"}).copy()
    tt["pc"] = tt["pc"].fillna("")
    tt["pco"] = tt["pco"].fillna("")
    tt["city"] = tt["city"].fillna("")
    dec = decide(cands, tt, counts, "registry_uk_api")
    dec.to_csv(REG_DIR / "registry_decisions_uk_api.csv", index=False)
    qa = qa_precise(dec, tt.assign(precise=tt["precise"]), "uk_api")
    qa.to_csv(REG_DIR / "registry_qa_uk_api.csv", index=False)
    print(f"decisions: {dec['decision'].value_counts().to_dict()}")


# ── web replay ───────────────────────────────────────────────────────────────────
def build_web_queries(name: str, city: str) -> list[tuple[str, str]]:
    return [("QF1", f'"{name}" {city} founded'),
            ("QF3", f"{name} site:dealroom.co")]


def arm_web() -> None:
    t = pd.read_csv(REG_DIR / "web_sweep_targets.csv")
    ids = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cb_website", "cbcsv_website",
                               "cbcsv_linked_in", "cbcsv_contact_email"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    t = t.merge(ids, on="firm_id", how="left")
    from src.names import dom
    t["domain"] = [dom(a) or dom(b) for a, b in zip(t["cb_website"], t["cbcsv_website"])]
    t["email_dom"] = [dom(str(e).split("@")[-1]) if pd.notna(e) and "@" in str(e)
                      else None for e in t["cbcsv_contact_email"]]
    t["li_slug"] = t["cbcsv_linked_in"].astype(str).str.extract(
        r"linkedin\.com/company/([^/?#]+)")[0].str.lower()
    out = []
    n_hit = 0
    for _, r in t.iterrows():
        cands, hits_all = [], []
        for qid, q in build_web_queries(str(r["cb_company_name"]), str(r["city"] or "")):
            f = webfetch.CACHE_DEFAULT / (hashlib.sha1(
                ("https://s.jina.ai/?q=" + uparse.quote(q)).encode()
            ).hexdigest() + ".json")
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            for h in webfetch.parse_search_results(rec):
                hits_all.append(h)
                txt = h["title"] + " — " + h["snippet"]
                for d in webfetch.extract_dates(txt, "founding"):
                    if d["precision"] in ("day", "month"):
                        cands.append((h["url"], webfetch.grade(r, h["url"], txt), d))
        u = webfetch.pick_dealroom_url(hits_all)
        if u:
            f = webfetch.CACHE_DEFAULT / (hashlib.sha1(
                ("https://r.jina.ai/" + u).encode()).hexdigest() + ".json")
            if f.exists():
                rec = json.loads(f.read_text())
                if rec["status"] == 200:
                    for d in webfetch.extract_dates(rec["body"][:20000], "founding"):
                        if d["precision"] in ("day", "month"):
                            cands.append((u, webfetch.grade(r, u, rec["body"][:2000]), d))
        cands = [(u_, g, d) for u_, g, d in cands if g in ("A", "B")
                 and int(r["ty"]) - 1 <= d["year"] <= int(r["ty"]) + 1]
        if not cands:
            continue
        ym = pd.Series([(d["year"], d["month"]) for _, _, d in cands])
        modal = ym.mode().iloc[0]
        u_, g, d = next(c for c in cands if (c[2]["year"], c[2]["month"]) == modal)
        n_hit += 1
        out.append(dict(firm_id=r["firm_id"], year=d["year"], month=d["month"],
                        grade=g, url=u_, n_candidates=len(cands),
                        disagree=int(ym.nunique() > 1),
                        adopt=int(d["year"] == int(r["ty"])),
                        context=d["context"]))
    w = pd.DataFrame(out)
    w.to_csv(REG_DIR / "web_month_final.csv", index=False)
    print(f"web months: {n_hit}/{len(t)} firms "
          f"({n_hit/max(len(t),1):.1%}); adoptable "
          f"{int(w['adopt'].sum()) if len(w) else 0}")


# ── uncertain evidence for the pair-by-pair review ──────────────────────────────
def _uncertain_review_subset() -> pd.DataFrame:
    """Uncertain pairs worth a verdict (classes gates/route/multi) from every
    dated source: UK/FR plus OC (de/dk/ie/lu/pl) and the open registers
    (be/fi/no/ee/lv). Stays out: cb_dup (structurally unresolvable), ch_oc (no
    incorporation dates, dissolution pool measured empty) and firms already
    auto-matched elsewhere."""
    unc = pd.read_csv(REG_DIR / "registry_match_uncertain.csv")
    unc = unc[~unc["source"].isin(["registry_ch_oc"])
              & unc["decision"].isin(["uncertain_gates", "uncertain_route",
                                      "uncertain_multi"])]
    fin = pd.read_csv(REG_DIR / "registry_match_final.csv", usecols=["firm_id"])
    return unc[~unc["firm_id"].isin(set(fin["firm_id"]))].copy()


def arm_uncertain_evidence() -> None:
    unc = _uncertain_review_subset()
    firms = unc.drop_duplicates("firm_id")[["firm_id", "decision"]]
    lands = sorted({s.split("_")[1].upper() for s in unc["source"]})
    t = load_targets(lands).merge(firms, on="firm_id")
    fa = pd.read_parquet(FA_PARQ, columns=["firm_id", "founding_v2_source",
                                           "industries", "status_closed"])
    desc = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                       usecols=["crunchbase_url", "cb_description"], dtype=str
                       ).rename(columns={"crunchbase_url": "firm_id"})
    t = t.merge(fa.drop(columns=["status_closed"]), on="firm_id").merge(
        desc, on="firm_id", how="left")
    print(f"review firms: {len(t)} (UK {(t['land'] == 'UK').sum()}, "
          f"FR {(t['land'] == 'FR').sum()}) from {len(unc)} uncertain pairs")

    # UK bulk candidates (with SIC + previous names) for all subset UK firms
    tset_uk = set(t.loc[t["land"] == "UK", "nn"]) - {""}
    parts = []
    usecols = ["CompanyName", "CompanyNumber", "RegAddress.PostTown",
               "RegAddress.PostCode", "CompanyStatus", "DissolutionDate",
               "IncorporationDate", "SICCode.SicText_1", "SICCode.SicText_2"] + \
              [f"PreviousName_{i}.CompanyName" for i in range(1, 11)]
    reader = pd.read_csv(CH_CSV, chunksize=250_000, dtype=str,
                         usecols=lambda c: c.strip() in usecols)
    for i, ch in enumerate(reader):
        ch.columns = [c.strip() for c in ch.columns]
        inc_iso = pd.to_datetime(ch["IncorporationDate"], format="%d/%m/%Y",
                                 errors="coerce").dt.strftime("%Y-%m-%d")
        dis_iso = pd.to_datetime(ch["DissolutionDate"], format="%d/%m/%Y",
                                 errors="coerce").dt.strftime("%Y-%m-%d")
        sic = ch["SICCode.SicText_1"].fillna("") + \
            ("; " + ch["SICCode.SicText_2"].fillna("")).replace("; ", "")
        base = dict(reg_id=ch["CompanyNumber"], pc=ch["RegAddress.PostCode"],
                    city=ch["RegAddress.PostTown"], creation=inc_iso,
                    status=ch["CompanyStatus"], dissolution=dis_iso, sic=sic)
        for route, col in [("current", "CompanyName")] + \
                [("previous", f"PreviousName_{i}.CompanyName") for i in range(1, 11)]:
            if col not in ch:
                continue
            names = ch[col]
            mask = names.notna()
            if not mask.any():
                continue
            nn = names[mask].map(nname_reg)
            hit = nn.isin(tset_uk)
            if hit.any():
                idx = nn[hit].index
                parts.append(pd.DataFrame({
                    "nn": nn[hit], "route": route, "matched_name": names[idx],
                    "reg_name": ch["CompanyName"][idx],
                    "cand_source": "registry_uk_bulk",
                    **{k: v[idx] for k, v in base.items()}}))
        if (i + 1) % 8 == 0:
            print(f"  UK chunk {i+1}, candidates {sum(len(p) for p in parts):,}",
                  flush=True)
    cand_uk = (pd.concat(parts, ignore_index=True) if parts
               else pd.DataFrame(columns=["nn"]))

    # UK API candidates (cached replay; covers dissolved companies absent from bulk)
    api_firms = t[t["firm_id"].isin(
        set(unc.loc[unc["source"] == "registry_uk_api", "firm_id"]))]
    rows = []
    for _, r in api_firms.iterrows():
        rec = webfetch.ch_get("/search/companies?q="
                              + uparse.quote(str(r["cb_company_name"]))
                              + "&items_per_page=10")
        items = []
        if rec["status"] == 200:
            try:
                items = json.loads(rec["body"]).get("items", [])
            except json.JSONDecodeError:
                pass
        for it in items:
            nn = nname_reg(it.get("title", ""))
            route, mname = "current", it.get("title")
            if nn != r["nn"]:
                sn = it.get("snippet") or ""
                hits = [p.strip() for p in sn.split("·")
                        if nname_reg(p) == r["nn"]]
                if not hits:
                    continue
                route, mname = "previous", hits[0]
            prec = webfetch.ch_get(f"/company/{it.get('company_number')}")
            if prec["status"] != 200:
                continue
            prof = json.loads(prec["body"])
            if route == "previous":
                prevs = [p.get("name", "")
                         for p in prof.get("previous_company_names") or []]
                pv = [p for p in prevs if nname_reg(p) == r["nn"]]
                if not pv:
                    continue
                mname = pv[0]
            addr = prof.get("registered_office_address") or {}
            rows.append(dict(
                nn=r["nn"], route=route, matched_name=mname,
                reg_name=prof.get("company_name"), cand_source="registry_uk_api",
                reg_id=prof.get("company_number"), pc=addr.get("postal_code"),
                city=addr.get("locality"), creation=prof.get("date_of_creation"),
                status=prof.get("company_status"),
                dissolution=prof.get("date_of_cessation"),
                sic="; ".join(prof.get("sic_codes") or [])))
    cand_api = pd.DataFrame(rows)
    print(f"UK candidates: bulk {len(cand_uk):,} + api {len(cand_api):,}")

    # FR candidates (with NAF activity code)
    import pyarrow.parquet as pq
    tset_fr = set(t.loc[t["land"] == "FR", "nn"]) - {""}
    parts = []
    pf = pq.ParquetFile(SIR_UL)
    cols = ["siren", "dateCreationUniteLegale", "denominationUniteLegale",
            "denominationUsuelle1UniteLegale", "etatAdministratifUniteLegale",
            "dateDebut", "activitePrincipaleUniteLegale"]
    for bi, batch in enumerate(pf.iter_batches(batch_size=1_000_000, columns=cols)):
        d = batch.to_pandas()
        for route, col in [("current", "denominationUniteLegale"),
                           ("usuelle", "denominationUsuelle1UniteLegale")]:
            names = d[col]
            mask = names.notna()
            if not mask.any():
                continue
            nn = names[mask].map(nname_reg)
            hit = nn.isin(tset_fr)
            if hit.any():
                idx = nn[hit].index
                etat = d["etatAdministratifUniteLegale"][idx]
                parts.append(pd.DataFrame({
                    "nn": nn[hit], "route": route, "matched_name": names[idx],
                    "reg_name": d["denominationUniteLegale"][idx],
                    "cand_source": "registry_fr_bulk",
                    "reg_id": d["siren"][idx],
                    "creation": d["dateCreationUniteLegale"][idx],
                    "status": etat,
                    "dissolution": d["dateDebut"][idx].where(etat == "C"),
                    "sic": d["activitePrincipaleUniteLegale"][idx]}))
    cand_fr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    sirens = set(cand_fr["reg_id"]) if len(cand_fr) else set()
    pcmap, ctmap = {}, {}
    pf2 = pq.ParquetFile(SIR_ET)
    cols2 = ["siren", "etablissementSiege", "codePostalEtablissement",
             "libelleCommuneEtablissement"]
    for batch in pf2.iter_batches(batch_size=2_000_000, columns=cols2):
        d = batch.to_pandas()
        d = d[d["etablissementSiege"].astype(str).str.lower().eq("true")
              & d["siren"].isin(sirens)]
        for s, p, c in zip(d["siren"], d["codePostalEtablissement"],
                           d["libelleCommuneEtablissement"]):
            pcmap.setdefault(s, p)
            ctmap.setdefault(s, c)
    if len(cand_fr):
        cand_fr["pc"] = cand_fr["reg_id"].map(pcmap)
        cand_fr["city"] = cand_fr["reg_id"].map(ctmap)
    print(f"FR candidates: {len(cand_fr):,}")

    # further sources: OC lands + dated open registers, candidates replayed
    # via the open_registries builders (lazy import; that module imports from
    # here, so a top-level import would be circular)
    import open_registries as oreg
    extras = []
    for src in sorted(set(unc["source"])):
        if src in ("registry_uk_bulk", "registry_uk_api", "registry_fr_bulk"):
            continue
        cc = src.split("_")[1]
        tl = t[t["land"] == cc.upper()]
        if not len(tl):
            continue
        if src.endswith("_oc"):
            c, _ = oreg.cands_oc(cc, tl)
        elif src == "registry_be_open":
            c, _ = oreg.cands_be(set(tl["nn"]) - {""})
        elif src == "registry_fi_open":
            c, _ = oreg.cands_api("fi", tl)
        elif src in ("registry_no_open", "registry_ee_open", "registry_lv_open"):
            c, _ = {"no": oreg.cands_no, "ee": oreg.cands_ee,
                    "lv": oreg.cands_lv}[cc](set(tl["nn"]) - {""})
        else:
            continue
        print(f"  {src}: {len(c)} candidate rows")
        if len(c):
            c = c.copy()
            c["cand_source"], c["land"] = src, cc.upper()
            c["matched_name"] = c["reg_name"]
            c["sic"] = None
            extras.append(c)

    for c, ld in ((cand_uk, "UK"), (cand_api, "UK"), (cand_fr, "FR")):
        if len(c):
            c["land"] = ld
    cands = pd.concat([cand_uk, cand_api, cand_fr, *extras], ignore_index=True)
    nf = (cands[cands["route"] == "current"].groupby(["land", "nn"])["reg_id"]
          .nunique().rename("name_freq_register").reset_index())

    # firm x candidate evidence rows (nn scoped to the land's register)
    ev = t.merge(cands, on=["nn", "land"], suffixes=("_cb", "_reg"))
    ev = ev.merge(nf, on=["land", "nn"], how="left")
    ev["name_freq_register"] = ev["name_freq_register"].fillna(0).astype(int)
    # a candidate can appear via bulk AND api: keep the info-richer row
    ev["_has_diss"] = ev["dissolution"].notna()
    ev = (ev.sort_values(["firm_id", "reg_id", "_has_diss"],
                         ascending=[True, True, False])
            .drop_duplicates(["firm_id", "reg_id"], keep="first")
            .drop(columns=["_has_diss"]))
    ev["creation_year"] = pd.to_datetime(ev["creation"], errors="coerce").dt.year
    ev["year_delta"] = ev["creation_year"] - ev["ty"]
    ev["geo_pc"] = [(a != "" and norm_pc(b) == a) for a, b in
                    zip(ev["pc_cb"], ev["pc_reg"])]
    ev["geo_pco"] = [(a != "" and pc_out(b) == a) for a, b in
                     zip(ev["pco"], ev["pc_reg"])]
    ev["geo_city"] = [city_eq(a, b) if (pd.notna(a) and pd.notna(b)) else False
                      for a, b in zip(ev["city_reg"], ev["city_cb"])]
    ev["n_candidates_firm"] = ev.groupby("firm_id")["reg_id"].transform("nunique")
    dead = ev["dissolution"].notna() | ev["status"].astype(str).str.lower().isin(
        ["dissolved", "c", "liquidation", "in administration"])
    # adoptable = year-exact OR Q1 of t+1 (registration lag rule)
    cmonth = pd.to_datetime(ev["creation"], errors="coerce").dt.month
    adoptable = ev["year_delta"].eq(0) | (ev["year_delta"].eq(1) & cmonth.le(3))
    dead_near = dead & ev["year_delta"].abs().le(1)
    ev["month_lever"] = (ev["founding_v2_source"].eq("january")
                         & adoptable.groupby(ev["firm_id"]).transform("any"))
    ev["fail_lever"] = (ev["status_closed"].astype(float).eq(1.0)
                        & dead_near.groupby(ev["firm_id"]).transform("any"))
    # cap runaway generic names: 10 closest-year candidates per firm (true count
    # stays visible in n_candidates_firm)
    ev = ev[ev["month_lever"] | ev["fail_lever"]].copy()
    ev["_ayd"] = ev["year_delta"].abs().fillna(99)
    ev = (ev.sort_values(["firm_id", "_ayd"]).groupby("firm_id").head(10)
            .drop(columns=["_ayd"]))
    out = ev[["firm_id", "cb_company_name", "land", "ty", "founded_precision",
              "status_closed", "founding_v2_source", "decision", "industries",
              "cb_description", "city_cb", "pc_cb", "cand_source", "route",
              "matched_name", "reg_name", "reg_id", "city_reg", "pc_reg",
              "creation", "creation_year", "year_delta", "status", "dissolution",
              "sic", "geo_pc", "geo_pco", "geo_city", "name_freq_register",
              "n_candidates_firm", "month_lever", "fail_lever"]].rename(
        columns={"reg_id": "registry_id", "ty": "t_year"})
    out["registry_id"] = out["registry_id"].astype(str)
    # verdict-covered pairs must never drop out of this file (finalize joins the
    # promoted confirms against it, and promoted firms leave the lever filter on
    # re-runs): carry their old rows forward
    vp, old_p = REG_DIR / "registry_match_verdicts.csv", \
        REG_DIR / "registry_uncertain_evidence.csv"
    if vp.exists() and old_p.exists():
        vv = pd.read_csv(vp, dtype={"registry_id": str})
        old = pd.read_csv(old_p, dtype={"registry_id": str})
        vkeys = set(zip(vv["firm_id"], vv["registry_id"]))
        nkeys = set(zip(out["firm_id"], out["registry_id"]))
        keep_old = old[[p in vkeys and p not in nkeys for p in
                        zip(old["firm_id"], old["registry_id"])]]
        if len(keep_old):
            out = pd.concat([out, keep_old], ignore_index=True)
            print(f"carried {len(keep_old)} verdict-covered pairs from the "
                  f"previous evidence file")
    out.to_csv(REG_DIR / "registry_uncertain_evidence.csv", index=False)
    print(f"evidence rows: {len(out):,} pairs for "
          f"{out['firm_id'].nunique():,} firms "
          f"(month lever {out.loc[out['month_lever'], 'firm_id'].nunique():,}, "
          f"fail lever {out.loc[out['fail_lever'], 'firm_id'].nunique():,}); "
          f"skipped no-yield firms: "
          f"{len(t) - out['firm_id'].nunique():,}")


# ── finalize ─────────────────────────────────────────────────────────────────────
def arm_finalize() -> None:
    fa = pd.read_parquet(FA_PARQ, columns=["firm_id", "cb_company_name", "land",
                                           "t_year", "founded_precision",
                                           "status_closed"])
    fa["ty"] = fa["t_year"].astype(int)
    ids = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cbcsv_headquarters_location"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    fa = fa.merge(ids, on="firm_id", how="left")
    fa["city"] = (fa["cbcsv_headquarters_location"].astype(str)
                  .str.split(",").str[0].str.strip().replace({"nan": ""}))
    decs = []
    for p in ([REG_DIR / f for f in ("registry_decisions_uk_bulk.csv",
                                  "registry_decisions_uk_api.csv",
                                  "registry_decisions_fr_bulk.csv")]
              + sorted(REG_DIR.glob("registry_decisions_open_*.csv"))):
        if p.exists():
            decs.append(pd.read_csv(p))
    dec = pd.concat(decs, ignore_index=True) if decs else pd.DataFrame()
    auto = dec[dec["decision"].isin(["auto", "auto_unique", "promoted_year",
                                     "auto_domain"])].copy()
    # land guard: a firm may only carry a verdict from ITS land's register.
    # The DE4.csv relabel (DE to CH) left 26 hits from the German register on
    # Swiss firms, all German subsidiary GmbHs of Swiss parents (AERA HEALTH
    # GERMANY GMBH, VISENSE SERVICE GMBH, ...), 14 of them adopted as founding
    # month. Arms are rerun per land, but a stale decisions file must never
    # survive this check.
    reg_cc = auto["source"].str.extract(r"^registry_([a-z]{2})", expand=False).str.upper()
    firm_cc = auto["firm_id"].map(fa.set_index("firm_id")["land"])
    cross = reg_cc.notna() & firm_cc.notna() & (reg_cc != firm_cc)
    if cross.any():
        print(f"land guard: {int(cross.sum())} auto rows from another land's register dropped")
        auto = auto[~cross].copy()
    # one registry verdict per firm: bulk wins over api (same register, bulk
    # fresher for live firms); national registers win over Orbis (file order;
    # PL is the lone alphabetical exception, measured overlap there: 0).
    # Exception: the UK bulk snapshot contains only LIVE companies, so for a
    # CB-closed firm a bulk auto that CONFLICTS with the API arm (different
    # company number) is a living namesake by construction; the API row (which
    # sees dissolved entities) wins for exactly those firms.
    ukmask = auto["source"].isin(["registry_uk_bulk", "registry_uk_api"])
    nun = auto[ukmask].groupby("firm_id")["registry_id"].nunique()
    closed = set(fa.loc[fa["status_closed"] == 1, "firm_id"])
    conflict = set(nun[nun > 1].index) & closed
    if conflict:
        auto["_prio"] = (~(auto["firm_id"].isin(conflict)
                           & auto["source"].eq("registry_uk_api"))).astype(int)
        auto = (auto.sort_values("_prio", kind="stable")
                .drop_duplicates("firm_id").drop(columns="_prio"))
        print(f"uk bulk/api id conflicts on closed firms: {len(conflict)} "
              f"-> api row kept")
    else:
        auto = auto.drop_duplicates("firm_id", keep="first")
    # pair-verdict store (grant-match pattern): append-only pair verdicts over
    # registry_uncertain_evidence.csv; confirms join the final as their own
    # tier decision='verified_llm' (audit trail = evidence + store)
    vp = REG_DIR / "registry_match_verdicts.csv"
    if vp.exists():
        v = pd.read_csv(vp, dtype={"registry_id": str})
        v = v.drop_duplicates(["firm_id", "registry_id"], keep="last")
        ev = pd.read_csv(REG_DIR / "registry_uncertain_evidence.csv",
                         dtype={"registry_id": str})
        conf = v[v["verdict"] == "confirm"].merge(
            ev, on=["firm_id", "registry_id"], how="left", validate="1:1")
        assert conf["reg_name"].notna().all(), "confirm verdict without evidence row"
        dupfirm = conf["firm_id"].duplicated(keep=False)
        assert not dupfirm.any(), ("multiple confirmed candidates for "
                                   f"{sorted(set(conf.loc[dupfirm, 'firm_id']))[:3]}")
        # reviewed verdicts outrank EVERY auto tier. An earlier version dropped
        # confirms wherever an auto row existed, which silently displaced 87 of
        # 222 reviewed confirms (14 with a different adopted month), so the
        # precedence is enforced explicitly:
        n_displaced = int(auto["firm_id"].isin(set(conf["firm_id"])).sum())
        auto = auto[~auto["firm_id"].isin(set(conf["firm_id"]))]
        if n_displaced:
            print(f"verdict precedence: {n_displaced} auto rows yielded to "
                  f"reviewed confirms")
        promo = pd.DataFrame(dict(
            firm_id=conf["firm_id"], source=conf["cand_source"],
            decision="verified_llm", n_candidates=conf["n_candidates_firm"],
            route=conf["route"], registry_id=conf["registry_id"],
            registry_name=conf["reg_name"], creation_date=conf["creation"],
            registry_status=conf["status"], dissolution_date=conf["dissolution"],
            geo_signal=conf["geo_pc"] | conf["geo_pco"] | conf["geo_city"],
            creation_year_eq=conf["year_delta"].eq(0)))
        auto = pd.concat([auto, promo], ignore_index=True)
        print(f"verdict store: {len(v)} pair verdicts -> {len(promo)} promoted "
              f"verified_llm ({int(promo['creation_year_eq'].sum())} year-exact, "
              f"{int(promo['dissolution_date'].notna().sum())} with dissolution)")
    auto["creation_year"] = pd.to_datetime(auto["creation_date"],
                                           errors="coerce").dt.year
    auto.to_csv(REG_DIR / "registry_match_final.csv", index=False)
    unc = dec[~dec["decision"].isin(["auto", "auto_unique", "promoted_year",
                                     "auto_domain", "miss"])]
    unc.to_csv(REG_DIR / "registry_match_uncertain.csv", index=False)
    qas = [pd.read_csv(REG_DIR / f) for f in
           ("registry_qa_uk_bulk.csv", "registry_qa_uk_api.csv",
            "registry_qa_fr_bulk.csv") if (REG_DIR / f).exists()]
    (pd.concat(qas, ignore_index=True) if qas else pd.DataFrame()).to_csv(
        REG_DIR / "registry_qa_cb_precise.csv", index=False)
    m = fa.merge(auto[["firm_id", "creation_year"]], on="firm_id", how="left")
    adopted = ((fa["founded_precision"] == "year")
               & (m["creation_year"] == m["ty"]))
    sweep = fa[(fa["founded_precision"] == "year") & ~adopted]
    sweep[["firm_id", "cb_company_name", "land", "city"]].assign(
        ty=sweep["ty"]).to_csv(REG_DIR / "web_sweep_targets.csv", index=False)
    print(f"registry auto-matches: {len(auto):,} (uncertain kept out: {len(unc):,})")
    print(f"year-only firms with adoptable registry month: {int(adopted.sum()):,} "
          f"of 12,217; web sweep targets: {len(sweep):,}")
    if (REG_DIR / "web_month_final.csv").exists():
        w = pd.read_csv(REG_DIR / "web_month_final.csv")
        print(f"web months present: {len(w):,} (adoptable {int(w['adopt'].sum()):,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["uk-bulk", "fr-bulk", "uk-api", "web", "finalize",
                             "uncertain-evidence"])
    a = ap.parse_args()
    {"uk-bulk": arm_uk_bulk, "fr-bulk": arm_fr_bulk, "uk-api": arm_uk_api,
     "web": arm_web, "finalize": arm_finalize,
     "uncertain-evidence": arm_uncertain_evidence}[a.arm]()


if __name__ == "__main__":
    main()
