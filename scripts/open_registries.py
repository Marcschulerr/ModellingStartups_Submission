"""
Open-registry arms, DATA ONLY: collect the data packages first, no analysis
is coupled to this script.

Adds verified founding months from free official registers for the lands the
UK/FR round could not cover, same match discipline as scripts/match_registry.py
(exact nname_reg + geo/year gates, two independent signals, uncertain kept out):

  Bulk arms (download once into data/external/openreg/):
    NO  Brønnøysund Enhetsregisteret   full gzip JSON, no key
    EE  e-Business Register            general-data JSON zip, no key
    LV  UR open data                   register CSV, no key
    DE  Offeneregister (OKFN dump)     jsonl.bz2; coverage ends ~2019, so only
                                       cohorts 2016–18 can match; date coverage
                                       verified in a smoke count before matching
    BE  KBO/CBE Open Data              manual download after free signup;
                                       extracted KboOpenData_*_Full folder;
                                       ACTIVE entities only, so no dissolutions
                                       and no months for already-closed BE firms
  OC arm (approved research plan; fetch via
  scripts/fetch_registry.py --stage oc): OpenCorporates search replays for
  DE/AT/PT/LU (`--src oc` = all four, or oc-de|oc-at|oc-pt|oc-lu). Search
  responses carry incorporation/dissolution dates, status, address and
  previous_names; previous-name hits become route='previous' (never
  auto-confirmed but verdict-round material, cf. the UK rename lesson).
  DE decisions OVERWRITE the dead Offeneregister arm (0 dates in that dump).
  LU postcode caveat: OC 'L-1855' vs CB variants; digits stored, city_eq
  carries the geo fallback.
  API arms (per-firm lookups, cached via src/webfetch.cached_get):
    FI  PRH avoindata v3               registrationDate in search response
    DK  cvrapi.dk                      startdate (day precision); polite UA
    CH  Zefix REST                     search + detail; founding date only if a
                                       registry date field is exposed (smoke)

Decisions land in data/scrape/registry/registry_decisions_open_<src>.csv with
source = "registry_<cc>_open"; scripts/match_registry.py --arm finalize merges
them; adoption rule unchanged (registry year == t_year). QA per land against
CB-precise firms printed (wrong-entity gate ≈ 0 like UK/FR).

Run:  conda run -n ml python scripts/open_registries.py --stage fetch [--src all|no|ee|lv|de|fi|dk|ch]
      conda run -n ml python scripts/open_registries.py --stage match [--src ...]
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib import parse as uparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.names import city_eq, dom, nname_reg  # noqa: E402
from src import webfetch  # noqa: E402
from match_registry import (REG_DIR, decide, load_targets, norm_pc,  # noqa: E402
                            pc_out, qa_precise)

EXT = ROOT / "data/external/openreg"
BULK = {
    "no": ("https://data.brreg.no/enhetsregisteret/api/enheter/lastned",
           "no_enheter.json.gz"),
    "ee": ("https://avaandmed.ariregister.rik.ee/sites/default/files/avaandmed/"
           "ettevotja_rekvisiidid__yldandmed.json.zip", "ee_yldandmed.json.zip"),
    "lv": ("https://dati.ur.gov.lv/register/register.csv", "lv_register.csv"),
    "de": ("https://daten.offeneregister.de/de_companies_ocdata.jsonl.bz2",
           "de_ocdata.jsonl.bz2"),
}


def fetch_bulk(src: str) -> None:
    import urllib.request
    url, fname = BULK[src]
    EXT.mkdir(parents=True, exist_ok=True)
    dest = EXT / fname
    if dest.exists() and dest.stat().st_size > 1e6:
        print(f"{src}: {fname} present ({dest.stat().st_size/1e6:.0f} MB)")
        return
    print(f"{src}: downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-research"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"{src}: done ({dest.stat().st_size/1e6:.0f} MB)")


def fetch_api(src: str, limit: int | None) -> None:
    lands = {"fi": "FI", "dk": "DK", "ch": "CH"}
    t = load_targets([lands[src]])
    if limit:
        t = t.head(limit)
    n_new = 0
    for i, (_, r) in enumerate(t.iterrows()):
        q = uparse.quote(str(r["cb_company_name"]))
        if src == "fi":
            rec = webfetch.cached_get(
                f"https://avoindata.prh.fi/opendata-ytj-api/v3/companies?name={q}&page=1",
                {"User-Agent": "thesis-research"})
        elif src == "dk":
            rec = webfetch.cached_get(
                f"https://cvrapi.dk/api?search={q}&country=dk",
                {"User-Agent": "Masterthesis TUM research marc.schuler5@gmail.com"})
        else:  # ch: POST not cacheable via cached_get -> GET search endpoint
            rec = webfetch.cached_get(
                f"https://www.zefix.admin.ch/ZefixREST/api/v1/firm/search.json?name={q}",
                {"User-Agent": "thesis-research", "Accept": "application/json"})
        n_new += 0 if rec.get("cached") else 1
        if (i + 1) % 100 == 0:
            print(f"  {src} {i+1}/{len(t)} new={n_new}", flush=True)
    print(f"{src}: {len(t)} firms fetched, {n_new} new requests")


# ── per-source candidate builders → rows(nn, route, reg_id, reg_name, pc, city,
#    creation, status, dissolution) matched against the land's target-name set ──
def cands_no(tset: set) -> tuple[pd.DataFrame, Counter]:
    rows, counts = [], Counter()
    with gzip.open(EXT / BULK["no"][1], "rt", encoding="utf-8") as f:
        data = json.load(f)
    for e in data:
        nn = nname_reg(e.get("navn", ""))
        if not nn:
            continue
        if nn in tset:
            counts[nn] += 1
            addr = e.get("forretningsadresse") or {}
            rows.append(dict(
                nn=nn, route="current", reg_id=e.get("organisasjonsnummer"),
                reg_name=e.get("navn"), pc=addr.get("postnummer"),
                city=addr.get("poststed"),
                creation=e.get("stiftelsesdato") or e.get("registreringsdatoEnhetsregisteret"),
                status="slettet" if e.get("slettedato") else "aktiv",
                dissolution=e.get("slettedato")))
    return pd.DataFrame(rows), counts


def cands_ee(tset: set) -> tuple[pd.DataFrame, Counter]:
    rows, counts = [], Counter()
    with zipfile.ZipFile(EXT / BULK["ee"][1]) as z:
        with z.open(z.namelist()[0]) as f:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    for e in data:
        nn = nname_reg(e.get("nimi", ""))
        if nn and nn in tset:
            counts[nn] += 1
            yld = e.get("yldandmed") or {}
            aad = ((yld.get("aadressid") or [{}]) or [{}])[0]
            # EE dates are DD.MM.YYYY, normalized to ISO at the source (the
            # unparsed form produced the 2%-exact-month QA disaster)
            c = pd.to_datetime(yld.get("esmaregistreerimise_kpv"), dayfirst=True,
                               errors="coerce")
            rows.append(dict(
                nn=nn, route="current", reg_id=e.get("ariregistri_kood"),
                reg_name=e.get("nimi"), pc=aad.get("indeks_postinumber"),
                city=aad.get("asukoht_ehak_tekstina"),
                creation=c.strftime("%Y-%m-%d") if pd.notna(c) else None,
                status=e.get("staatus_tekstina"), dissolution=None))
    return pd.DataFrame(rows), counts


def cands_lv(tset: set) -> tuple[pd.DataFrame, Counter]:
    d = pd.read_csv(EXT / BULK["lv"][1], sep=";", dtype=str)
    d.columns = [c.lower() for c in d.columns]
    rows, counts = [], Counter()
    for _, e in d.iterrows():
        nn = nname_reg(e.get("name", ""))
        if nn and nn in tset:
            counts[nn] += 1
            rows.append(dict(nn=nn, route="current", reg_id=e.get("regcode"),
                             reg_name=e.get("name"), pc=None,
                             city=str(e.get("address", "")).split(",")[0],
                             creation=e.get("registered"),
                             status=e.get("closed") and "closed" or "active",
                             dissolution=e.get("terminated")))
    return pd.DataFrame(rows), counts


def cands_de(tset: set) -> tuple[pd.DataFrame, Counter]:
    rows, counts = [], Counter()
    n_lines = n_dated = 0
    with bz2.open(EXT / BULK["de"][1], "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            nn = nname_reg(e.get("name", ""))
            if not nn or nn not in tset:
                continue
            counts[nn] += 1
            reg = e.get("all_attributes") or {}
            creation = (e.get("incorporation_date")
                        or reg.get("native_company_number_date") or None)
            if creation:
                n_dated += 1
            rows.append(dict(
                nn=nn, route="current", reg_id=e.get("company_number"),
                reg_name=e.get("name"), pc=None,
                city=str(reg.get("registered_office") or ""),
                creation=creation, status=e.get("current_status"),
                dissolution=e.get("dissolution_date")))
            if n_lines % 1_000_000 == 0:
                print(f"  de: {n_lines/1e6:.0f}M lines, cands {len(rows)}", flush=True)
    print(f"  de: {n_lines:,} lines scanned; candidates {len(rows)}, "
          f"with creation date {n_dated}")
    return pd.DataFrame(rows), counts


def cands_be(tset: set) -> tuple[pd.DataFrame, Counter]:
    kbo = sorted(EXT.glob("KboOpenData_*_Full"))
    assert kbo, "KBO full dump missing: data/external/openreg/KboOpenData_*_Full/"
    kbo = kbo[-1]
    ent = pd.read_csv(kbo / "enterprise.csv", dtype=str,
                      usecols=["EnterpriseNumber", "Status", "StartDate"])
    # KBO dates are DD-MM-YYYY, ISO at the source (dayfirst trap, cf. UK/EE fix)
    ent["creation"] = pd.to_datetime(ent["StartDate"], format="%d-%m-%Y",
                                     errors="coerce").dt.strftime("%Y-%m-%d")
    ent = ent.set_index("EnterpriseNumber")
    den = pd.read_csv(kbo / "denomination.csv", dtype=str)
    # 001 legal name, 003 commercial name (brand, the FR-usuelle analogue); language
    # duplicates of the same name collapse per entity before anything is counted
    den = den[den["TypeOfDenomination"].isin(["001", "003"])].copy()
    den["nn"] = den["Denomination"].map(nname_reg)
    den = den[den["nn"].ne("") & den["EntityNumber"].isin(ent.index)]
    den = den.drop_duplicates(["EntityNumber", "nn", "TypeOfDenomination"])
    legal = den[den["TypeOfDenomination"] == "001"].drop_duplicates(
        ["EntityNumber", "nn"])
    counts = Counter(legal.loc[legal["nn"].isin(tset), "nn"].tolist())
    hit = (den[den["nn"].isin(tset)]
           .sort_values("TypeOfDenomination")            # 001 wins over 003
           .drop_duplicates(["EntityNumber", "nn"]))
    adr = pd.read_csv(kbo / "address.csv", dtype=str,
                      usecols=["EntityNumber", "TypeOfAddress", "Zipcode",
                               "MunicipalityNL", "MunicipalityFR"])
    adr = (adr[adr["TypeOfAddress"] == "REGO"]
           .drop_duplicates("EntityNumber").set_index("EntityNumber"))
    rows = []
    for num, nn, typ, name in zip(hit["EntityNumber"], hit["nn"],
                                  hit["TypeOfDenomination"], hit["Denomination"]):
        e = ent.loc[num]
        a = adr.loc[num] if num in adr.index else None
        city = None if a is None else (
            a["MunicipalityNL"] if pd.notna(a["MunicipalityNL"])
            else a["MunicipalityFR"])
        rows.append(dict(
            nn=nn, route="current" if typ == "001" else "usuelle",
            reg_id=num, reg_name=name,
            pc=None if a is None else a["Zipcode"], city=city,
            creation=e["creation"], status=e["Status"], dissolution=None))
    df = pd.DataFrame(rows)
    if len(df):
        # commercial-name route counts as current-quality only with geo/year
        # gates downstream, same relabel convention as the FR usuelle route
        df["route"] = df["route"].replace({"usuelle": "current"})
    return df, counts


OC_JC = {"de": "DE", "at": "AT", "pt": "PT", "lu": "LU",
         "dk": "DK", "pl": "PL", "ie": "IE", "ch": "CH"}
OC_PC = {"de": r"\b(\d{5})\b", "at": r"\b(\d{4})\b", "lu": r"\b(\d{4})\b",
         "pt": r"\b(\d{4}-\d{3})\b", "dk": r"\b(\d{4})\b",
         "pl": r"\b(\d{2}-\d{3})\b", "ch": r"\b(\d{4})\b",
         "ie": r"(?!x)x"}  # Eircodes unreliable in OC addresses -> city_eq only

# PL register names carry multi-word legal forms + quoted brands, stripped at
# the SOURCE before nname_reg (arm-local, the global normalizer stays untouched)
_PL_FORMS = __import__("re").compile(
    r"sp[oó]łka z ograniczon[aą] odpowiedzialno[sś]ci[aą]"
    r"|sp[oó]łka komandytowo-akcyjna|sp[oó]łka komandytowa|sp[oó]łka akcyjna"
    r"|sp[oó]łka jawna|sp\.?\s?z\s?o\.?\s?o\.?", __import__("re").IGNORECASE)


def cands_oc(jc: str, targets: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    import hashlib
    import re
    rows, counts = [], Counter()
    pcre = re.compile(OC_PC[jc])
    for _, r in targets.iterrows():
        url = ("https://api.opencorporates.com/v0.4/companies/search?q="
               + uparse.quote(str(r["cb_company_name"]))
               + f"&jurisdiction_code={jc}&per_page=100&order=score")
        f = webfetch.CACHE_DEFAULT / (hashlib.sha1(url.encode()).hexdigest()
                                      + ".json")
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec["status"] != 200:
            continue
        try:
            comps = json.loads(rec["body"])["results"]["companies"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        for c in comps:
            c = c.get("company") or {}
            addr = str(c.get("registered_address_in_full") or "")
            pcs = pcre.findall(addr)
            pc = pcs[-1] if pcs else None
            city = None
            if pc:
                tail = addr.rsplit(pc, 1)[-1].strip(" ,\n")
                city = tail.split(",")[0].strip() or None
            names = [("current", c.get("name"))] + \
                    [("previous", p.get("company_name"))
                     for p in (c.get("previous_names") or [])
                     if isinstance(p, dict)]
            for route, nm in names:
                nm_clean = _PL_FORMS.sub(" ", nm or "") if jc == "pl" else (nm or "")
                nn = nname_reg(nm_clean)
                if nn != r["nn"]:
                    continue
                if route == "current":
                    counts[nn] += 1
                rows.append(dict(
                    nn=nn, route=route, reg_id=c.get("company_number"),
                    reg_name=c.get("name"), pc=pc, city=city,
                    creation=c.get("incorporation_date"),
                    status=c.get("current_status"),
                    dissolution=c.get("dissolution_date")))
    df = (pd.DataFrame(rows).drop_duplicates(["nn", "route", "reg_id"])
          if rows else pd.DataFrame())
    return df, counts


def cands_api(src: str, targets: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    rows, counts = [], Counter()
    for _, r in targets.iterrows():
        q = uparse.quote(str(r["cb_company_name"]))
        if src == "fi":
            url = f"https://avoindata.prh.fi/opendata-ytj-api/v3/companies?name={q}&page=1"
        elif src == "dk":
            url = f"https://cvrapi.dk/api?search={q}&country=dk"
        else:
            url = f"https://www.zefix.admin.ch/ZefixREST/api/v1/firm/search.json?name={q}"
        f = webfetch.CACHE_DEFAULT / (__import__("hashlib").sha1(url.encode())
                                      .hexdigest() + ".json")
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec["status"] != 200:
            continue
        try:
            data = json.loads(rec["body"])
        except json.JSONDecodeError:
            continue
        if src == "fi":
            for c in data.get("companies", []):
                names = [n.get("name", "") for n in (c.get("names") or [])]
                for nm in names[:3]:
                    nn = nname_reg(nm)
                    if nn and nn in set([r["nn"]]):
                        counts[nn] += 1
                        addr = ((c.get("addresses") or [{}]) or [{}])[0]
                        po = addr.get("postOffices") or []
                        city = po[0].get("city") if po and isinstance(po[0], dict) else None
                        rows.append(dict(
                            nn=nn, route="current",
                            reg_id=(c.get("businessId") or {}).get("value"),
                            reg_name=nm, pc=addr.get("postCode"),
                            city=city,
                            creation=(c.get("businessId") or {}).get("registrationDate"),
                            status=(c.get("status") or ""), dissolution=c.get("endDate")))
        elif src == "dk":
            nn = nname_reg(str(data.get("name", "")))
            if nn and nn == r["nn"]:
                counts[nn] += 1
                sd = str(data.get("startdate") or "")
                # cvrapi format "DD/MM - YYYY" -> ISO
                iso = None
                m = pd.Series([sd]).str.extract(r"(\d{2})/(\d{2}) - (\d{4})")
                if pd.notna(m.iloc[0, 0]):
                    iso = f"{m.iloc[0,2]}-{m.iloc[0,1]}-{m.iloc[0,0]}"
                rows.append(dict(nn=nn, route="current", reg_id=str(data.get("vat")),
                                 reg_name=data.get("name"), pc=str(data.get("zipcode")),
                                 city=data.get("city"), creation=iso,
                                 status="ended" if data.get("enddate") else "active",
                                 dissolution=None))
        else:  # ch
            for c in data.get("list", []):
                nn = nname_reg(c.get("name", ""))
                if nn and nn == r["nn"]:
                    counts[nn] += 1
                    rows.append(dict(nn=nn, route="current", reg_id=c.get("uid"),
                                     reg_name=c.get("name"), pc=None,
                                     city=c.get("legalSeat"),
                                     creation=c.get("sogcDate"),
                                     status=c.get("status"),
                                     dissolution=c.get("deleteDate")))
    return pd.DataFrame(rows), counts


def match_src(src: str) -> None:
    if src.startswith("oc-"):
        jc = src[3:]
        t = load_targets([OC_JC[jc]])
        print(f"{src}: {len(t)} targets ({int(t['precise'].sum())} precise)")
        cands, counts = cands_oc(jc, t)
        source, outfile = f"registry_{jc}_oc", f"registry_decisions_open_{jc}.csv"
        qa_label = f"{jc}_oc"
    else:
        land = {"no": "NO", "ee": "EE", "lv": "LV", "de": "DE",
                "fi": "FI", "dk": "DK", "ch": "CH", "be": "BE"}[src]
        t = load_targets([land])
        tset = set(t["nn"]) - {""}
        print(f"{src}: {len(t)} targets ({int(t['precise'].sum())} precise)")
        builders = {"no": cands_no, "ee": cands_ee, "lv": cands_lv,
                    "de": cands_de, "be": cands_be}
        if src in builders:
            cands, counts = builders[src](tset)
        else:
            cands, counts = cands_api(src, t)
        source, outfile = f"registry_{src}_open", f"registry_decisions_open_{src}.csv"
        qa_label = f"{src}_open"
    print(f"{src}: candidate rows {len(cands)}")
    if not len(cands):
        return
    dec = decide(cands, t, counts, source)
    dec.to_csv(REG_DIR / outfile, index=False)
    qa = qa_precise(dec, t, qa_label)
    qa.to_csv(REG_DIR / f"registry_qa_open_{src.replace('oc-', '')}.csv"
              if src.startswith("oc-") else REG_DIR / f"registry_qa_open_{src}.csv",
              index=False)
    print(f"{src}: decisions {dec['decision'].value_counts().to_dict()}")


def fetch_site(limit: int | None) -> None:
    """Own-website arm: reader on the firm's OWN site for firms still
    January-imputed after registry and web; identity is grade A by construction."""
    fa = pd.read_parquet(ROOT / "data/processed/firms_analysis.parquet",
                         columns=["firm_id", "founding_month_imputed_v2"])
    ids = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cb_website", "cbcsv_website"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    t = fa[fa["founding_month_imputed_v2"] == 1].merge(ids, on="firm_id", how="left")
    t["site"] = t["cb_website"].fillna(t["cbcsv_website"])
    t = t[t["site"].notna()]
    if limit:
        t = t.head(limit)
    print(f"site targets: {len(t)}")
    n_new = 0
    for i, (_, r) in enumerate(t.iterrows()):
        u = str(r["site"])
        if not u.startswith("http"):
            u = "http://" + u
        try:
            rec = webfetch.jina_read(u)
            n_new += 0 if rec.get("cached") else 1
        except webfetch.JinaCreditsExhausted:
            print("!! jina credits exhausted, top up and re-run (cache resumes)")
            sys.exit(2)
        if (i + 1) % 200 == 0:
            print(f"  site {i+1}/{len(t)} new={n_new}", flush=True)
    print(f"site: {len(t)} fetched, {n_new} new requests")


def match_site() -> None:
    import hashlib
    fa = pd.read_parquet(ROOT / "data/processed/firms_analysis.parquet",
                         columns=["firm_id", "t_year", "founding_month_imputed_v2"])
    ids = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cb_website", "cbcsv_website"],
                      dtype=str).rename(columns={"crunchbase_url": "firm_id"})
    t = fa[fa["founding_month_imputed_v2"] == 1].merge(ids, on="firm_id", how="left")
    t["site"] = t["cb_website"].fillna(t["cbcsv_website"])
    t = t[t["site"].notna()]
    out = []
    for _, r in t.iterrows():
        u = str(r["site"])
        if not u.startswith("http"):
            u = "http://" + u
        f = webfetch.CACHE_DEFAULT / (hashlib.sha1(("https://r.jina.ai/" + u)
                                                   .encode()).hexdigest() + ".json")
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec["status"] != 200:
            continue
        cands = [d for d in webfetch.extract_dates(rec["body"][:30000], "founding")
                 if d["precision"] in ("day", "month")
                 and int(r["t_year"]) - 1 <= d["year"] <= int(r["t_year"]) + 1]
        if not cands:
            continue
        ym = pd.Series([(d["year"], d["month"]) for d in cands])
        modal = ym.mode().iloc[0]
        d = next(c for c in cands if (c["year"], c["month"]) == modal)
        out.append(dict(firm_id=r["firm_id"], year=d["year"], month=d["month"],
                        grade="A", url=u, n_candidates=len(cands),
                        disagree=int(ym.nunique() > 1),
                        adopt=int(d["year"] == int(r["t_year"])),
                        context=d["context"]))
    w = pd.DataFrame(out)
    w.to_csv(REG_DIR / "web_site_final.csv", index=False)
    print(f"site months: {len(w)} firms; adoptable "
          f"{int(w['adopt'].sum()) if len(w) else 0}")


# ── Orbis arm ────────────────────────────────────────────────────────────────
# GUI exports (TUM licence) go by hand into data/scrape/orbis/: company slices
# (legal-form + incorporation-window filtered, status ALL incl. inactive) and a
# Zephyr deals export. Both shapes feed ONE pool; the export files serve founding
# verification only and deal facts never leave this arm (leakage rule).
ORBIS = ROOT / "data/scrape/orbis"


def _orbis_pool(land: str) -> pd.DataFrame:
    """All Orbis export CSVs → one deduped candidate pool for `land`.
    Land comes from the BvD-ID prefix (slices) / target country (deals), NOT
    from the export file names (they proved unreliable on day one)."""
    iso = {"UK": "GB"}.get(land, land)
    frames = []
    for p in sorted(ORBIS.rglob("*.csv")):   # exports are sorted into per-land subfolders
        head = pd.read_csv(p, nrows=1, sep=None, engine="python", dtype=str)
        cols = {c.strip('﻿"').strip() for c in head.columns}
        if not ({"Date of incorporation", "Target date of incorporation"} & cols):
            continue  # our upload/QA artifacts, not Orbis exports
        d = pd.read_csv(p, sep=None, engine="python", dtype=str)
        d.columns = [c.strip('﻿"').strip() or "rowno" for c in d.columns]
        if "Date of incorporation" in d.columns:      # company-slice shape
            d = d[d["BvD ID number"].astype(str).str.upper().str.startswith(iso)]
            prev = d.get("Previous company name Latin Alphabet")
            frames.append(pd.DataFrame(dict(
                reg_id=d["BvD ID number"], reg_name=d["Company name"],
                creation=d["Date of incorporation"], status=d["Status"],
                status_date=d["Status date"], city=d["City Latin Alphabet"],
                pc=d["Postcode Latin Alphabet"], web=d["Website address"],
                prev_name=prev if prev is not None else None)))
        else:                                          # deals shape (Zephyr)
            d = d[d["Target country code"].astype(str).str.upper().eq(iso)]
            d = d.drop_duplicates("Target name")
            frames.append(pd.DataFrame(dict(
                reg_id="deal" + d["Deal Number"].astype(str),
                reg_name=d["Target name"],
                creation=d["Target date of incorporation"], status=None,
                status_date=None, city=None, pc=d["Target postcode"],
                web=d["Target web site address"], prev_name=None)))
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool["creation"] = pd.to_datetime(pool["creation"], format="%d/%m/%Y",
                                      errors="coerce").dt.strftime("%Y-%m-%d")
    # PL: registry names spell out the legal form -> same prestrip as the OC arm
    _nn = ((lambda s: nname_reg(_PL_FORMS.sub(" ", str(s or ""))))
           if land == "PL" else nname_reg)
    pool["nn"] = pool["reg_name"].map(_nn)
    pool["pnn"] = pool["prev_name"].map(
        lambda s: _nn(s) if isinstance(s, str) and s.strip() else "")
    pool["dm"] = pool["web"].map(dom)
    # chunk overlaps + deals/slice doubles: BvD rows win over deal rows
    pool["is_deal"] = pool["reg_id"].str.startswith("deal")
    pool = (pool.sort_values("is_deal").drop_duplicates("reg_id")
            .drop_duplicates(["nn", "creation"]))
    # Status date ≈ dissolution date only for genuine terminal-failure statuses:
    # "Dissolved (merger or take-over)" is an M&A exit, not a closure (12% of
    # all non-active pool rows), and "Inactive (no precision)" carries an
    # observation date, not a closure date, so neither maps to dissolution
    # (the status itself stays recorded).
    st = pool["status"].fillna("").str.lower()
    closed = (st.str.contains("dissol|liquidat|bankrupt")
              & ~st.str.contains("merger|take-over|take over"))
    pool["dissolution"] = pool["status_date"].where(closed)
    return pool.reset_index(drop=True)


def match_orbis(land: str) -> None:
    t = load_targets([land])
    pool = _orbis_pool(land)
    if not len(pool):
        print(f"orbis-{land.lower()}: no export files for {land} yet")
        return
    print(f"orbis-{land.lower()}: pool {len(pool):,} candidates "
          f"({int(pool['is_deal'].sum()):,} deal-only), targets {len(t)} "
          f"({int(t['precise'].sum())} precise)")
    tset = set(t["nn"]) - {""}
    name_c = pool[pool["nn"].isin(tset)]
    counts = Counter(name_c["nn"])       # current-route only, like cands_oc
    cols = ["nn", "route", "reg_id", "reg_name", "pc", "city", "creation",
            "status", "dissolution"]
    cands = name_c.assign(route="current")[cols]
    # previous-name route (exports from 2019-20_2 on carry the column):
    # rename candidates are NEVER auto-confirmed (the UK lesson), decide()
    # turns them into uncertain_route = verdict material
    prev_c = pool[pool["pnn"].ne("") & pool["pnn"].isin(tset)]
    if len(prev_c):
        cands = pd.concat([cands, prev_c.assign(route="previous", nn=prev_c["pnn"])
                           [cols]], ignore_index=True)
        print(f"orbis-{land.lower()}: {len(prev_c)} previous-name candidate rows")
    source = f"registry_{land.lower()}_orbis"
    dec = decide(cands, t, counts, source)

    # Domain overlay: Orbis web address == CB domain is a second, name-
    # independent identity signal (ES legal names often ≠ brand: "The Smart
    # Lollipop" = HAPPY INNOVA SL). Auto-tier only when the name route came up
    # empty, the CB domain is unique among targets and exactly ONE Orbis
    # candidate sits in the ±1y window; everything else -> uncertain_domain.
    got = set(dec.loc[dec["decision"].isin(
        ["auto", "auto_unique", "promoted_year"]), "firm_id"])
    web = pd.read_csv(ROOT / "data/raw/companies_fat.csv",
                      usecols=["crunchbase_url", "cb_website"], dtype=str
                      ).rename(columns={"crunchbase_url": "firm_id"})
    td = t.merge(web, on="firm_id", how="left")
    td["dm"] = td["cb_website"].map(dom)
    td = td[td["dm"].notna() & ~td["firm_id"].isin(got)]
    td = td[~td["dm"].duplicated(keep=False)]          # shared CB domains out
    j = td.merge(pool[pool["dm"].notna()], on="dm", suffixes=("", "_ox"))
    j["yr"] = pd.to_datetime(j["creation"], errors="coerce").dt.year
    j = j[j["yr"].between(j["ty"] - 1, j["ty"] + 1)]
    over = []
    for fid, g in j.groupby("firm_id"):
        if g["reg_id"].nunique() > 1:
            over.append(dict(firm_id=fid, source=source,
                             decision="uncertain_domain",
                             n_candidates=int(g["reg_id"].nunique())))
            continue
        p = g.iloc[0]
        geo = ((p["pc"] != "" and norm_pc(p["pc_ox"]) == p["pc"])
               or (p["pco"] != "" and pc_out(p["pc_ox"]) == p["pco"])
               or (p["city"] != "" and city_eq(p["city_ox"], p["city"])))
        over.append(dict(firm_id=fid, source=source, decision="auto_domain",
                         n_candidates=1, route="domain",
                         registry_id=p["reg_id"], registry_name=p["reg_name"],
                         creation_date=p["creation"],
                         registry_status=p["status"],
                         dissolution_date=p["dissolution"],
                         geo_signal=bool(geo),
                         creation_year_eq=bool(p["yr"] == p["ty"])))
    if over:
        ov = pd.DataFrame(over)
        dec = pd.concat([dec[~dec["firm_id"].isin(set(ov["firm_id"]))], ov],
                        ignore_index=True)
    dec.to_csv(REG_DIR / f"registry_decisions_open_orbis_{land.lower()}.csv",
               index=False)
    print(f"orbis-{land.lower()}: decisions "
          f"{dec['decision'].value_counts().to_dict()}")
    qa1 = qa_precise(dec, t, f"{land.lower()}_orbis")
    dq = dec[dec["decision"] == "auto_domain"].assign(decision="auto")
    qa2 = (qa_precise(dq, t, f"{land.lower()}_orbis_domain")
           if len(dq) else pd.DataFrame())
    qas = [q for q in (qa1, qa2) if len(q)]
    if qas:
        pd.concat(qas, ignore_index=True).to_csv(
            REG_DIR / f"registry_qa_open_orbis_{land.lower()}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "match"])
    ap.add_argument("--src", default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.src == "site":
        fetch_site(a.limit) if a.stage == "fetch" else match_site()
        return
    srcs = (["no", "ee", "lv", "de", "fi", "dk", "ch", "be"]
            if a.src == "all" else
            ["oc-" + j for j in OC_JC] if a.src == "oc" else [a.src])
    for s in srcs:
        if s.startswith("oc-") and a.stage == "fetch":
            print(f"{s}: fetch runs via scripts/fetch_registry.py --stage oc")
            continue
        if a.stage == "fetch":
            if s == "be":
                kbo = sorted(EXT.glob("KboOpenData_*_Full"))
                print(f"be: manual download (KBO Open Data login); present: "
                      f"{kbo[-1].name if kbo else 'MISSING'}")
            elif s in BULK:
                fetch_bulk(s)
            else:
                fetch_api(s, a.limit)
        elif s.startswith("orbis-"):
            match_orbis(s[6:].upper())
        else:
            match_src(s)


if __name__ == "__main__":
    main()
