"""
Network arms of the founding-date enrichment: all HTTP lives here.

Bulk downloads are done BY HAND (documented, resumable; snapshots 2026-08-01):
  curl -sSL -C - -o data/external/ch-bulk/BasicCompanyDataAsOneFile-2026-08-01.zip \
    https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-08-01.zip
  curl -sSL -C - -o data/external/sirene/StockUniteLegale.parquet \
    https://static.data.gouv.fr/resources/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/20260801-073937/stock-stockunitelegale-parquet.parquet
  curl -sSL -C - -o data/external/sirene/StockEtablissement.parquet \
    https://static.data.gouv.fr/resources/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/20260801-074451/stock-stocketablissement-parquet.parquet
(The old files.data.gouv.fr/insee-sirene/ path is dead; resolve fresh URLs via https://www.data.gouv.fr/api/1/datasets/base-sirene-…)

Stages:
  verify-bulk  presence/size/row-count prints (soft pins).
  ch-api       Companies House search+profile for outputs/tables/
               registry_uk_api_targets.csv (closed firms + bulk misses);
               profiles fetched for every exact-nname_reg candidate.
  jina         web sweep over outputs/tables/web_sweep_targets.csv:
               QF1 + site:dealroom.co searches; ≤1 dealroom reader per firm,
               only when snippets carry no month; --max-new-requests brake;
               HTTP 402 aborts cleanly (top up credits, re-run to resume).
  oc           OpenCorporates research-plan lookups (open_data plan, 100k/day) for DE/AT/PT/LU: one
               /companies/search call per firm (jurisdiction-filtered,
               per_page=100, order=score); the search response already carries
               incorporation_date, dissolution_date, status, address and
               previous_names, so no detail calls are needed. Three 403s in a
               row (rate ceiling) abort cleanly; re-run resumes via cache.

Everything cache-first (data/scrape/cache/), so re-runs cost 0 requests.
Run:  conda run -n ml python scripts/fetch_registry.py --stage verify-bulk|ch-api|jina [--limit N] [--max-new-requests N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import parse as uparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import webfetch  # noqa: E402
from src.names import nname_reg  # noqa: E402
from match_registry import CH_CSV, SIR_ET, SIR_UL, REG_DIR, build_web_queries  # noqa: E402


def stage_verify_bulk() -> None:
    import pyarrow.parquet as pq
    assert CH_CSV.exists(), f"missing {CH_CSV}, see docstring for curl command"
    n_ch = sum(1 for _ in open(CH_CSV, "rb")) - 1
    print(f"ch-bulk: {CH_CSV.name}  {CH_CSV.stat().st_size/1e9:.2f} GB  rows={n_ch:,}")
    for p in (SIR_UL, SIR_ET):
        assert p.exists(), f"missing {p}, see docstring for curl command"
        print(f"sirene: {p.name}  {p.stat().st_size/1e9:.2f} GB  "
              f"rows={pq.ParquetFile(p).metadata.num_rows:,}")


def stage_ch_api(limit: int | None) -> None:
    t = pd.read_csv(REG_DIR / "registry_uk_api_targets.csv")
    if limit:
        t = t.head(limit)
    n_new = n_prof = 0
    t0 = time.time()
    for i, (_, r) in enumerate(t.iterrows()):
        rec = webfetch.ch_get("/search/companies?q="
                              + uparse.quote(str(r["cb_company_name"]))
                              + "&items_per_page=10")
        n_new += 0 if rec.get("cached") else 1
        if rec["status"] != 200:
            continue
        try:
            items = json.loads(rec["body"]).get("items", [])
        except json.JSONDecodeError:
            continue
        for it in items:
            if nname_reg(it.get("title", "")) == r["nn"]:
                prec = webfetch.ch_get(f"/company/{it.get('company_number')}")
                n_prof += 1
                n_new += 0 if prec.get("cached") else 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(t)}  new={n_new}  profiles={n_prof}  "
                  f"{int(time.time()-t0)}s", flush=True)
    print(f"ch-api done: {len(t)} targets, {n_new} new requests, "
          f"{n_prof} profile fetches, {int(time.time()-t0)}s")


def stage_jina(limit: int | None, max_new: int | None) -> None:
    t = pd.read_csv(REG_DIR / "web_sweep_targets.csv")
    if limit:
        t = t.head(limit)
    n_new = n_read = 0
    t0 = time.time()
    try:
        for i, (_, r) in enumerate(t.iterrows()):
            if max_new and n_new >= max_new:
                print(f"!! --max-new-requests {max_new} reached, stopping cleanly "
                      f"(re-run resumes via cache)")
                break
            if webfetch.host_banned("jina"):
                print("!! jina failure rate >30%, aborting (re-run resumes)")
                break
            hits_all, month_ok = [], False
            for qid, q in build_web_queries(str(r["cb_company_name"]),
                                            str(r["city"] or "")):
                rec = webfetch.jina_search(q)
                n_new += 0 if rec.get("cached") else 1
                for h in webfetch.parse_search_results(rec):
                    hits_all.append(h)
                    for d in webfetch.extract_dates(
                            h["title"] + " — " + h["snippet"], "founding"):
                        if d["precision"] in ("day", "month"):
                            month_ok = True
            if not month_ok:
                u = webfetch.pick_dealroom_url(hits_all)
                if u:
                    rec = webfetch.jina_read(u)
                    n_read += 1
                    n_new += 0 if rec.get("cached") else 1
            if (i + 1) % 100 == 0:
                el = int(time.time() - t0)
                print(f"  {i+1}/{len(t)}  new={n_new}  reads={n_read}  {el}s  "
                      f"(~{el/(i+1)*(len(t)-i-1)/60:.0f} min left)", flush=True)
    except webfetch.JinaCreditsExhausted as e:
        print(f"!! {e}")
        sys.exit(2)
    print(f"jina done: {len(t)} targets, {n_new} new requests, {n_read} reads, "
          f"{int(time.time()-t0)}s")


# OC coverage (national-champion probes): dated registers =
# de (partial), lu, dk (full CVR!), pl (KRS), ie (CRO, incl. dissolutions);
# ch = names + dissolution dates only (no incorporation dates); at/pt = empty
# jurisdiction codes; se/nl/es = names without dates; it/lt = no data. Fetched:
# the dated lands + ch (dissolution lever) + at/pt (to document the emptiness).
OC_JC = {"DE": "de", "AT": "at", "PT": "pt", "LU": "lu",
         "DK": "dk", "PL": "pl", "IE": "ie", "CH": "ch"}


def stage_uk_renames(limit: int | None) -> None:
    """Rename hunt: CH search matches PREVIOUS company names and returns them
    in item `snippet` (·-separated); the ch-api stage only
    fetched profiles for exact CURRENT-name hits, discarding renamed firms.
    This walks the CACHED searches (no new searches) and fetches the missing
    profiles for snippet hits, cache-first; ~2 req/s stays under the CH limit."""
    import hashlib
    t = pd.read_csv(REG_DIR / "registry_uk_api_targets.csv")
    if limit:
        t = t.head(limit)
    n_hit = n_new = 0
    t0 = time.time()
    for i, (_, r) in enumerate(t.iterrows()):
        url = ("https://api.company-information.service.gov.uk/search/companies?q="
               + uparse.quote(str(r["cb_company_name"])) + "&items_per_page=10")
        f = webfetch.CACHE_DEFAULT / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec["status"] != 200:
            continue
        try:
            items = json.loads(rec["body"]).get("items", [])
        except json.JSONDecodeError:
            continue
        for it in items:
            if nname_reg(it.get("title", "")) == r["nn"]:
                continue  # profile already fetched by the ch-api stage
            sn = it.get("snippet") or ""
            if not any(nname_reg(p) == r["nn"] for p in sn.split("·")):
                continue
            prec = webfetch.ch_get(f"/company/{it.get('company_number')}")
            n_hit += 1
            if not prec.get("cached"):
                n_new += 1
                time.sleep(0.5)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(t)}  snippet-hits={n_hit}  new={n_new}  "
                  f"{int(time.time()-t0)}s", flush=True)
    print(f"uk-renames done: {len(t)} cached searches, {n_hit} snippet hits, "
          f"{n_new} new profile fetches, {int(time.time()-t0)}s")


def stage_oc(limit: int | None, max_new: int | None) -> None:
    from match_registry import load_targets
    st = webfetch.oc_get("/account_status", force=True)
    if st["status"] == 200:
        print(f"oc account_status: {st['body'][:220]}")
    t = load_targets(list(OC_JC))
    if limit:
        t = t.head(limit)
    n_new = n403 = 0
    t0 = time.time()
    for i, (_, r) in enumerate(t.iterrows()):
        if max_new and n_new >= max_new:
            print(f"!! --max-new-requests {max_new} reached, stopping cleanly")
            break
        rec = webfetch.oc_get(
            "/companies/search?q=" + uparse.quote(str(r["cb_company_name"]))
            + f"&jurisdiction_code={OC_JC[r['land']]}&per_page=100&order=score")
        if rec.get("cached"):
            continue
        n_new += 1
        if rec["status"] == 403:  # rate ceiling (not cached as terminal)
            n403 += 1
            if n403 >= 3:
                print("!! three 403s, rate ceiling hit, stopping cleanly "
                      "(re-run resumes via cache)")
                break
            time.sleep(30)
        else:
            n403 = 0
            time.sleep(0.15)  # polite pacing; limits allow far more
        if (i + 1) % 200 == 0:
            el = int(time.time() - t0)
            print(f"  {i+1}/{len(t)}  new={n_new}  {el}s "
                  f"(~{el/(i+1)*(len(t)-i-1)/60:.0f} min left)", flush=True)
    print(f"oc done: {len(t)} targets, {n_new} new requests, "
          f"{int(time.time()-t0)}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["verify-bulk", "ch-api", "jina", "oc", "uk-renames"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-requests", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "verify-bulk":
        stage_verify_bulk()
    elif a.stage == "ch-api":
        assert webfetch.KEYS.get("CH_API_KEY"), "CH_API_KEY missing (keys.env)"
        stage_ch_api(a.limit)
    elif a.stage == "uk-renames":
        assert webfetch.KEYS.get("CH_API_KEY"), "CH_API_KEY missing (keys.env)"
        stage_uk_renames(a.limit)
    elif a.stage == "oc":
        assert webfetch.KEYS.get("OC_API_TOKEN"), "OC_API_TOKEN missing (keys.env)"
        stage_oc(a.limit, a.max_new_requests)
    else:
        assert webfetch.KEYS.get("JINA_API_KEY"), "JINA_API_KEY missing (keys.env)"
        stage_jina(a.limit, a.max_new_requests)


if __name__ == "__main__":
    main()
