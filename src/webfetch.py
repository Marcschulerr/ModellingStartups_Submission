"""
Shared HTTP machinery for the web and registry enrichment.

Design rules, learned in a small pilot before the full sweep:
  - cache-first sha1-JSON store, default data/scrape/cache/, so identical
    queries replay for free and every run is resumable
  - HTTP 402 (Jina credits exhausted) raises JinaCreditsExhausted at fetch
    time, so runs abort cleanly; 402 responses are cached but treated as
    retryable on replay
  - reader escalation is restricted to dealroom.co profile pages (PitchBook
    readers were dropped: token-heavy reads with year-only content)
  - date extraction only within +-160 chars of a trigger keyword; year-level
    plausibility gates live in the callers (kept wrong-entity at 0.0 %)

Keys: env vars JINA_API_KEY / CH_API_KEY first, then ROOT/keys.env. Keys never
appear in logs, cache filenames or cached bodies. Identifier fields are used
read-only for query building and identity verification.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib import error as urlerr
from urllib import parse as uparse
from urllib import request as ureq

import numpy as np
import pandas as pd

from .names import dom, nname  # noqa: F401, re-exported for callers

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEFAULT = ROOT / "data/scrape/cache"


class JinaCreditsExhausted(RuntimeError):
    """HTTP 402 from Jina: top up credits, then simply re-run (cache resumes)."""


def load_keys() -> dict:
    keys = {k: os.environ.get(k, "") for k in ("JINA_API_KEY", "CH_API_KEY")}
    envf = ROOT / "keys.env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if not keys.get(k.strip()):
                    keys[k.strip()] = v.strip()
    return keys


KEYS = load_keys()

_LAST: dict[str, float] = {}
_HEALTH: dict[str, deque] = {}


def _interval(host: str) -> float:
    if "jina.ai" in host:
        return 0.7 if KEYS.get("JINA_API_KEY") else 3.5
    return 0.6


def http_get(url: str, headers: dict, timeout: int = 45) -> dict:
    host = uparse.urlparse(url).netloc
    for attempt in range(4):
        wait = _interval(host) - (time.monotonic() - _LAST.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.monotonic()
        try:
            with ureq.urlopen(ureq.Request(url, headers=headers), timeout=timeout) as r:
                _HEALTH.setdefault(host, deque(maxlen=50)).append(1)
                return {"status": r.status,
                        "body": r.read().decode("utf-8", errors="replace")}
        except urlerr.HTTPError as e:
            _HEALTH.setdefault(host, deque(maxlen=50)).append(0)
            if e.code in (429, 500, 502, 503) and attempt < 3:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit()
                           else 2 * 2 ** attempt + np.random.default_rng().uniform(0, 1))
                continue
            return {"status": e.code,
                    "body": e.read().decode("utf-8", "replace")[:2000]}
        except Exception as e:  # timeout, TLS, DNS
            _HEALTH.setdefault(host, deque(maxlen=50)).append(0)
            if attempt < 3:
                time.sleep(2 * 2 ** attempt)
                continue
            return {"status": -1, "body": f"error: {e}"}
    return {"status": -1, "body": "unreachable"}


def host_banned(host_part: str) -> bool:
    for host, dq in _HEALTH.items():
        if host_part in host and len(dq) >= 30 and (1 - sum(dq) / len(dq)) > 0.30:
            return True
    return False


def cached_get(url: str, headers: dict, cache_dir: Path = CACHE_DEFAULT,
               force: bool = False, fetch_url: str | None = None,
               terminal: tuple = (200, 403, 404, 422)) -> dict:
    # cache key and stored record use `url`; the wire request uses `fetch_url`
    # when given, which lets query-param credentials (OpenCorporates api_token)
    # stay out of cache keys AND cache files. `terminal` = statuses accepted on
    # replay (OC passes (200, 404): its 403 means rate ceiling, retry next run).
    f = cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if f.exists() and not force:
        rec = json.loads(f.read_text())
        if rec["status"] in terminal:
            rec["cached"] = True
            return rec
    rec = http_get(fetch_url or url, headers)
    if rec["status"] == 402:
        raise JinaCreditsExhausted(f"402 on {url[:80]}, top up Jina credits and re-run")
    rec["url"], rec["ts"] = url, time.strftime("%Y-%m-%d %H:%M:%S")
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, ensure_ascii=False))
    rec["cached"] = False
    return rec


def jina_search(q: str, cache_dir: Path = CACHE_DEFAULT) -> dict:
    h = {"Accept": "application/json", "X-Respond-With": "no-content",
         "User-Agent": "thesis-research"}
    if KEYS.get("JINA_API_KEY"):
        h["Authorization"] = "Bearer " + KEYS["JINA_API_KEY"]
    return cached_get("https://s.jina.ai/?q=" + uparse.quote(q), h, cache_dir)


def jina_read(url: str, cache_dir: Path = CACHE_DEFAULT) -> dict:
    h = {"Accept": "text/plain", "X-Return-Format": "markdown",
         "User-Agent": "thesis-research"}
    if KEYS.get("JINA_API_KEY"):
        h["Authorization"] = "Bearer " + KEYS["JINA_API_KEY"]
    return cached_get("https://r.jina.ai/" + url, h, cache_dir)


def oc_get(path_qs: str, cache_dir: Path = CACHE_DEFAULT, force: bool = False) -> dict:
    """OpenCorporates v0.4 GET (https, version pinned in the URL, api_token as
    query param). The token goes only on the wire; cache key and cache file
    carry the clean URL. 403 = rate ceiling and is NOT terminal (refetched on
    the next run)."""
    clean = "https://api.opencorporates.com/v0.4" + path_qs
    sep = "&" if "?" in path_qs else "?"
    return cached_get(clean, {"User-Agent": "tum-thesis-research"}, cache_dir,
                      force=force,
                      fetch_url=clean + sep + "api_token=" + KEYS["OC_API_TOKEN"],
                      terminal=(200, 404))


def ch_get(path: str, cache_dir: Path = CACHE_DEFAULT) -> dict:
    tok = base64.b64encode((KEYS["CH_API_KEY"] + ":").encode()).decode()
    return cached_get("https://api.company-information.service.gov.uk" + path,
                      {"Authorization": "Basic " + tok}, cache_dir)


def parse_search_results(rec: dict) -> list[dict]:
    if rec["status"] != 200:
        return []
    try:
        data = json.loads(rec["body"])
    except json.JSONDecodeError:
        return []
    items = data.get("data") or data.get("results") or []
    if isinstance(items, dict):
        items = items.get("results", [])
    return [{"title": str(it.get("title", "")), "url": str(it.get("url", "")),
             "snippet": str(it.get("description", "") or it.get("content", ""))}
            for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def pick_dealroom_url(hits: list[dict]) -> str | None:
    for h in hits:
        u = h["url"].lower()
        if "dealroom.co" in u and "linkedin.com" not in u:
            return h["url"]
    return None


# ── date extraction ─────────────────────────────────────────────────────────────
MONTHS: dict[str, int] = {}
for _i, _names in enumerate([
    ["january", "jan", "januar", "janvier", "enero", "gennaio", "januari", "janeiro"],
    ["february", "feb", "februar", "fevrier", "février", "febrero", "febbraio",
     "februari"],
    ["march", "mar", "maerz", "märz", "mars", "marzo", "maart"],
    ["april", "apr", "avril", "abril", "aprile"],
    ["may", "mai", "mayo", "maggio", "mei", "maj"],
    ["june", "jun", "juni", "juin", "junio", "giugno"],
    ["july", "jul", "juli", "juillet", "julio", "luglio"],
    ["august", "aug", "aout", "août", "agosto", "augustus", "augusti"],
    ["september", "sep", "sept", "septembre", "septiembre", "settembre"],
    ["october", "oct", "oktober", "octobre", "octubre", "ottobre", "okt"],
    ["november", "nov", "novembre", "noviembre"],
    ["december", "dec", "dez", "dezember", "decembre", "décembre", "diciembre",
     "dicembre"],
], start=1):
    for _n in _names:
        MONTHS[_n] = _i
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

TRIG_FOUND = ["founded", "established", "incorporated", "launched", "since",
              "founding", "year founded", "founded date", "launch date", "creation",
              "gegruendet", "gegründet", "gruendung", "gründung", "fondee", "fondée",
              "creee", "créée", "immatricul", "fundada", "fondata", "opgericht",
              "grundat", "grundlagt", "date of incorporation"]
TRIG_CLOSE = ["closed", "shut down", "shut its doors", "ceased", "dissolved",
              "liquidation", "insolvency", "insolvent", "bankrupt", "wound up",
              "winding up", "out of business", "defunct", "insolvenz", "aufgeloest",
              "aufgelöst", "eingestellt", "liquidation judiciaire", "radiee",
              "radiée", "cessation", "concurso", "fallimento", "faillissement",
              "konkurs", "dissolution"]

PAT_ISO = re.compile(r"\b(20[0-2]\d)[-./](0?[1-9]|1[0-2])(?:[-./](0?[1-9]|[12]\d|3[01]))?\b")
PAT_DMY = re.compile(r"\b(0?[1-9]|[12]\d|3[01])[./](0?[1-9]|1[0-2])[./](20[0-2]\d)\b")
PAT_MON = re.compile(r"\b(" + MONTH_ALT + r")\.?,?\s+(?:of\s+)?(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(20[0-2]\d)\b", re.I)
PAT_DMON = re.compile(r"\b(\d{1,2})\.?\s+(" + MONTH_ALT + r")\.?,?\s+(20[0-2]\d)\b", re.I)
PAT_YR = re.compile(r"\b(20[0-2]\d)\b")


def extract_dates(text: str, target: str) -> list[dict]:
    """All plausible dates near a trigger keyword; precision day > month > year."""
    text = re.sub(r"\s+", " ", str(text))
    low = text.lower()
    out = []
    for t in (TRIG_FOUND if target == "founding" else TRIG_CLOSE):
        for m in re.finditer(re.escape(t), low):
            a, b = max(0, m.start() - 160), min(len(text), m.end() + 160)
            win = text[a:b]
            found = []
            for mm in PAT_ISO.finditer(win):
                found.append((int(mm[1]), int(mm[2]), int(mm[3]) if mm[3] else None,
                              "day" if mm[3] else "month"))
            for mm in PAT_DMY.finditer(win):
                found.append((int(mm[3]), int(mm[2]), int(mm[1]), "day"))
            for mm in PAT_DMON.finditer(win):
                found.append((int(mm[3]), MONTHS[mm[2].lower()], int(mm[1]), "day"))
            for mm in PAT_MON.finditer(win):
                found.append((int(mm[3]), MONTHS[mm[1].lower()],
                              int(mm[2]) if mm[2] else None,
                              "day" if mm[2] else "month"))
            if not found:
                found = [(int(mm[1]), None, None, "year") for mm in PAT_YR.finditer(win)]
            for y, mo, d, prec in found:
                out.append(dict(year=y, month=mo, day=d, precision=prec,
                                trigger=t, context=win.strip()[:200]))
    return out


def grade(row, url: str, text: str) -> str:
    """Identity grade: A domain/linkedin/email-domain, B name+city, C name only."""
    ud = dom(url)
    blob = f"{url} {text}".lower()
    if (row.get("domain") and (ud == row["domain"] or str(row["domain"]) in blob)) or \
       (row.get("email_dom") and str(row["email_dom"]) in blob) or \
       (isinstance(row.get("li_slug"), str) and row["li_slug"] and row["li_slug"] in blob):
        return "A"
    nn = nname(row["cb_company_name"])
    if nn and nn in nname(text[:400]) and str(row.get("city") or "").lower() \
       and str(row["city"]).lower() in blob:
        return "B"
    return "C"
