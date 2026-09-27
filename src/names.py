"""
Canonical name and domain normalization, in one place so the CORDIS matcher,
the grant-match validator and the registry-enrichment matcher share a single
implementation.

  nname()      normalized legal name for CORDIS matching
  dom()        registrable website domain
  fold()       lowercase + NFKD accent-strip + alnum/space collapse
  nname_reg()  nname with accent folding and registry legal forms stripped
               (Sirene: SASU/EURL/SCI/...; CH: CIC/LP), used for registry names
               AND the CB side when matching against registries
  city_eq()    fold-based containment compare for PostTown/commune vs CB city
               (Sirene communes carry arrondissement suffixes: "PARIS 8E ...")

`scripts/match_cordis.py` re-exports nname/dom, so `from match_cordis import
nname, dom` keeps working. `validate_grant_match.py` keeps its own local
`fold` for its geo logic; new code imports `fold` from here.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

import pandas as pd

SUFF = (r"\b(gmbh|ag|ug|kg|co|kgaa|sas|sarl|sa|sl|slu|srl|spa|bv|nv|ab|aps|as|asa|oy|oyj"
        r"|ltd|limited|plc|llp|llc|inc|corp|company|holding|group|services|technologies"
        r"|technology|tech)\b")

REG_SUFF = (r"\b(sasu|eurl|sci|snc|scop|selarl|selas|sel|sca|scs|sarlu"
            r"|community interest company|cic|lp|lbg|rtm"
            r"|ou|ivs|ay|tmi|mtu|enk|hb|kb)\b")  # incl. EE/DK/FI/NO/SE forms

# CORDIS legal names carry LT/LV/EE/PT forms the SUFF list lacks (UAB GENOMIKA,
# SIA ATOM TECH, GSCAN OU, ... LDA); without them several verified treated firms
# would be name-route-blind for that reason alone.
CORDIS_SUFF = r"\b(uab|sia|ou|lda|unipessoal)\b"

# Insolvency/liquidation status phrases appended to legal names. 73 % of the
# dead Italian firms in the Orbis pool carry "IN LIQUIDAZIONE" in the name, a
# survival-correlated blind spot of the name route if left unstripped.
STATUS_SUFF = (r"\b(in liquidazione|in fallimento|in concordato preventivo"
               r"|in scioglimento|in liquidation|in administration|en liquidation"
               r"|en liquidacion|i likvidation|i konkurs|in insolvenz)\b")

# Private multi-registrant registries: two firms can share e.g. *.uk.com or
# *.eu.org without any relation, so keep three domain labels for these.
PRIVATE_REGISTRIES = {"uk.com", "eu.com", "eu.org", "us.com", "de.com", "gb.com", "gb.net",
                      "uk.net", "se.net", "co.com", "br.com", "cn.com", "ar.com",
                      "kr.com", "jpn.com", "ru.com", "sa.com", "za.com", "web.com"}


def nname(s) -> str:
    # dots and slashes are DELETED (not spaced) before suffix stripping, so dotted
    # legal forms collapse into strippable tokens: "USONO B.V." -> "usono bv" -> "usono"
    s = str(s).lower()
    s = re.sub(r"[&+]", " and ", s)
    s = re.sub(r"[./\\']", "", s)
    # accents are FOLDED (Ä -> a), not deleted: "ÄIO" must be able to meet
    # CORDIS "AIO TECH OU"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(STATUS_SUFF, " ", s)
    s = re.sub(SUFF, " ", s)
    s = re.sub(CORDIS_SUFF, " ", s)
    return re.sub(r"\s", "", s)


def fold(s) -> str:
    """lowercase, strip accents, collapse to alnum+space"""
    if isinstance(s, (list, tuple)):
        s = s[0] if len(s) else ""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def nname_reg(s) -> str:
    """nname plus accent folding and registry legal forms, for registry matching."""
    s = re.sub(r"[&+]", " and ", str(s))
    s = re.sub(r"[./\\']", "", s)
    s = fold(s)
    s = re.sub(STATUS_SUFF, " ", s)   # "... SRL IN LIQUIDAZIONE" must still match
    s = re.sub(SUFF, " ", s)
    s = re.sub(REG_SUFF, " ", s)
    return re.sub(r"\s", "", s)


def city_eq(a, b) -> bool:
    """Registry PostTown/commune vs CB city: fold + containment (min 4 chars)."""
    fa, fb = fold(a), fold(b)
    if len(fa) < 4 or len(fb) < 4:
        return bool(fa) and fa == fb
    return fa in fb or fb in fa


def dom(u):
    if pd.isna(u) or not str(u).strip():
        return None
    u = str(u).strip().lower()
    if not u.startswith("http"):
        u = "http://" + u
    try:
        h = urlparse(u).netloc.split(":")[0].removeprefix("www.")
    except ValueError:
        return None
    p = h.split(".")
    if len(p) >= 3 and (p[-2] in {"co", "com", "org", "net", "ac", "gov"}
                        or ".".join(p[-2:]) in PRIVATE_REGISTRIES):
        return ".".join(p[-3:])
    return ".".join(p[-2:]) if len(p) >= 2 else None
