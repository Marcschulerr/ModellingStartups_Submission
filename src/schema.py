"""
Closed schema of data/processed/firms_analysis.parquet: the single source of
truth for what the analysis table contains.

Purpose:
  1. Leak-proofing: `validate()` asserts the final frame's columns equal this
     list EXACTLY (order + names). End-state Crunchbase columns
     (cbcsv_funding_status, investors, employees, revenue, articles,
     patents, social media, ...) and cbm_reasoning can never slip in; they
     are simply not in the list. Scoped exception: the "Survival & exit"
     group carries status-at-cutoff flags derived from cb_operating_status,
     acquisition status, IPO status and website liveness, usable ONLY as
     survival outcomes or validation, never as covariates or outcomes in the
     funding models.
  2. Traceability: every column carries its definition, source and cleaning
     rule; `render_markdown()` writes docs/variables.md from this list, so
     documentation cannot drift from code.

Convention: all flags are int8 0/1 (identical in parquet, the CSV mirror and R).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str
    definition: str
    source: str
    cleaning: str = ""
    group: str = ""


def _c(group):
    def make(name, dtype, definition, source, cleaning=""):
        return Col(name, dtype, definition, source, cleaning, group)
    return make


_id = _c("Identity")
_fo = _c("Founding")
_ci = _c("Circularity")
_ts = _c("Team & sector controls")
_o36 = _c("Outcomes 36m")
_gr = _c("EU grants (verified CORDIS match)")
_sv = _c("Survival & exit (status at data cutoff, outcome-only)")
_v2 = _c("Founding v2 (registry/web enrichment, parallel columns)")
_o2 = _c("Outcomes 36m under the v2 founding month (parallel columns)")

WIN36 = "36m window = founding month + 35 following months (year-precision: Jan–Dec(Y+2))"

# Top-20 industry tags by frequency (pinned from cbcsv_industries, 24,891
# firms with tags). Change deliberately, never silently.
TOP_INDUSTRY_TAGS: dict[str, str] = {
    "Software": "ind_software",
    "Artificial Intelligence (AI)": "ind_ai",
    "Information Technology": "ind_information_technology",
    "Health Care": "ind_health_care",
    "SaaS": "ind_saas",
    "Financial Services": "ind_financial_services",
    "FinTech": "ind_fintech",
    "E-Commerce": "ind_ecommerce",
    "Biotechnology": "ind_biotechnology",
    "Manufacturing": "ind_manufacturing",
    "Machine Learning": "ind_machine_learning",
    "Internet": "ind_internet",
    "Blockchain": "ind_blockchain",
    "Food and Beverage": "ind_food_beverage",
    "Apps": "ind_apps",
    "Medical": "ind_medical",
    "Analytics": "ind_analytics",
    "Mobile Apps": "ind_mobile_apps",
    "Finance": "ind_finance",
    "Marketplace": "ind_marketplace",
}

SCHEMA: list[Col] = [
    # ── Identity ────────────────────────────────────────────────────────────
    _id("firm_id", "string", "Unique firm key", "lean.crunchbase_url"),
    _id("cb_company_name", "string", "Company name (QA convenience)", "lean.cb_company_name"),
    _id("cb_source_file", "string", "Scrape source file (audit trail for land)", "lean.cb_source_file"),
    _id("land", "string", "Sampling-frame country, 20 values", "lean.cb_source_file",
        "leading letter prefix, e.g. 'UK_2018_1' → 'UK'; asserted ⊆ 20-land set"),
    # ── Founding ────────────────────────────────────────────────────────────
    _fo("founded_date", "date", "Founding date", "fat.cbcsv_founded_date", "100% filled; parse errors=raise"),
    _fo("founded_precision", "string", "day / month / year", "fat.cbcsv_founded_date_precision"),
    _fo("t_year", "int16", "Founding year (analysis cohorts 2016–2022, rows outside dropped, "
        "see funnel)", "year(founded_date)", "asserted == lean.cb_founded_year for all rows"),
    _fo("founding_month_idx", "int32", "Founding month as integer index year*12+(month-1)",
        "founded_date", "year-precision firms imputed to January of t_year"),
    _fo("founding_month_imputed", "int8 0/1", "1 = founding month imputed (precision == 'year')", "derived"),
    # ── Circularity ─────────────────────────────────────────────────────────
    _ci("cbm_type", "string", "Raw business-model type incl. 'parse_error'", "lean.cbm_type"),
    _ci("circ", "int8 0/1", "1 = cbm_type ∈ {Design,Platform,Nature,Service,Waste}-based",
        "lean.cbm_type", "parse_error recoded to 0 here, exactly once"),
    _ci("circ_subtype", "string", "Circular sub-type: Design/Platform/Nature/Service/Waste; "
        "'None' for non-circular; the mechanism dimension of the gap RQ", "lean.cbm_type"),
    _ci("parse_error", "int8 0/1", "1 = cbm_type == 'parse_error'", "lean.cbm_type"),
    _ci("cbm_confidence", "float32", "Classifier confidence: robustness cutoffs ONLY, never a covariate",
        "lean.cbm_confidence"),
    # ── Team & sector ───────────────────────────────────────────────────────
    _ts("n_founders", "Int16", "Number of founders (~75% filled); composition dimension",
        "fat.cbcsv_number_of_founders"),
    _ts("industries", "string", "Raw industry tag list (traceability source of the ind_* dummies)",
        "fat.cbcsv_industries"),
    _ts("n_industries", "Int16", "Number of industry tags (<NA> = no tags)", "derived"),
]
for _tag, _slug in TOP_INDUSTRY_TAGS.items():
    SCHEMA.append(_ts(_slug, "int8 0/1", f"Industry tag '{_tag}'", "fat.cbcsv_industries",
                      "multi-hot over the pinned top-20 tag list; 0 also when no tags at all"))
SCHEMA += [
    # ── Outcomes 36m ────────────────────────────────────────────────────────
    _o36("n_fund_events_36", "int16", f"Number of months with ≥1 funding round in window; {WIN36}",
         "panel_fat.funding_event"),
    _o36("n_private_months_36", "Int16", "Number of window months with ≥1 non-Grant round "
         "(private analog of n_fund_events_36, for the success-flip "
         "decomposition)", "panel_fat.funding_types"),
    _o36("fund36", "int8 0/1", "1[any funding round in window]; includes CB-recorded grants "
         "(anchor-comparable to the frozen reference artifact)", "derived"),
    _o36("fund36_private", "int8 0/1", "MAIN OUTCOME: 1[any round in window whose type is not "
         "'Grant'], market financing only", "panel_fat.funding_types",
         "a month counts as private if ANY of its round types ≠ Grant"),
    _o36("grant36_cb", "int8 0/1", "1[any 'Grant'-type round in window]; CB-recorded grants incl. "
         "national ones; complements the CORDIS block", "panel_fat.funding_types"),
    _o36("sum_fund_usd_36", "float64", "Σ funding_total_usd in window (missing amounts = 0)",
         "panel_fat.funding_total_usd", "14.7% of events carry no positive amount, an undercount caveat"),
    _o36("logfund36", "float64", "log(1 + sum_fund_usd_36)", "derived"),
    _o36("sum_fund_usd_private_36", "float64", "Σ funding_total_usd over window months with ≥1 "
         "non-Grant round", "derived", "mixed grant+private months count fully as private "
         "(monthly amounts are not separable by round; mixed-month count reported at build)"),
    _o36("logfund36_private", "float64", "log(1 + sum_fund_usd_private_36)", "derived"),
    _o36("first_private_offset_m", "Int32", "Months from founding to FIRST private (non-Grant) round, "
         "uncensored horizon; <NA> = none observed by 2026-03", "derived"),
    _o36("success36", "int8 0/1", "1[≥3 event-months ∨ a month > $10M ∨ M&A in window] "
         "(BA-compatible composite, kept as SECONDARY outcome)", "derived"),
    _o36("success36_bigmonth", "int8 0/1", "Component: 1[a window month > $10M], kept to "
         "decompose the success flip (the third component, ≥3 rounds, derives "
         "from n_fund_events_36)", "derived"),
    _o36("success36_ma", "int8 0/1", "Component: 1[≥1 M&A event in window]", "panel_fat.ma_event"),
    # ── Grants ──────────────────────────────────────────────────────────────
    _gr("has_grant", "int8 0/1", "1 = ≥1 individually verified CORDIS participation "
        "(H2020/Horizon Europe)",
        "outputs/tables/grant_match_final.csv"),
    _gr("has_grant_uncertain", "int8 0/1", "1 = only unresolved match pairs (excluded from treated "
        "set, never mixed in silently)", "grant_match_final.csv"),
    _gr("first_grant_date", "date", "Earliest grant date: ecSignatureDate, else project startDate",
        "cordis project.csv"),
    _gr("first_grant_scheme", "string", "fundingScheme of the first grant (raw, e.g. SME-1, "
        "HORIZON-EIC-ACC-BF; '-BF' = blended finance incl. EIC equity)", "cordis project.csv"),
    _gr("first_grant_scheme_group", "string", "SME-Instrument / EIC / consortium-R&D",
        "derived", "SME* → SME-Instrument; *EIC* → EIC; else consortium-R&D"),
    _gr("first_grant_mono", "int8 0/1", "1 = first grant's project has exactly one beneficiary "
        "(firm-targeted instrument)", "cordis organization.csv"),
    _gr("ec_total_eur", "float64", "Σ EC contribution (EUR) over verified participations",
        "cordis organization.csv", "decimal commas normalized"),
    _gr("n_grant_participations", "Int16", "Number of verified firm×project participations",
        "grant_match_final.csv"),
    _gr("grant_match_tier", "string", "verified_auto (2 independent signals) / verified_llm "
        "(pair-by-pair review); worst tier among the firm's verified pairs; <NA> = no grant",
        "grant_match_final.csv"),
    # ── Survival & exit ─────────────────────────────────────────────────────
    # Scoped exception to the end-state ban: outcome and validation use ONLY,
    # never covariates or outcomes of the funding models.
    _sv("status_closed", "int8 0/1", "1 = operating status 'Closed' at export (2026-03); UNDATED, "
        "the estimand is P(closed by cutoff), exposure handled via land×cohort FE",
        "lean.cb_operating_status", "asserted binary Active/Closed; 62% of closures carry no "
        "closed date, hence no time-to-failure"),
    _sv("exit_any", "int8 0/1", "1 = 'Was Acquired' ∨ IPO status Public/Delisted at export "
        "(Kennedy et al. 2024 'exit' analog, undated variant)",
        "fat.cbcsv_acquisition_status + fat.cbcsv_ipo_status"),
    _sv("ma_any_dated", "int8 0/1", "1 = ≥1 dated M&A event month at offset ≥0 (any horizon); "
        "the dated subset of exit_any's M&A part", "panel_fat.ma_event"),
    _sv("first_ma_offset_m", "Int32", "Months from founding to FIRST dated M&A event, uncensored "
        "horizon; <NA> = none observed by 2026-03 (KM curve input)", "panel_fat.ma_event"),
    _sv("site_dead_2026_03", "Int8 0/1", "1 = website scrape failed at 2026-03-27 (liveness proxy: "
        "failed-share 53.6% among Closed vs 21.4% among Active); <NA> = no website/skipped; "
        "validation and 'dead by any signal' robustness only", "fat.jina_scrape_status"),
    # ── Founding v2 (parallel registry/web enrichment, baseline untouched) ──
    _v2("founding_month_idx_v2", "int32", "v2 founding month index: CB month where precise; "
        "verified registry incorporation month (11 registers + Orbis GUI arm) or verified web "
        "month (grade A/B, modal) where CB is year-only; January fallback else. Inside t_year "
        "OR Q1 of t_year+1 (adoption rule: registry year == t_year, or registration "
        "in Jan-Mar of the following year)",
        "data/scrape/registry/registry_match_final.csv + web_month_final.csv",
        "precedent: Guzman & Kacperczyk 2019 define founding via registration "
        "records; two-independent-signal auto-confirm only"),
    _v2("founding_month_imputed_v2", "int8 0/1", "1 = still January fallback under v2", "derived"),
    _v2("founding_v2_source", "string", "cb_day / cb_month / registry_<cc> (uk fr no ee lv de "
        "fi dk ch be es se nl it pl pt at ie lu; incl. the Orbis arm) / web / january "
        "(precedence-resolved)", "derived"),
    _v2("registry_id", "string", "Official register id of the auto-confirmed match "
        "(CH company number / SIREN / Orgnr / Y-tunnus / CVR / regcode); <NA> = unmatched; "
        "the permanent linkage key for re-runs and future joins", "registry_match_final.csv"),
    _v2("registry_status", "string", "Register status of the matched entity (dissolved/active/"
        "C(essée)/…), COLLECT-ONLY (scoped end-state exception, dating only)",
        "registry_match_final.csv"),
    _v2("registry_dissolution_date", "date", "Register-grade closure date (CH date_of_cessation; "
        "Sirene dateDebut of the C period; Brreg slettedato; …), COLLECT-ONLY; pilot QA: "
        "CH lag median 0.0 months vs cbcsv_closed_date", "registry_match_final.csv"),
    # ── Outcomes 36m under v2 founding month ────────────────────────────────
    _o2("fund36_private_v2", "Int8 0/1", "fund36_private recomputed on the v2 window", "derived"),
    _o2("grant36_cb_v2", "Int8 0/1", "grant36_cb recomputed on the v2 window", "derived"),
    _o2("sum_fund_usd_private_36_v2", "float64", "private USD sum on the v2 window", "derived"),
    _o2("logfund36_private_v2", "float64", "log1p of sum_fund_usd_private_36_v2", "derived"),
    _o2("success36_v2", "Int8 0/1", "success36 composite recomputed on the v2 window", "derived"),
    _o2("first_private_offset_m_v2", "Int32", "months from v2 founding month to first private "
        "round (uncensored horizon); input to the v2 Cox and window-robustness models", "derived"),
]


def expected_columns() -> list[str]:
    return [c.name for c in SCHEMA]


def validate(df: pd.DataFrame) -> None:
    """Assert the frame's columns EQUAL the schema (names and order)."""
    expected = expected_columns()
    got = list(df.columns)
    if got != expected:
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise AssertionError(
            f"schema violation, missing: {missing or 'none'}; extra: {extra or 'none'}; "
            f"(order also enforced)")


def render_markdown(df: pd.DataFrame | None = None) -> str:
    """Data dictionary (docs/variables.md) straight from the schema list."""
    n = f"{len(df):,}" if df is not None else "…"
    lines = [
        "# Variables: `data/processed/firms_analysis.parquet`",
        "",
        "**Auto-generated by `scripts/build_analysis.py` from `src/schema.py`, do not edit by hand.**",
        "",
        f"One row per analysis firm (cohorts 2016–2022, present in panel_fat; N = {n}). "
        "Row drops are documented in `outputs/tables/cleaning_funnel.csv`; the raw files keep "
        "all firms. Flags are int8 0/1. Window convention: a W-month window covers the founding "
        "month plus W−1 following months; year-precision firms start at January of the founding year.",
        "",
    ]
    group = None
    for c in SCHEMA:
        if c.group != group:
            group = c.group
            lines += [f"## {group}", "", "| Column | Dtype | Definition | Source | Cleaning rule | Non-null |",
                      "|---|---|---|---|---|---|"]
        nn = f"{int(df[c.name].notna().sum()):,}" if df is not None and c.name in df.columns else ""
        esc = lambda s: s.replace("|", "\\|")
        lines.append(f"| `{c.name}` | {c.dtype} | {esc(c.definition)} | {esc(c.source)} | "
                     f"{esc(c.cleaning)} | {nn} |")
    lines.append("")
    return "\n".join(lines)
