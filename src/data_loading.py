"""
Data loading utilities for the Circular Economy Startup project.

Raw data lives in data/raw/ and should never be modified.
Use the load_* functions to get clean pandas DataFrames.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _csv(name: str, **kwargs) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    return pd.read_csv(path, low_memory=False, **kwargs)


def open_panel_fat(columns: list[str] | None = None, column_types: dict | None = None,
                   block_size: int = 1 << 26):
    """Streaming pyarrow reader over data/raw/panel_fat.csv (31 GB, never load it whole).

    Returns a pyarrow CSVStreamingReader; iterate its RecordBatches. `columns`
    restricts what is materialized, `column_types` overrides inference (e.g. force
    strings). Multiline quoted text fields require newlines_in_values=True, which
    makes parsing single-threaded, so restrict `columns` and expect minutes, not
    seconds, for a full pass.
    """
    import pyarrow.csv as pv
    return pv.open_csv(
        RAW_DIR / "panel_fat.csv",
        read_options=pv.ReadOptions(block_size=block_size),
        parse_options=pv.ParseOptions(newlines_in_values=True),
        convert_options=pv.ConvertOptions(include_columns=columns, column_types=column_types),
    )


# ── Public loaders ────────────────────────────────────────────────────────────

def load_companies_lean() -> pd.DataFrame:
    """Cross-section, 25 421 companies × 34 columns. Fast to load."""
    df = _csv("companies_lean")
    df["cb_founded_year"] = pd.to_numeric(df["cb_founded_year"], errors="coerce")
    return df


# Columns of companies_fat that are safe at-founding / event-date information.
# Everything else in the fat export is either an end-state snapshot (leaky for
# the causal design) or effectively empty: never load it.
FAT_WHITELIST = [
    "crunchbase_url",
    "cbcsv_founded_date", "cbcsv_founded_date_precision",
    "cbcsv_number_of_founders",
    "cbcsv_industries",
]

# End-state columns allowed ONLY as survival/exit OUTCOMES, never as covariates
# or outcomes of the funding models. Loaded exclusively by src/survival.py; the
# funding pipeline keeps using FAT_WHITELIST.
SURVIVAL_COLS = [
    "crunchbase_url",
    "cbcsv_acquisition_status",   # 'Was Acquired' (⊂ exit_any)
    "cbcsv_ipo_status",           # Public/Delisted (⊂ exit_any)
    "jina_scrape_status",         # website liveness at scrape 2026-03-27 (validation)
]


def load_companies_fat_selective(usecols: list[str] | None = None) -> pd.DataFrame:
    """Selective fat loader (382 MB file, never load all 116 columns).

    The multiline quoted text fields parse correctly with the C engine.
    """
    cols = list(usecols) if usecols is not None else list(FAT_WHITELIST)
    return _csv("companies_fat", usecols=cols, engine="c")


def load_panel_lean(columns: list[str] | None = None) -> pd.DataFrame:
    """Monthly panel, ~2 M rows × 26 columns. Optimised dtypes for speed/memory.

    `columns` restricts the read to a subset (pd.read_csv usecols); the dtype
    map is filtered accordingly. Default None loads all columns (old behaviour).
    """
    # Categorical dtypes for low-cardinality string columns (25k companies × 82 months)
    cats = [
        "crunchbase_url", "cb_company_name", "cb_categories", "cb_operating_status",
        "cbm_type", "cbcsv_headquarters_location", "cbcsv_headquarters_regions",
        "cbcsv_number_of_employees", "cbcsv_estimated_revenue_range",
        "cbcsv_industry_groups", "funding_stages", "cbcsv_funding_status",
        "cbcsv_ipo_status",
    ]
    dtypes = {c: "category" for c in cats}
    dtypes.update({
        "cb_founded_year": "Int16",
        "funding_event": "int8",
        "ma_event": "int8",
        "funding_n_rounds": "Int16",
        "cbcsv_number_of_funding_rounds": "Int16",
        "funding_n_investors": "float32",
        "cbcsv_number_of_investors": "float32",
        "cbcsv_number_of_acquisitions": "float32",
        "cbm_confidence": "float32",
    })
    if columns is not None:
        dtypes = {c: dt for c, dt in dtypes.items() if c in columns}
    df = _csv("panel_lean", dtype=dtypes, usecols=columns)
    if columns is None or "month" in columns:
        df["month"] = pd.to_datetime(df["month"])
    return df
