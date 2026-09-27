"""
Outcome aggregation from the monthly panel over the 36m post-founding window.

Source: panel_fat.csv, whose `funding_types` labels every funding-event month
with round types (incl. 'Grant'), so private rounds and grants are separable,
which panel_lean could not do. The 31 GB file is never loaded whole:
`extract_panel_events()` streams it ONCE into
data/processed/panel_events.parquet (~6 event columns, all 2,063,279 rows,
incl. pre-founding months), and everything downstream reads that parquet.
Re-extract with force=True whenever panel_fat.csv changes.

Window definition (the off-by-one, stated once): a W-month window covers month
offsets [0, W-1] relative to the founding month. Year-precision firms have
founding_month_idx imputed to January (cohort.py), so their 36m window is
Jan(Y)..Dec(Y+2). A firm founded 2022-12 has a window ending 2025-11 (asserted).

Private/grant rules:
  - a month is PRIVATE if ANY of its round types != 'Grant'
  - a month is GRANT  if ANY of its round types == 'Grant' (mixed months are both)
  - private amount sums exclude pure-grant months only; mixed months count fully
    as private (monthly totals are not separable by round), and the mixed
    in-window month count is reported at build.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import month_index, month_index_scalar

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PARQUET = ROOT / "data" / "processed" / "panel_events.parquet"

MAIN_WINDOW = 36
SUCCESS_MIN_EVENTS = 3                   # success := >=3 event-months ...
SUCCESS_MIN_MAX_MONTH_USD = 10_000_000   # ... or a month with > $10M ...
                                         # ... or an M&A event in the window
PANEL_LAST_MONTH = "2026-03-01"
EXPECTED_FAT_ROWS = 2_063_279            # pinned by the streaming audit of the raw export
EXPECTED_FAT_FIRMS = 25_297              # 124 firms of the firm table have no panel_fat rows

# Pin the inclusive-window arithmetic with a fixed example.
assert month_index_scalar("2022-12-01") + 35 == month_index_scalar("2025-11-01")


def extract_panel_events(force: bool = False) -> None:
    """One streaming pass over panel_fat.csv -> panel_events.parquet."""
    if EVENTS_PARQUET.exists() and not force:
        return
    import pyarrow as pa
    from .data_loading import open_panel_fat

    cols = ["crunchbase_url", "month", "funding_event", "funding_total_usd",
            "funding_types", "ma_event"]
    reader = open_panel_fat(columns=cols, column_types={c: pa.string() for c in cols})
    parts = []
    for batch in reader:
        d = batch.to_pandas()
        parts.append(pd.DataFrame({
            "crunchbase_url": d["crunchbase_url"],
            "month": pd.to_datetime(d["month"]),
            "funding_event": pd.to_numeric(d["funding_event"]).astype("int8"),
            "funding_total_usd": pd.to_numeric(d["funding_total_usd"]).fillna(0.0).astype("float64"),
            # pyarrow keeps literal 'NA' in string columns unless told otherwise
            "funding_types": d["funding_types"].replace({"NA": None, "": None}),
            "ma_event": pd.to_numeric(d["ma_event"]).astype("int8"),
        }))
    ev = pd.concat(parts, ignore_index=True)
    assert len(ev) == EXPECTED_FAT_ROWS, f"panel_fat rows {len(ev):,} != {EXPECTED_FAT_ROWS:,}"
    EVENTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVENTS_PARQUET, index=False)
    print(f"      extracted {len(ev):,} panel rows -> {EVENTS_PARQUET.name}")


def load_panel_events() -> pd.DataFrame:
    """Event slice with month index and private/grant month flags."""
    ev = pd.read_parquet(EVENTS_PARQUET)
    assert len(ev) == EXPECTED_FAT_ROWS, f"panel rows {len(ev):,} != {EXPECTED_FAT_ROWS:,}"
    n_firms = ev["crunchbase_url"].nunique()
    assert n_firms == EXPECTED_FAT_FIRMS, f"panel firms {n_firms:,} != {EXPECTED_FAT_FIRMS:,}"
    ev["midx"] = month_index(ev["month"]).astype("int32")
    assert ev["midx"].max() == month_index_scalar(PANEL_LAST_MONTH)

    is_event = ev["funding_event"] == 1
    types = ev["funding_types"]
    assert bool(types.notna().eq(is_event).all()), \
        "funding_types must be filled exactly on event months"
    toks = types[is_event].map(lambda s: {t.strip() for t in str(s).split(";")})
    ev["private_month"] = np.zeros(len(ev), dtype="int8")
    ev.loc[is_event, "private_month"] = toks.map(
        lambda s: any(t != "Grant" for t in s)).astype("int8")
    ev["grant_month"] = np.zeros(len(ev), dtype="int8")
    ev.loc[is_event, "grant_month"] = toks.map(lambda s: "Grant" in s).astype("int8")
    return ev


def aggregate_panel_windows(
    firms: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    fidx_col: str = "founding_month_idx",
) -> pd.DataFrame:
    """Per-firm 36m window aggregates, indexed by firm_id.

    `fidx_col` selects the founding-month convention; the sanity script uses a
    Jan-of-founding-year column to replicate the original anchor convention.
    Firms without panel rows are simply absent from the result (the build drops
    them, documented in the funnel).
    """
    if panel is None:
        panel = load_panel_events()

    fmap = firms.set_index("firm_id")[fidx_col].astype("int32")
    key = panel["crunchbase_url"]
    f = key.map(fmap)
    keep = f.notna()          # panel rows of firms outside `firms` are ignored
    panel, key, f = panel[keep], key[keep], f[keep]
    offset = (panel["midx"].astype("int32") - f.astype("int32")).astype("int32")

    d = pd.DataFrame({
        "key": key,
        "event": panel["funding_event"].astype("int16"),
        "private": panel["private_month"].astype("int16"),
        "grant": panel["grant_month"].astype("int16"),
        "usd": panel["funding_total_usd"],
        "ma": panel["ma_event"].astype("int16"),
        "offset": offset,
    })
    d["usd_private"] = d["usd"].where(d["private"] == 1, 0.0)
    d["mixed"] = ((d["private"] == 1) & (d["grant"] == 1)).astype("int16")

    pe = d[(d["private"] == 1) & (d["offset"] >= 0)]
    me = d[(d["ma"] >= 1) & (d["offset"] >= 0)]
    diag = pd.DataFrame({
        "first_private_offset_m": pe.groupby("key", observed=True)["offset"].min().astype("Int32"),
        # dated M&A, any post-founding horizon: the timing complement to the
        # undated exit_any snapshot (src/survival.py)
        "first_ma_offset_m": me.groupby("key", observed=True)["offset"].min().astype("Int32"),
    })

    w = MAIN_WINDOW
    win = d[(d["offset"] >= 0) & (d["offset"] <= w - 1)]
    gw = win.groupby("key", observed=True)
    agg = pd.concat([pd.DataFrame({
        f"n_fund_events_{w}": gw["event"].sum().astype("int16"),
        f"n_private_months_{w}": gw["private"].sum().astype("int16"),
        f"n_grant_months_{w}": gw["grant"].sum().astype("int16"),
        f"n_mixed_months_{w}": gw["mixed"].sum().astype("int16"),
        f"sum_fund_usd_{w}": gw["usd"].sum(),
        f"sum_fund_usd_private_{w}": gw["usd_private"].sum(),
        f"max_month_fund_usd_{w}": gw["usd"].max(),
        f"n_ma_{w}": gw["ma"].sum().astype("int16"),
        f"n_panel_months_{w}": gw.size().astype("int16"),
    }), diag], axis=1)
    agg.index.name = "firm_id"
    return agg


def attach_outcomes(firms: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Join aggregates and derive the schema outcome columns.

    Firms absent from panel_fat keep <NA> aggregates here; the build drops them
    (funnel-documented) before schema validation. Internal ingredient columns
    (max month USD, M&A count, month counters) are removed after deriving
    success36: the schema stores outcomes, not ingredients.
    """
    df = firms.merge(agg, left_on="firm_id", right_index=True, how="left", validate="1:1")
    df["in_panel"] = df[f"n_panel_months_{MAIN_WINDOW}"].notna()

    w = MAIN_WINDOW
    present = df["in_panel"]
    n_mixed = int(df.loc[present, f"n_mixed_months_{w}"].sum())
    if n_mixed:
        print(f"      note: {n_mixed:,} in-window months carry BOTH a private round and a "
              f"grant, counted as private in amount sums (rule in src/outcomes.py)")

    n = df[f"n_fund_events_{w}"]
    df[f"fund{w}"] = (n >= 1).astype("Int8")
    df[f"fund{w}_private"] = (df[f"n_private_months_{w}"] >= 1).astype("Int8")
    df[f"grant{w}_cb"] = (df[f"n_grant_months_{w}"] >= 1).astype("Int8")
    df[f"logfund{w}"] = np.log1p(df[f"sum_fund_usd_{w}"])
    df[f"logfund{w}_private"] = np.log1p(df[f"sum_fund_usd_private_{w}"])
    # components kept as columns: the success flip is a core finding and its
    # decomposition needs the margins separable in R
    df[f"success{w}_bigmonth"] = (df[f"max_month_fund_usd_{w}"]
                                  > SUCCESS_MIN_MAX_MONTH_USD).astype("Int8")
    df[f"success{w}_ma"] = (df[f"n_ma_{w}"] >= 1).astype("Int8")
    df[f"success{w}"] = (
        (n >= SUCCESS_MIN_EVENTS)
        | (df[f"success{w}_bigmonth"] == 1)
        | (df[f"success{w}_ma"] == 1)
    ).astype("Int8")

    # n_private_months kept as the private analog of n_fund_events_36, needed
    # to test whether the success flip survives on grant-free round counts
    df[f"n_private_months_{w}"] = df[f"n_private_months_{w}"].astype("Int16")
    # dated M&A flag: full horizon, not window-bound
    df["ma_any_dated"] = df["first_ma_offset_m"].notna().astype("int8")
    df = df.drop(columns=[f"n_grant_months_{w}", f"n_mixed_months_{w}",
                          f"max_month_fund_usd_{w}", f"n_ma_{w}", f"n_panel_months_{w}"])
    return df
