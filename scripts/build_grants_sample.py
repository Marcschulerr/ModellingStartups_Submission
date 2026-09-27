"""
Matched sample for the grants RQ.

Design: matched comparison with pseudo-grant dates.
  - Treated: firms whose FIRST verified grant is SME-Instrument/EIC (main) or
    consortium-R&D (secondary), signed on/after founding, with a complete
    24-month outcome window (signature + 23 months ≤ 2026-03).
  - Controls: never-granted firms (has_grant == 0, has_grant_uncertain == 0)
    from the same founding cohort × circ × land whose pre-grant-funding status
    matches the treated firm's on BOTH margins: any private round ever before
    the grant month AND any private round in the 12 months right before it,
    evaluated at the control's own PSEUDO-grant month = its founding month +
    the treated firm's grant age, which aligns firm age AND calendar time.
    (The recent-12m margin exists because the first balance check failed on
    it: treated 38.4% vs control 28.4% recently funded, std diff 0.21, the
    classic apply-while-fundraising momentum pattern. Matching on it directly
    removes the imbalance instead of caveating it.)
    No same-land candidate -> land coarsened to all-Europe (rate reported).
  - Up to K=4 controls per treated firm, sampled without replacement within a
    set (controls may serve several sets; reuse reported). Seed pinned.

Outcomes per unit at the (pseudo-)grant month g: any/number of private-round
months and private USD in [g, g+23]; placebo window [g-12, g-1]; pre-grant
private round count (balance).

Writes: data/processed/grants_matched.csv, outputs/tables/grants_balance.csv.
Run (pipeline step 5; three variants, see run_pipeline.sh):
  conda run -n ml python scripts/build_grants_sample.py                # v1
  ... --fidx founding_month_idx_v2 --out grants_matched_v2.csv         # v2
  ... --fidx founding_month_idx_v2 --mv-only --out grants_matched_mv.csv
      (mv = the reported treatment definition)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import outcomes  # noqa: E402
from src.cohort import month_index, month_index_scalar  # noqa: E402

OUT_CSV = ROOT / "data" / "processed" / "grants_matched.csv"
BALANCE_CSV = ROOT / "outputs" / "tables" / "grants_balance.csv"
SEED = 20260711
K = 4
POST_M = 24                                   # outcome window: [g, g+23]
LAST_IDX = month_index_scalar(outcomes.PANEL_LAST_MONTH)

GROUPS = {"sme_eic": ("SME-Instrument", "EIC"), "consortium": ("consortium-R&D",)}
BALANCE_VARS = ["pre_n_private", "pre12_any_private", "n_founders", "n_industries",
                "founding_month_imputed", "grant_age_m"]


def private_event_arrays(ev: pd.DataFrame) -> dict[str, np.ndarray]:
    """firm -> sorted array of (midx, usd) for private-round months."""
    p = ev[ev["private_month"] == 1][["crunchbase_url", "midx", "funding_total_usd"]]
    p = p.sort_values(["crunchbase_url", "midx"])
    return {k: g[["midx", "funding_total_usd"]].to_numpy()
            for k, g in p.groupby("crunchbase_url", observed=True)}


def grant_month_arrays(ev: pd.DataFrame) -> dict[str, np.ndarray]:
    """firm -> sorted array of midx for CB grant-round months (runway-mechanism tests)."""
    p = ev[ev["grant_month"] == 1].sort_values(["crunchbase_url", "midx"])
    return {k: g["midx"].to_numpy() for k, g in p.groupby("crunchbase_url", observed=True)}


def ma_month_arrays(ev: pd.DataFrame) -> dict[str, np.ndarray]:
    """firm -> sorted array of midx for M&A event months (success composite)."""
    p = ev[ev["ma_event"] == 1].sort_values(["crunchbase_url", "midx"])
    return {k: g["midx"].to_numpy() for k, g in p.groupby("crunchbase_url", observed=True)}


def investor_arrays(ev: pd.DataFrame) -> dict[str, np.ndarray]:
    """firm -> (midx, n_investors) on PRIVATE round months (BA-bridge margin).

    Investor counts live in panel_lean (not extracted to the events parquet);
    joined here on firm x month and restricted to private months."""
    from src.data_loading import load_panel_lean
    from src.cohort import month_index
    lean = load_panel_lean(columns=["crunchbase_url", "month", "funding_event",
                                    "funding_n_investors"])
    lean = lean[lean["funding_event"] == 1].copy()
    lean["midx"] = month_index(lean["month"]).astype("int32")
    lean["ninv"] = pd.to_numeric(lean["funding_n_investors"], errors="coerce").fillna(0)
    priv = ev.loc[ev["private_month"] == 1, ["crunchbase_url", "midx"]]
    j = priv.merge(lean[["crunchbase_url", "midx", "ninv"]],
                   on=["crunchbase_url", "midx"], how="left")
    j["ninv"] = j["ninv"].fillna(0)
    j = j.sort_values(["crunchbase_url", "midx"])
    return {k: g[["midx", "ninv"]].to_numpy()
            for k, g in j.groupby("crunchbase_url", observed=True)}


def window_stats(arr: np.ndarray | None, garr: np.ndarray | None,
                 iarr: np.ndarray | None, maarr: np.ndarray | None, g: int) -> dict:
    """Private-round / grant-round / investor / M&A stats around (pseudo-)grant month g.

    Runway-mechanism extensions: early/late split of the 24m window,
    cumulative 36m/48m horizons (interpret only where the row's window is complete),
    grant-round counts (treadmill check). BA-bridge extensions: investor accumulation
    on private rounds, M&A and max monthly amount (success-composite ingredients)."""
    if arr is None or not len(arr):
        out = dict(pre_n_private=0, pre12_any_private=0, post24_any_private=0,
                   post24_n_private=0, post24_usd_private=0.0, post24_max_usd_private=0.0,
                   post12_any_private=0, post13_24_any_private=0,
                   post36_any_private=0, post48_any_private=0,
                   post_first_private_offset_m=np.nan)
    else:
        m, usd = arr[:, 0], arr[:, 1]
        post = (m >= g) & (m <= g + POST_M - 1)
        pm = m[m >= g]
        out = dict(
            pre_n_private=int((m < g).sum()),
            pre12_any_private=int(((m >= g - 12) & (m < g)).any()),
            post24_any_private=int(post.any()),
            post24_n_private=int(post.sum()),
            post24_usd_private=float(usd[post].sum()),
            post24_max_usd_private=float(usd[post].max()) if post.any() else 0.0,
            post12_any_private=int(((m >= g) & (m <= g + 11)).any()),
            post13_24_any_private=int(((m >= g + 12) & (m <= g + 23)).any()),
            post36_any_private=int(((m >= g) & (m <= g + 35)).any()),
            post48_any_private=int(((m >= g) & (m <= g + 47)).any()),
            # months from (pseudo-)grant month to FIRST private round at or after it,
            # uncensored horizon; NaN = none by panel end.
            # Same >= g convention as post24_any_private (offset <= 23 reproduces it).
            post_first_private_offset_m=float(pm.min() - g) if len(pm) else np.nan,
        )
    gm = garr if garr is not None else np.empty(0)
    out["post24_n_grant_months"] = int(((gm >= g) & (gm <= g + POST_M - 1)).sum())
    # excluding months g..g+3: the treated firm's own EU award is often recorded in CB
    # near signature (34% within ±3m), so this variant removes that mechanical component
    out["post24_n_grant_months_ex"] = int(((gm >= g + 4) & (gm <= g + POST_M - 1)).sum())
    if iarr is not None and len(iarr):
        im, inv = iarr[:, 0], iarr[:, 1]
        ipost = (im >= g) & (im <= g + POST_M - 1)
        out["post24_n_investors"] = float(inv[ipost].sum())
        out["post24_any_multi_inv"] = int((inv[ipost] >= 2).any())
    else:
        out["post24_n_investors"] = 0.0
        out["post24_any_multi_inv"] = 0
    mam = maarr if maarr is not None else np.empty(0)
    out["post24_ma"] = int(((mam >= g) & (mam <= g + POST_M - 1)).any())
    return out


def main() -> None:
    import argparse
    global OUT_CSV, BALANCE_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidx", default="founding_month_idx",
                    help="founding-month column for ages/pseudo-dates "
                         "(v2 variant: founding_month_idx_v2)")
    ap.add_argument("--out", default=None,
                    help="output CSV name under data/processed/ (v2 variant)")
    ap.add_argument("--mv-only", action="store_true", dest="mv_only",
                    help="restrict BOTH treated and control pool to firms with "
                         "a month-verified founding date (the January "
                         "imputation is measurement error in the time axis: "
                         "pseudo-award months of imputed controls sit "
                         "~5 months early and imputed treated shift their "
                         "whole set; one sample for both RQs)")
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "data" / "processed" / "firms_analysis.parquet")
    if args.fidx != "founding_month_idx":
        assert args.fidx in df.columns, args.fidx
        imp = args.fidx.replace("idx", "imputed")
        df["founding_month_idx"] = df[args.fidx].astype("int32")
        df["founding_month_imputed"] = df[imp].astype("int8")
        print(f"NOTE: matching under {args.fidx} (parallel v2 variant; "
              f"v1 sample and pins untouched)")
    if args.mv_only:
        n0 = len(df)
        df = df[df["founding_month_imputed"] == 0]
        print(f"NOTE: --mv-only, frame filtered to month-verified founding "
              f"dates: {len(df):,} of {n0:,} firms")
    if args.out:
        OUT_CSV = ROOT / "data" / "processed" / args.out
        sfx = Path(args.out).stem.replace("grants_matched", "") or "_v2"
        BALANCE_CSV = BALANCE_CSV.with_name(
            BALANCE_CSV.stem + sfx + BALANCE_CSV.suffix)
    ev = outcomes.load_panel_events()
    events = private_event_arrays(ev)
    gr_events = grant_month_arrays(ev)
    inv_events = investor_arrays(ev)
    ma_events = ma_month_arrays(ev)
    rng = np.random.default_rng(SEED)

    # descriptive context, printed for the narrative: investor breadth in the
    # first 36m from founding vs success36
    multi = pd.Series({fid: bool(len(a) and (a[(a[:, 0] >= f) & (a[:, 0] <= f + 35), 1] >= 2).any())
                       for fid, f, a in ((r.firm_id, r.founding_month_idx,
                                          inv_events.get(r.firm_id, np.empty((0, 2))))
                                         for r in df.itertuples())})
    s36 = df.set_index("firm_id")["success36"]
    print(f"BA-link check: P(success36) = {s36[multi.reindex(s36.index).fillna(False)].mean():.1%} "
          f"with a ≥2-investor private round in 36m vs "
          f"{s36[~multi.reindex(s36.index).fillna(False)].mean():.1%} without")

    df["g_midx"] = month_index(pd.to_datetime(df["first_grant_date"]))
    ctrl = df[(df["has_grant"] == 0) & (df["has_grant_uncertain"] == 0)]
    print(f"control pool: {len(ctrl):,} never-granted firms")
    cells = {key: sub for key, sub in ctrl.groupby(["t_year", "circ", "land"])}
    cells_nl = {key: sub for key, sub in ctrl.groupby(["t_year", "circ"])}

    rows, reuse = [], {}
    stats = {}
    for gname, schemes in GROUPS.items():
        tr = df[(df["has_grant"] == 1)
                & df["first_grant_scheme_group"].isin(schemes)].copy()
        post_ok = tr["g_midx"] >= tr["founding_month_idx"]
        win_ok = tr["g_midx"] + POST_M - 1 <= LAST_IDX
        usable = tr[post_ok & win_ok]
        n_unmatched = n_coarse = 0
        set_id = 0
        for _, t in usable.iterrows():
            age = int(t.g_midx - t.founding_month_idx)
            t_stats = window_stats(events.get(t.firm_id), gr_events.get(t.firm_id),
                                   inv_events.get(t.firm_id), ma_events.get(t.firm_id),
                                   int(t.g_midx))
            bucket = (t_stats["pre_n_private"] > 0, t_stats["pre12_any_private"] > 0)
            cand = cells.get((t.t_year, t.circ, t.land))
            picked, coarse = _pick(cand, age, bucket, events, rng)
            if picked is None:
                cand = cells_nl.get((t.t_year, t.circ))
                picked, _ = _pick(cand, age, bucket, events, rng)
                coarse = True
            if picked is None:
                n_unmatched += 1
                continue
            n_coarse += int(coarse)
            set_key = f"{gname}_{set_id}"
            set_id += 1
            scheme = str(t.first_grant_scheme)
            bf = int("-BF" in scheme)
            # grant size class (runway dose test): SME-1 / plain SME = ~EUR 50k lump
            # (no runway), everything else grant-only = EUR 0.5-2.5M, -BF separate.
            size = "blended" if bf else (
                "small" if scheme in ("SME-1", "SME") else "large")
            rows.append(_row(t, set_key, gname, 1, age, int(t.g_midx), bf,
                             scheme, size, coarse, t_stats))
            for _, c in picked.iterrows():
                g_c = int(c.founding_month_idx + age)
                reuse[c.firm_id] = reuse.get(c.firm_id, 0) + 1
                rows.append(_row(c, set_key, gname, 0, age, g_c, bf, scheme, size,
                                 coarse,
                                 window_stats(events.get(c.firm_id),
                                              gr_events.get(c.firm_id),
                                              inv_events.get(c.firm_id),
                                              ma_events.get(c.firm_id), g_c)))
        stats[gname] = dict(usable=len(usable), matched=set_id, unmatched=n_unmatched,
                            coarsened=n_coarse,
                            circ=int(usable.circ.sum()))

    m = pd.DataFrame(rows)
    m["log_post24_usd"] = np.log1p(m["post24_usd_private"])
    # horizon completeness per ROW (controls' pseudo-dates differ within a set;
    # fixest drops sets that lose all controls to the filter)
    m["win36_complete"] = (m["g_midx"] + 35 <= LAST_IDX).astype(int)
    m["win48_complete"] = (m["g_midx"] + 47 <= LAST_IDX).astype(int)
    # BA-compatible success composite on the post-grant window (≥3 private rounds,
    # a >$10M month, or M&A) + log investor accumulation
    m["post24_success"] = ((m["post24_n_private"] >= 3)
                           | (m["post24_max_usd_private"] > 10_000_000)
                           | (m["post24_ma"] == 1)).astype(int)
    m["log_post24_investors"] = np.log1p(m["post24_n_investors"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_CSV, index=False)

    print(f"\nmatched sample -> {OUT_CSV.name}  ({len(m):,} rows, seed {SEED})")
    for gname, s in stats.items():
        sets = m[(m.group == gname) & (m.treated == 1)]
        nctrl = m[(m.group == gname) & (m.treated == 0)]
        print(f"  {gname:10s} usable {s['usable']:>4} (circ {s['circ']}) | matched "
              f"{s['matched']:>4} | unmatched {s['unmatched']} | land-coarsened "
              f"{s['coarsened']} ({100 * s['coarsened'] / max(s['matched'], 1):.1f}%) | "
              f"controls {len(nctrl):,} (avg {len(nctrl) / max(s['matched'], 1):.2f}/set)")
    ru = pd.Series(reuse)
    print(f"  control reuse: {int((ru > 1).sum()):,} firms serve >1 set (max {int(ru.max())})")

    bal_rows = []
    for gname in GROUPS:
        g = m[m.group == gname]
        for v in BALANCE_VARS:
            a = pd.to_numeric(g.loc[g.treated == 1, v], errors="coerce")
            b = pd.to_numeric(g.loc[g.treated == 0, v], errors="coerce")
            sd = np.sqrt((a.var() + b.var()) / 2)
            bal_rows.append({"group": gname, "variable": v,
                             "mean_treated": round(a.mean(), 4),
                             "mean_control": round(b.mean(), 4),
                             "std_diff": round(float((a.mean() - b.mean()) / sd), 4)
                             if sd > 0 else 0.0})
    bal = pd.DataFrame(bal_rows)
    bal.to_csv(BALANCE_CSV, index=False)
    print(f"\nbalance -> {BALANCE_CSV.name}")
    print(bal.to_string(index=False))


def _pick(cand: pd.DataFrame | None, age: int, bucket: tuple[bool, bool],
          events: dict, rng: np.random.Generator):
    """Eligible controls in a candidate cell; sample up to K. Returns (df|None, coarse=False)."""
    if cand is None or not len(cand):
        return None, False
    g_c = cand["founding_month_idx"].to_numpy() + age
    ok_win = g_c + POST_M - 1 <= LAST_IDX
    ids = cand["firm_id"].to_numpy()
    elig = []
    for i in np.flatnonzero(ok_win):
        arr = events.get(ids[i])
        if arr is None or not len(arr):
            pre_any_c, pre12_c = False, False
        else:
            m = arr[:, 0]
            pre_any_c = bool((m < g_c[i]).any())
            pre12_c = bool(((m >= g_c[i] - 12) & (m < g_c[i])).any())
        if (pre_any_c, pre12_c) == bucket:
            elig.append(i)
    if not elig:
        return None, False
    take = rng.choice(len(elig), size=min(K, len(elig)), replace=False)
    return cand.iloc[[elig[i] for i in take]], False


def _row(f, set_key, gname, treated, age, g_midx, bf, scheme, size, coarse, w) -> dict:
    return dict(set_id=set_key, group=gname, treated=treated, firm_id=f.firm_id,
                land=f.land, t_year=int(f.t_year), circ=int(f.circ),
                circ_subtype=f.circ_subtype, grant_age_m=age, g_midx=g_midx,
                bf_set=bf, set_scheme=scheme, size_class=size,
                land_coarsened=int(coarse),
                n_founders=f.n_founders, n_industries=f.n_industries,
                founding_month_imputed=int(f.founding_month_imputed), **w)


if __name__ == "__main__":
    main()
