"""
Pre-dynamics around the (pseudo-)grant month for the matched Grants sample.

Motivation: the matching balances pre-grant
funding on two margins (any private round ever, any in the 12 months before the
grant). If treated firms were on a diverging trajectory right before applying
(Ashenfelter dip / apply-when-weak selection), the quarterly funding rates of
treated vs matched controls would show it. This script computes
P(>=1 private-round month) per quarter relative to the (pseudo-)grant month,
q = -8..+8, separately for sme_eic grant-only, sme_eic blended-finance and
consortium sets, plus the within-set treated-minus-control difference with a
set-level SE (the descriptive analogue of the set-FE estimator).

Reading guide: quarters -4..-1 lie inside the matched 12m window (the ANY margin
there is balanced by construction; the quarterly SHAPE is not), quarters -8..-5
are fully outside the matching -> the honest pre-trend test. Quarters that start
before a unit's founding are dropped; grant age is identical within a set by
construction, so sets drop symmetrically and treated/control stay comparable.

Writes: outputs/estimates/grants_predynamics.csv
        outputs/tables/grants_set_sizes.csv   (matching transparency: controls/set)
Run (pipeline step 6):
      conda run -n ml python scripts/grants_predynamics.py
      conda run -n ml python scripts/grants_predynamics.py --v2
        (v2 sets + eligibility filter, so the pre-trend diagnostic runs on
         the SAME sample as the headline ATT ->
         grants_predynamics_v2.csv / grants_set_sizes_v2.csv)
      conda run -n ml python scripts/grants_predynamics.py --mv
        (mv sets, the source of the fig3 pre-dynamics panel)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import outcomes  # noqa: E402

# --mv: month-verified-only sets (grants_matched_mv.csv) with the same
# eligibility filter as --v2, so the diagnostic runs on the headline sample.
MV = "--mv" in sys.argv
V2 = ("--v2" in sys.argv) or MV
_sfx = "_mv" if MV else ("_v2" if V2 else "")
OUT_CSV = ROOT / "outputs" / "estimates" / f"grants_predynamics{_sfx}.csv"
SIZES_CSV = ROOT / "outputs" / "tables" / f"grants_set_sizes{_sfx}.csv"
Q_MIN, Q_MAX = -8, 8


def main() -> None:
    m = pd.read_csv(ROOT / "data" / "processed" / f"grants_matched{_sfx}.csv")
    if V2:
        # eligibility filter: drop sets whose treated firm sits in a
        # non-eligible land x period (UK/CH with grant year >= 2021)
        tl = m[m["treated"] == 1][["set_id", "land", "g_midx"]]
        bad = set(tl.loc[tl["land"].isin(["UK", "CH"])
                         & (tl["g_midx"] // 12 >= 2021), "set_id"])
        m = m[~m["set_id"].isin(bad)]
        print(f"v2 mode: E4 dropped {len(bad)} sets -> "
              f"{m['set_id'].nunique()} sets")
    ev = outcomes.load_panel_events()
    p = ev[ev["private_month"] == 1]
    months = {k: g["midx"].to_numpy()
              for k, g in p.groupby("crunchbase_url", observed=True)}

    # -- matching transparency: distribution of controls per set
    sizes = (m[m["treated"] == 0].groupby(["group", "set_id"]).size()
             .rename("n_controls").reset_index())
    dist = (sizes.groupby(["group", "n_controls"]).size()
            .rename("n_sets").reset_index())
    dist["share"] = (dist["n_sets"]
                     / dist.groupby("group")["n_sets"].transform("sum")).round(4)
    SIZES_CSV.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(SIZES_CSV, index=False)
    print(f"set sizes -> {SIZES_CSV.name}")
    print(dist.to_string(index=False))

    # -- per unit x quarter: any private-round month in [g+3q, g+3q+2]
    split = np.where(m["group"] == "consortium", "consortium",
                     np.where(m["bf_set"] == 1, "sme_eic_bf", "sme_eic_grant_only"))
    m = m.assign(split=split)
    recs = []
    for r in m.itertuples():
        arr = months.get(r.firm_id)
        g, age = int(r.g_midx), int(r.grant_age_m)
        for q in range(Q_MIN, Q_MAX + 1):
            if age + 3 * q < 0:  # quarter starts before founding
                continue
            lo = g + 3 * q
            any_p = int(((arr >= lo) & (arr <= lo + 2)).any()) if arr is not None else 0
            recs.append((r.split, r.set_id, r.treated, q, any_p))
    d = pd.DataFrame(recs, columns=["split", "set_id", "treated", "q", "any_private"])

    # -- aggregate: raw rates per arm + within-set difference (SE over sets)
    rates = (d.groupby(["split", "q", "treated"])["any_private"].mean()
             .unstack("treated").rename(columns={0: "rate_control", 1: "rate_treated"}))
    per_set = (d.groupby(["split", "q", "set_id", "treated"])["any_private"].mean()
               .unstack("treated"))
    per_set["diff"] = per_set[1] - per_set[0]
    agg = per_set.groupby(["split", "q"])["diff"].agg(["mean", "std", "count"])
    out = rates.join(agg).reset_index().rename(
        columns={"mean": "diff_withinset", "count": "n_sets"})
    out["se_set"] = out["std"] / np.sqrt(out["n_sets"])
    out = out.drop(columns=["std"]).round(6)
    out.to_csv(OUT_CSV, index=False)
    print(f"\npre/post dynamics -> {OUT_CSV.name}")

    for s in out["split"].unique():
        sub = out[(out["split"] == s) & (out["q"] < 0)]
        worst = sub.iloc[(sub["diff_withinset"].abs()).argmax()]
        print(f"  {s:20s} pre-quarters: max |within-set diff| = "
              f"{worst['diff_withinset']*100:+.2f}pp (q={int(worst['q'])}, "
              f"se {worst['se_set']*100:.2f}pp, {int(worst['n_sets'])} sets)")
        far = sub[sub["q"] <= -5]
        print(f"  {'':20s} out-of-matching (q<=-5) mean diff = "
              f"{far['diff_withinset'].mean()*100:+.2f}pp")


if __name__ == "__main__":
    main()
