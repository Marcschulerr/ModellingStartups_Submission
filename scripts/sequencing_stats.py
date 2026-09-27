"""
Grant-first financing sequences (thesis Ch. 4.5): fixed-horizon conversion
rates instead of an un-censored "ever converted" share.

Definitions (frame = all 22,039 analysis firms):
  grant-first firm   first post-founding financing event is a PURE grant
                     month (funding_types == 'Grant'); founding = v1
                     month index (frame-wide defined), pre-founding events
                     excluded by the window definition
  conversion at h    any private-round month (funding_types with a non-Grant
                     type) in months 1..h after the grant-first month, among
                     firms with follow-up >= h from that month

The chapter cites the MONTH-VERIFIED variant (sample == 'mv': firms with
founding_month_imputed_v2 == 0, founding = the verified v2 month); the frame
variant (sample == 'frame', v1 founding) stays as the pinned robustness
layer and its grant-first shares must keep reproducing 10.7% (circular) vs
6.8% (non-circular).

Output: outputs/estimates/sequencing_stats.csv  (column `sample`)
        outputs/tables/prefounding_private.csv  (mv firms with a PRIVATE round
        before the verified founding month; feeds the "pre-founding rounds
        counted as month-0 events" robustness row in 3_time_to_event.R.
        With the register clock these are real pre-registration financings,
        not a Crunchbase artifact)
Run:    conda run -n ml python scripts/sequencing_stats.py
        (pipeline step 6, must run BEFORE analysis/3_time_to_event.R,
         which reads prefounding_private.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUTOFF_IDX = 2026 * 12 + 2  # month_index("2026-03"), single definition src/cohort.py

fa = pd.read_parquet(ROOT / "data" / "processed" / "firms_analysis.parquet",
                     columns=["firm_id", "circ", "founding_month_idx",
                              "founding_month_idx_v2", "founding_month_imputed_v2"])
pe = pd.read_parquet(ROOT / "data" / "processed" / "panel_events.parquet",
                     columns=["crunchbase_url", "month", "funding_event",
                              "funding_types"])
pe = pe[pe.funding_event == 1].copy()
mm = pd.to_datetime(pe["month"])
pe["midx"] = mm.dt.year * 12 + (mm.dt.month - 1)
pe["pure_grant"] = pe.funding_types == "Grant"
pe["private"] = pe.funding_types.notna() & (pe.funding_types != "Grant")

rows = []
for sample, base, fcol in (
        ("frame", fa, "founding_month_idx"),
        ("mv", fa[fa.founding_month_imputed_v2 == 0], "founding_month_idx_v2")):
    ev = pe.merge(base[["firm_id", "circ", fcol]],
                  left_on="crunchbase_url", right_on="firm_id")
    ev = ev[ev.midx >= ev[fcol]]  # post-founding events only

    # first post-founding event month per firm; grant-first = that month is a
    # pure grant month (no private event in it, by construction of pure_grant)
    first_m = ev.groupby("firm_id").midx.min().rename("first_midx")
    ev = ev.merge(first_m, on="firm_id")
    first_events = ev[ev.midx == ev.first_midx]
    gf = (first_events.groupby("firm_id")
          .agg(pure=("pure_grant", "all"), circ=("circ", "first"),
               gmidx=("midx", "first")))
    gf = gf[gf.pure]

    bi = base.set_index("firm_id")
    share_c = 100 * len(gf[gf.circ == 1]) / (bi.circ == 1).sum()
    share_n = 100 * len(gf[gf.circ == 0]) / (bi.circ == 0).sum()
    pin = " (pins 10.7 / 6.8)" if sample == "frame" else ""
    print(f"[{sample}] grant-first shares: circular {share_c:.1f}% | "
          f"non-circular {share_n:.1f}%{pin} | n = {len(gf)}")
    rows.append(dict(sample=sample, group="share_circ", horizon_m=np.nan,
                     n=int((bi.circ == 1).sum()), k=len(gf[gf.circ == 1]),
                     share=share_c / 100, ci_lo=np.nan, ci_hi=np.nan))
    rows.append(dict(sample=sample, group="share_noncirc", horizon_m=np.nan,
                     n=int((bi.circ == 0).sum()), k=len(gf[gf.circ == 0]),
                     share=share_n / 100, ci_lo=np.nan, ci_hi=np.nan))

    # first private month AFTER the grant-first month (any private month is
    # > gmidx by construction: grant-first firms have no earlier event and
    # the gmidx month itself is pure grant)
    priv = ev[ev.private].groupby("firm_id").midx.min().rename("first_priv_any")
    gf = gf.join(priv)
    gf["dp"] = gf.first_priv_any - gf.gmidx
    assert (gf.dp.dropna() >= 1).all()
    gf["fu"] = CUTOFF_IDX - gf.gmidx

    for label, d in (("all", gf), ("circular", gf[gf.circ == 1]),
                     ("non-circular", gf[gf.circ == 0])):
        for h in (24, 36, 48):
            dd = d[d.fu >= h]
            n = len(dd)
            k = int((dd.dp <= h).sum())
            p = k / n
            se = np.sqrt(p * (1 - p) / n)
            rows.append(dict(sample=sample, group=label, horizon_m=h, n=n, k=k,
                             share=p, ci_lo=p - 1.96 * se, ci_hi=p + 1.96 * se))
            print(f"  {label:>12} within {h}m: {100*p:.1f}% "
                  f"[{100*(p-1.96*se):.1f}, {100*(p+1.96*se):.1f}]  (k={k}, n={n})")

out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "estimates" / "sequencing_stats.csv", index=False)
print("wrote sequencing_stats.csv")

# pre-founding PRIVATE rounds on the v2 clock (mv firms): flag for the
# month-0 robustness variant
mvb = fa[fa.founding_month_imputed_v2 == 0]
evp = pe[pe.private].merge(mvb[["firm_id", "circ", "founding_month_idx_v2"]],
                           left_on="crunchbase_url", right_on="firm_id")
pre = evp[evp.midx < evp.founding_month_idx_v2]
flag = (pre.groupby("firm_id").size() > 0).rename("pre_private_any")
outp = mvb[["firm_id", "circ"]].merge(flag, on="firm_id", how="left")
outp["pre_private_any"] = outp.pre_private_any.fillna(False).astype(int)
outp[["firm_id", "pre_private_any"]].to_csv(
    ROOT / "outputs" / "tables" / "prefounding_private.csv", index=False)
sh = outp.groupby("circ").pre_private_any.mean()
print(f"pre-founding PRIVATE rounds (mv, v2 clock): {int(outp.pre_private_any.sum())} firms "
      f"({100*outp.pre_private_any.mean():.1f}%) | circ {100*sh[1]:.1f}% vs non-circ {100*sh[0]:.1f}%")
