"""
The four main-text thesis figures, everything on the month-verified samples.

Style: made for a Times New Roman 12pt thesis; serif fonts, NO in-figure
titles beyond small panel labels (captions live in the Word document), only
axis labels, ticks and direct curve labels. Muted gray/blue palette with
RdBu accents.

Main text (files named by chapter figure number):
  fig1_financing_states     share of firms by financing state (no recorded
                            financing / grant only / private only / both) 0-60 since
                            founding, circ vs non-circ; observed counts;
                            month-verified sample; grants from panel_events
  fig2_time_to_private      KM cumulative probability of a first private
                            round (1-S(t)), circ vs non-circ; light reference
                            lines at 24/36/48m; at-risk strip
  fig3_award_panel          A: treated-minus-control quarterly funding rates,
                            quarters -8..-1 (95% CIs; predynamics_mv); B: KM
                            cumulative probability of a subsequent private
                            round, treated vs controls; at-risk strip
  fig4_bridge_km            post-award KM by pre-award private-financing
                            status (no-prior / prior sets; grants_km_bridge)
All appendix exhibits are tables, so no appendix figures are rendered here.

Inputs:  outputs/estimates/*.csv (steps 5b/6) + data/processed parquets.
Outputs: outputs/figures/fig{1..4}_*.png (300 dpi, for the Word document)
All in-figure text is ink-colored (no gray type); only the curves keep the
gray/blue palette.

Run:  conda run -n ml python scripts/render_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EST = ROOT / "outputs" / "estimates"
FIG = ROOT / "outputs" / "figures"
PROC = ROOT / "data" / "processed"

BLUE, GRAY, RED = "#2166ac", "#808080", "#b2182b"
DARKBLUE, LIGHTGRAY = "#053061", "#d9d9d9"  # RdBu ends for the states figure
INK = "#1a1a1a"
REF = "#c8c8c8"  # light reference lines
CUTOFF_IDX = 2026 * 12 + 2  # month_index("2026-03"), single definition src/cohort.py

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

RISK_TS = [0, 12, 24, 36, 48, 60]


def km_curve(ax, d, color, label, tmax, dy=0):
    """Cumulative-probability step curve (1-S) with CI band and end label."""
    d = d[d.time_m <= tmax].sort_values("time_m")
    x = [0] + list(d.time_m)
    y = [0] + list(1 - d.surv)
    ax.step(x, y, where="post", color=color, linewidth=1.4, zorder=3)
    ax.fill_between(list(d.time_m), 1 - d.upper, 1 - d.lower, step="post",
                    color=color, alpha=0.15, linewidth=0, zorder=2)
    if label:
        ax.annotate(label, (x[-1], y[-1]), xytext=(5, dy),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK)
    return y[-1]


def base_axis(ax, tmax, xlab):
    ax.set_xlim(0, tmax * 1.01)
    ax.set_xticks(range(0, tmax + 1, 12))
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel(xlab)
    ax.grid(False)


def light_refs(ax, ts, ymax=0.9):
    """Thin light reference lines at selected horizons (no labels)."""
    for t in ts:
        ax.plot([t, t], [0, ymax], color=REF, linewidth=0.6, zorder=1)


def at_risk(km_d, t):
    """Number at risk at month t (n_risk of the first event time >= t)."""
    after = km_d[km_d.time_m >= t]
    return int(after.n_risk.iloc[0]) if len(after) else int(km_d.n_risk.iloc[-1])


def risk_strip(ax, groups, ts, y0=-46, dy=9, row_labels=None):
    """At-risk rows beneath the x-axis; all text in ink (no gray type).
    groups = [(km_df, _), ...]; row_labels names each row at the left."""
    for i, (g, _) in enumerate(groups):
        if row_labels:
            ax.annotate(row_labels[i], (ts[0], 0), xytext=(-26, y0 - dy * i),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7, color=INK, annotation_clip=False)
        for t in ts:
            n = at_risk(g, t) if t > 0 else int(g.n_risk.iloc[0])
            ax.annotate(f"{n:,}", (t, 0), xytext=(0, y0 - dy * i),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=INK, annotation_clip=False)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


# ── f7 [Fig 2]: KM first private round, circular vs non-circular ───────────
# main sample = month-verified founding dates; round 4: light reference lines
# at 24/36/48 instead of gap connectors (values live in text/tables)
km7 = pd.read_csv(EST / "timing_km_private.csv")
d7 = km7[km7["sample"] == "main"]
fig, ax = plt.subplots(figsize=(6.3, 3.6))
light_refs(ax, (24, 36, 48))
for grp, color, lab, dy in (("circ=0", GRAY, "non-circular", 5),
                            ("circ=1", BLUE, "circular", -5)):
    km_curve(ax, d7[d7.group == grp], color, lab, tmax=60, dy=dy)
base_axis(ax, 60, "Months since founding")
ax.set_ylabel("Share with ≥1 private round")
risk_strip(ax, [(d7[d7.group == "circ=0"], GRAY), (d7[d7.group == "circ=1"], BLUE)],
           RISK_TS, row_labels=["non-circular", "circular"])
save(fig, "fig2_time_to_private")

# ── f12 [Fig 3]: pre-award rates + post-award KM (mv-only sets) ──────────────
pre = pd.read_csv(EST / "grants_predynamics_mv.csv")
pre = pre[(pre.split == "sme_eic_grant_only") & (pre.q < 0)].sort_values("q")
km6 = pd.read_csv(EST / "grants_km_postgrant.csv")
km6["time_m"] = km6["time_m"] - 1
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 3.3),
                               gridspec_kw={"width_ratios": [1, 1.35]})
fig.subplots_adjust(wspace=0.35)
axA.axhline(0, color="#4d4d4d", linewidth=0.8, zorder=1)
axA.axvline(0, color=REF, linewidth=0.8, zorder=1)
axA.annotate("award", (0, 0.085), xytext=(-4, 0), textcoords="offset points",
             ha="right", va="center", fontsize=7.5, color=INK)
axA.errorbar(pre.q, pre.diff_withinset, yerr=1.96 * pre.se_set, fmt="o",
             color=BLUE, markersize=3.5, linewidth=1.1, capsize=2, zorder=3)
axA.set_xlim(-8.6, 0.6)
axA.set_xticks(range(-8, 1, 2))
axA.set_ylim(-0.1, 0.1)
axA.set_yticks([-.08, -.04, 0, .04, .08])
axA.set_yticklabels(["−8 pp", "−4 pp", "0", "+4 pp", "+8 pp"])
axA.set_xlabel("Quarters relative to the award")
axA.set_ylabel("Difference in private-round share (pp)")
axA.set_title("A  Before the award", fontsize=9, loc="left", pad=4)
axA.grid(False)
for grp, color, lab, dy in (("treated=0", GRAY, "matched\ncontrols", 8),
                            ("treated=1", BLUE, "grant\nrecipients", -8)):
    km_curve(axB, km6[km6.group == grp], color, lab, tmax=48, dy=dy)
base_axis(axB, 48, "Months since the award")
axB.set_ylabel("Share with ≥1 subsequent private round")
axB.set_title("B  After the award", fontsize=9, loc="left", pad=4)
risk_strip(axB, [(km6[km6.group == "treated=0"], GRAY),
                 (km6[km6.group == "treated=1"], BLUE)], [0, 12, 24, 36, 48],
           row_labels=["controls", "recipients"])
save(fig, "fig3_award_panel")

# ── f13 [Fig 4]: post-award KM by pre-award private-financing status ─────────
kb = pd.read_csv(EST / "grants_km_bridge.csv")
kb["time_m"] = kb["time_m"] - 1
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True)
fig.subplots_adjust(wspace=0.12)
LBL4 = {"noprior": [("matched controls", 5, 0.63), ("grant recipients", 25, 0.22)],
        "prior": [("matched controls", 25, 0.40), ("grant recipients", 11, 0.77)]}
for ax, sp, plab in ((axes[0], "noprior", "A  No private round before the award"),
                     (axes[1], "prior", "B  Private round before the award")):
    d = kb[kb.split == sp]
    for grp, color in (("treated=0", GRAY), ("treated=1", BLUE)):
        km_curve(ax, d[d.group == grp], color, None, tmax=48)
    for lab, lx, ly in LBL4[sp]:
        ax.annotate(lab, (lx, ly), ha="left", va="center", fontsize=8.5,
                    color=INK)
    ax.set_title(plab, fontsize=9, loc="left", pad=4)
    base_axis(ax, 48, "Months since the award")
    risk_strip(ax, [(d[d.group == "treated=0"], GRAY),
                    (d[d.group == "treated=1"], BLUE)], [0, 12, 24, 36, 48],
               y0=-46, row_labels=(["controls", "recipients"] if sp == "noprior"
                                   else None))
axes[0].set_ylabel("Share with ≥1 subsequent private round")
axes[1].set_ylabel("")
save(fig, "fig4_bridge_km")

# ── shared firm-level slice for Fig 1 ────────────────────────────────────────
fa = pd.read_parquet(PROC / "firms_analysis.parquet",
                     columns=["firm_id", "circ", "founding_month_idx",
                              "founding_month_idx_v2",
                              "founding_month_imputed_v2",
                              "first_private_offset_m_v2",
                              "fund36_private_v2", "grant36_cb_v2"])
fa["cohort"] = fa["founding_month_idx"] // 12  # CB founding year (month_index def)
r = fa.groupby("circ")[["fund36_private_v2", "grant36_cb_v2"]].mean()
print("  rate check: private@36m circ {:.1f} vs non-circ {:.1f} (T1: 66.5/71.9); "
      "grant@36m {:.1f} vs {:.1f} (T1: 14.6/9.1)".format(
          100 * r.loc[1, "fund36_private_v2"], 100 * r.loc[0, "fund36_private_v2"],
          100 * r.loc[1, "grant36_cb_v2"], 100 * r.loc[0, "grant36_cb_v2"]))

# ── Fig 1: financing states by month since founding (0-60) ─────────────────
# Share of firms in state {no financing yet / grant only / private only /
# both} at each month 0..60; month-verified sample (clean time axis, as f7).
# Private = first_private_offset_m_v2 (authoritative df column); grant months
# on the fly from panel_events (funding_types contains 'Grant', pre-founding
# months excluded). Denominator at m = firms with follow-up >= m (at-risk
# counts printed beneath the axis); no dataframe column is created.
dm = fa[fa.founding_month_imputed_v2 == 0].copy()
pe = pd.read_parquet(PROC / "panel_events.parquet",
                     columns=["crunchbase_url", "month", "funding_event",
                              "funding_types"])
pe = pe[pe.funding_event == 1]
mm = pd.to_datetime(pe["month"])
pe["midx"] = mm.dt.year * 12 + (mm.dt.month - 1)
pe["is_grant"] = pe.funding_types.str.contains("Grant", na=False)
g = pe[pe.is_grant].merge(dm[["firm_id", "founding_month_idx_v2"]],
                          left_on="crunchbase_url", right_on="firm_id")
g["off"] = g.midx - g.founding_month_idx_v2
g = g[g.off >= 0]
first_grant = g.groupby("firm_id")["off"].min()
dm["go"] = dm.firm_id.map(first_grant)
dm["fu"] = CUTOFF_IDX - dm.founding_month_idx_v2

TMAX_S = 60
months = np.arange(0, TMAX_S + 1)
states, nrisk = {}, {}
for circ in (0, 1):
    d = dm[dm.circ == circ]
    po = d.first_private_offset_m_v2.to_numpy(dtype=float)
    go = d.go.to_numpy(dtype=float)
    fu = d.fu.to_numpy(dtype=float)
    rows, ns = [], {}
    for m in months:
        obs = fu >= m
        p = np.nan_to_num(po[obs] <= m)
        q = np.nan_to_num(go[obs] <= m)
        rows.append([np.mean(p & ~q), np.mean(p & q), np.mean(~p & q),
                     np.mean(~p & ~q)])
        if m in RISK_TS:
            ns[m] = int(obs.sum())
    states[circ] = np.array(rows)
    nrisk[circ] = ns

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True)
fig.subplots_adjust(wspace=0.18)
LABELS = ["private only", "private + grant", "grant only", "no recorded financing"]
COLORS = [BLUE, DARKBLUE, RED, LIGHTGRAY]
for ax, circ, plab in ((axes[0], 0, "A  Non-circular"), (axes[1], 1, "B  Circular")):
    st = states[circ].T
    ax.stackplot(months, st, colors=COLORS, linewidth=0)
    cum = np.cumsum(st, axis=0)
    if circ == 1:  # direct labels once, on the right panel margin
        for k, (lab, color) in enumerate(zip(LABELS, COLORS)):
            lo = cum[k - 1, -1] if k else 0.0
            mid = (lo + cum[k, -1]) / 2
            ax.annotate(lab, (months[-1], mid), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7.5,
                        color=INK)
    ax.set_xlim(0, TMAX_S)
    ax.set_xticks(range(0, TMAX_S + 1, 12))
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Months since founding")
    ax.set_title(plab, fontsize=9, loc="left", pad=4)
    ax.grid(False)
    if circ == 0:
        ax.annotate("Observed", (0, 0), xytext=(-26, -44),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=INK, annotation_clip=False)
    for t in RISK_TS:
        dx = -6 if (circ == 0 and t == TMAX_S) else (6 if (circ == 1 and t == 0) else 0)
        ax.annotate(f"{nrisk[circ][t]:,}", (t, 0), xytext=(dx, -44),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=INK, annotation_clip=False)
axes[0].set_ylabel("Share of firms")
save(fig, "fig1_financing_states")
for circ, lab in ((1, "circ"), (0, "non-circ")):
    s35 = states[circ][35]
    print("  f9 check @35m {}: private any {:.1f}% (fund36 {:.1f}%) | "
          "grant-only {:.1f}% | both {:.1f}%".format(
              lab, 100 * (s35[0] + s35[1]),
              100 * dm[dm.circ == circ].fund36_private_v2.mean(),
              100 * s35[2], 100 * s35[1]))

print("done.")
