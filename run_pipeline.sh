#!/bin/sh
# One-command pipeline. See README.md for the two reproduction tiers:
#   Tier 1 (data package): run steps 6-9 individually (README command block);
#     the matched samples ship pre-built and step 5 reads the raw panel_lean.csv.
#   Tier 2 (full rebuild from raw Crunchbase + CORDIS bulks): this script, 1-9.
# Normal run:      sh run_pipeline.sh
# New data export: REBASE=1 SANITY_REPORT=1 sh run_pipeline.sh
#   (pins are reported, not enforced; procedure in scripts/sanity_checks.py).
# Step 2b (network fetches for the registry enrichment) is NOT included;
# it is optional, cache-first and started individually when needed (README).
set -e
cd "$(dirname "$0")"
PY="conda run -n ml python"
RS="${RSCRIPT:-Rscript}"   # override with RSCRIPT=/path/to/Rscript if needed

echo "== [1/9] Raw CORDIS match =="
$PY scripts/match_cordis.py
echo "== [2/9] Finalize the verified grant match =="
$PY scripts/validate_grant_match.py finalize
echo "== [2b] Registry matching from existing cache/bulks (offline) =="
$PY scripts/match_registry.py --arm finalize || echo "   (no registry data present, v2 == v1, ok)"
echo "== [3/9] Analysis dataframe build =="
$PY scripts/build_analysis.py
echo "== [4/9] Sanity gate =="
$PY scripts/sanity_checks.py
echo "== [5/9] Matched grant samples (v1 + v2 + mv) =="
$PY scripts/build_grants_sample.py
$PY scripts/build_grants_sample.py --fidx founding_month_idx_v2 --out grants_matched_v2.csv
$PY scripts/build_grants_sample.py --fidx founding_month_idx_v2 --mv-only --out grants_matched_mv.csv
echo "== [6/9] Pre-dynamics + financing-sequence stats =="
$PY scripts/grants_predynamics.py
$PY scripts/grants_predynamics.py --v2
$PY scripts/grants_predynamics.py --mv
$PY scripts/sequencing_stats.py
$PY scripts/frame_robustness_build.py
echo "== [7/9] R estimation (numbered analysis sequence) =="
REBASE="${REBASE:-}" $RS analysis/2_descriptive_statistics.R
REBASE="${REBASE:-}" $RS analysis/3_time_to_event.R
REBASE="${REBASE:-}" $RS analysis/4_matched_grants.R
REBASE="${REBASE:-}" $RS analysis/5_robustness_windows.R
REBASE="${REBASE:-}" $RS analysis/6_survival_exploration.R
REBASE="${REBASE:-}" $RS analysis/8_frame_robustness.R
echo "== [8/9] Figures =="
$PY scripts/render_figures.py
echo "== [9/9] Appendix tables =="
$PY scripts/appendix_tables.py
echo "== PIPELINE COMPLETE =="
