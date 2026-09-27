# Circular Startup Financing in Europe

Code and results for my master's thesis on the financing of circular startups
in Europe: 22,039 funded startups from 20 countries (Crunchbase, founding
cohorts 2016-2022), a circularity classification, an individually verified
match to EU grants (CORDIS), registry-enriched founding dates and the
estimations in R (Kaplan-Meier and Cox models plus a cumulative incidence
decomposition for the financing gap, a matched grant comparison with pseudo
award dates for the EU grant treatment).

Every number in the thesis reproduces from `scripts/` and `analysis/`. The
result CSVs behind each table sit in `outputs/estimates/` and
`outputs/tables/appendix/`, the four main-text figures in `outputs/figures/`.
The repository is a numbered analysis sequence: code and results only, no
proprietary data.

Marc Schuler, 2026

## Repository structure

- `src/` shared modules (loaders, cohort and outcome definitions, the closed
  column schema)
- `scripts/` the pipeline steps (matching, dataframe build, matched samples,
  figures, appendix tables)
- `analysis/` the numbered analysis sequence: `1_data_cleaning.ipynb` is the
  executed cleaning walkthrough, `2` to `6` and `8` are the R models,
  `7_figures.ipynb` is the figure QA companion
- `outputs/` estimate CSVs, appendix tables and figures
- `data/` empty here, filled from the data package

## Data availability

Crunchbase and registry data are licensed, so I cannot put them here. A
separate data package accompanies the thesis. Unpack it in the repository
root so its files land at the paths below; `data/` and the firm-level match
tables are git-ignored, nothing put there can end up in the repository.

Tier 1 (~30 MB) reproduces all estimations, tables and figures:

| File(s) | Role |
|---|---|
| `data/processed/firms_analysis.parquet` + `.csv` | the analysis dataframe (22,039 firms x 76 columns) |
| `data/processed/panel_events.parquet` | dated funding and M&A events |
| `data/processed/grants_matched*.csv` | matched grant samples |
| `data/scrape/registry/registry_match_final.csv` | verified registry matches |
| `data/frozen_input/firms_analysis.csv` | build reconciliation reference |
| `outputs/tables/grant_match_final.csv`, `cordis_firm_grants.csv` + verdict store | verified grant match, pair and participation level |
| `variables.md` | codebook of the 76 analysis columns |

The raw firm table `data/raw/companies_fat.csv` from the original Crunchbase
export is not repacked; the descriptive statistics and appendix tables read
it at that path (patent and IP columns). Tier 2, the full rebuild from the
raw Crunchbase and CORDIS bulks (~42 GB), is available on request.

## How to run

Python 3.11 in a conda env named `ml` (`pip install -r requirements.txt`) and
R 4.5.3 with `fixest` and `survival`; the exact versions behind the thesis
numbers are logged in `outputs/estimates/*_sessioninfo.txt`. Run everything
from the repository root.

With the Tier 1 package in place, run pipeline steps 6 to 9:

```
conda run -n ml python scripts/grants_predynamics.py
conda run -n ml python scripts/grants_predynamics.py --v2
conda run -n ml python scripts/grants_predynamics.py --mv
conda run -n ml python scripts/sequencing_stats.py
conda run -n ml python scripts/frame_robustness_build.py
Rscript analysis/2_descriptive_statistics.R
Rscript analysis/3_time_to_event.R
Rscript analysis/4_matched_grants.R
Rscript analysis/5_robustness_windows.R
Rscript analysis/6_survival_exploration.R
Rscript analysis/8_frame_robustness.R
conda run -n ml python scripts/render_figures.py
conda run -n ml python scripts/appendix_tables.py
```

The regenerated `outputs/estimates/*.csv` match the shipped ones (all
sampling is seeded). With the Tier 2 data in place, `sh run_pipeline.sh` runs
the whole chain from the raw files. The optional registry fetch stage is
network-bound and cache-first; it needs API keys in `keys.env` (template in
`keys.env.example`, never committed).

## Which output feeds which thesis exhibit

| Thesis exhibit | Source |
|---|---|
| Table 1 (descriptives) | `outputs/estimates/descriptives_t1.csv` (+ `_frame` variant) |
| Gap tables (Ch. 4.2/4.3) | `outputs/estimates/timing_estimates.csv`, `timing_schoenfeld.csv` |
| Grant tables (Ch. 4.4/4.5) | `outputs/estimates/grant_estimates.csv`, `sequencing_stats.csv`, `grants_predynamics_mv.csv` |
| Survival section | `outputs/estimates/survival_estimates.csv` |
| Frame robustness (Ch. 5.3) | `outputs/estimates/frame_robustness.csv`, `frame_entry_tabulation.csv` |
| Figures 1 to 4 | `outputs/figures/fig1` to `fig4` from `scripts/render_figures.py` |
| Appendix 4 to 12 tables | `outputs/tables/appendix/appendix{04..12}*.csv` |
| Founding-window robustness | `outputs/estimates/gap_estimates.csv`, `gap_km_private.csv` |

## Pinned checks

Row counts, anchor rates and sample sizes are pinned as constants in the
scripts and break the run when the data change. That protects the thesis
numbers against silent drift. For a new data export run
`REBASE=1 SANITY_REPORT=1 sh run_pipeline.sh`, review the reported deviations
and re-anchor the constants (details at the top of `scripts/sanity_checks.py`).
