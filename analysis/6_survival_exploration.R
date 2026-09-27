# 6_survival_exploration.R: survival and exit outcomes.
# Template: Kennedy et al. (2024, Nature Energy) exit-vs-failure outcomes on a
# funded-only frame; documented deviation: status-at-cutoff LPMs instead of
# dated Cox (62% of closures undated). Same S0/S1/S2 spec trio as the gap
# models (3_time_to_event.R):
#   S0  raw difference in means
#   S1  + land x cohort fixed effects            (exposure lives here: cohort FE)
#   S2  S1 + sector dummies + team composition
# Outcomes: status_closed (failure analog), exit_any (Was Acquired v IPO), plus a
# "dead by any signal" robustness (Closed v website dead at the 2026-03 scrape).
# KM curve data ONLY for dated exits (first_ma_offset_m), censored at panel end 2026-03.
# NO subtype regressions (cells too small, e.g. ~8 Waste closures); descriptive rates only.
#
# Inputs:  data/processed/firms_analysis.csv  (built by scripts/build_analysis.py)
# Outputs: outputs/estimates/survival_estimates.csv + survival_km_exit.csv
#          (+ survival_sessioninfo.txt)
# Run:     from the repo root, after build_analysis.py (run_pipeline.sh step 6).
#
# Circular Startup Financing. Master's thesis, 2026.
###############################################################################
library(fixest)
library(survival)

# ---- Load + composition controls ----
if (!file.exists("data/processed/firms_analysis.csv"))
  stop("Run this script from the repository root (see run_pipeline.sh).")
df <- read.csv("data/processed/firms_analysis.csv")  # pipeline step 3 output
if (Sys.getenv("REBASE") != "1") stopifnot(nrow(df) == 22039)  # REBASE=1 lifts pins (new data export, see scripts/sanity_checks.py)

IND <- grep("^ind_", names(df), value = TRUE)
if (Sys.getenv("REBASE") != "1") stopifnot(length(IND) == 20)

df$founders_cat <- cut(df$n_founders, c(0, 1, 2, 3, 4, Inf),
                       labels = c("1", "2", "3", "4", "5plus"))
df$founders_cat <- factor(ifelse(is.na(df$founders_cat), "missing",
                                 as.character(df$founders_cat)))
df$nind_cat <- cut(df$n_industries, c(0, 2, 4, 6, Inf),
                   labels = c("1-2", "3-4", "5-6", "7plus"))
df$nind_cat <- factor(ifelse(is.na(df$nind_cat), "none", as.character(df$nind_cat)))

S2_RHS <- paste(c("founders_cat", "nind_cat", IND), collapse = " + ")

# dead by any signal: Closed status OR dead website at the 2026-03-27 scrape.
# Firms without a website on file (n small) count via status only.
df$dead_any <- as.integer(df$status_closed == 1 |
                          (!is.na(df$site_dead_2026_03) & df$site_dead_2026_03 == 1))

rows <- list()
grab <- function(m, run_id, outcome, spec, vcov_lab, terms = NULL) {
  ct <- coeftable(m)
  keep <- if (is.null(terms)) rownames(ct)[grepl("circ", rownames(ct))] else terms
  keep <- intersect(keep, rownames(ct))
  for (tm in keep) {
    rows[[length(rows) + 1]] <<- data.frame(
      run_id = run_id, outcome = outcome, spec = spec, term = tm,
      estimate = ct[tm, 1], se = ct[tm, 2], p = ct[tm, 4],
      n = m$nobs, vcov = vcov_lab)
  }
}

# ---- Frame trio: S0/S1/S2 LPMs for the three outcomes ----
OUTCOMES <- c("status_closed", "exit_any", "dead_any")
for (y in OUTCOMES) {
  f0 <- as.formula(paste(y, "~ circ"))
  f1 <- as.formula(paste(y, "~ circ | land^t_year"))
  f2 <- as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year"))
  grab(feols(f0, df, vcov = "hetero"), paste0("SV0_", y), y, "S0_raw", "HC1")
  grab(feols(f1, df, vcov = "hetero"), paste0("SV1_", y), y, "S1_landxcohort", "HC1")
  grab(feols(f2, df, vcov = "hetero"), paste0("SV2_", y), y, "S2_composition", "HC1")
}

# ---- Robustness: land-clustered SEs ----
# 20 clusters, few-cluster caveat; both main outcomes
for (y in c("status_closed", "exit_any")) {
  grab(feols(as.formula(paste(y, "~ circ | land^t_year")), df, cluster = ~land),
       paste0("SV1_", y, "_cland"), y, "S1_landxcohort", "cluster_land")
}

# ---- Month-verified layer (Chapter 4.5 reporting sample) ----
# Chapter 4.5 reports the survival paragraph on the mv estimation sample; the
# frame trio above stays as the appendix robustness. Raw rates persisted
# alongside.
d_mv <- df[df$founding_month_imputed_v2 == 0, ]
cat(sprintf("month-verified layer: %d firms\n", nrow(d_mv)))
for (y in OUTCOMES) {
  grab(feols(as.formula(paste(y, "~ circ")), d_mv, vcov = "hetero"),
       paste0("SV0_MV_", y), y, "S0_raw_mv", "HC1")
  grab(feols(as.formula(paste(y, "~ circ | land^t_year")), d_mv, vcov = "hetero"),
       paste0("SV1_MV_", y), y, "S1_landxcohort_mv", "HC1")
  grab(feols(as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year")), d_mv,
             vcov = "hetero"), paste0("SV2_MV_", y), y, "S2_composition_mv", "HC1")
  for (cc in 0:1) {
    rows[[length(rows) + 1]] <- data.frame(
      run_id = paste0("RATE_MV_", y, "_circ", cc), outcome = y,
      spec = "descriptive_rate_mv", term = paste0("circ=", cc),
      estimate = mean(d_mv[[y]][d_mv$circ == cc]), se = NA, p = NA,
      n = sum(d_mv$circ == cc), vcov = "none")
  }
}

# ---- Descriptive subtype rates (frame) ----
# closure/exit rates by circular sub-type; NO regressions: cells too small
for (y in c("status_closed", "exit_any")) {
  r <- aggregate(df[[y]], list(subtype = df$circ_subtype), mean)
  n <- aggregate(df[[y]], list(subtype = df$circ_subtype), length)
  for (i in seq_len(nrow(r))) {
    rows[[length(rows) + 1]] <- data.frame(
      run_id = paste0("RATE_", y, "_", r$subtype[i]), outcome = y,
      spec = "descriptive_rate", term = r$subtype[i],
      estimate = r$x[i], se = NA, p = NA, n = n$x[i], vcov = "none")
  }
}

# ---- Appendix 11: MV subtype rates + S2 subtype LPM (dead_any) ----
# additive appendix rows:
# (a) subtype rates on the MV sample (the frame rates above stay), now also
# for dead_any; (b) S2 subtype LPM for dead_any ONLY, the larger-cell
# outcome; closed stays descriptive per the documented small-cell decision.
for (y in c("status_closed", "dead_any")) {
  r <- aggregate(d_mv[[y]], list(subtype = d_mv$circ_subtype), mean)
  n <- aggregate(d_mv[[y]], list(subtype = d_mv$circ_subtype), length)
  for (i in seq_len(nrow(r))) {
    rows[[length(rows) + 1]] <- data.frame(
      run_id = paste0("RATE_MV_", y, "_", r$subtype[i]), outcome = y,
      spec = "descriptive_rate_mv", term = r$subtype[i],
      estimate = r$x[i], se = NA, p = NA, n = n$x[i], vcov = "none")
  }
}
d_mv$circ_subtype <- relevel(factor(d_mv$circ_subtype), ref = "None")
SUBT <- paste0("circ_subtype", c("Design", "Platform", "Nature", "Service", "Waste"))
grab(feols(as.formula(paste("dead_any ~ circ_subtype +", S2_RHS, "| land^t_year")),
           d_mv, vcov = "hetero"),
     "SV2_MV_SUBTYPE_dead_any", "dead_any", "S2_subtypes_mv", "HC1", terms = SUBT)

# ---- KM curve data for dated exits ----
# Kaplan-Meier for DATED exits only (first M&A event month).
# Time in months from founding; firms without a dated M&A are censored at panel end
# 2026-03 (month_index 2026*12 + 2 = 24314, single definition in src/cohort.py).
CUTOFF_IDX <- 2026 * 12 + 2
df$exit_time <- ifelse(df$ma_any_dated == 1, df$first_ma_offset_m,
                       CUTOFF_IDX - df$founding_month_idx)
stopifnot(all(df$exit_time >= 0))
km <- survfit(Surv(exit_time, ma_any_dated) ~ circ, data = df)
kmd <- data.frame(group = rep(names(km$strata), km$strata),
                  time_m = km$time, surv = km$surv,
                  lower = km$lower, upper = km$upper, n_risk = km$n.risk)
dir.create("outputs/estimates", showWarnings = FALSE, recursive = TRUE)
write.csv(kmd, "outputs/estimates/survival_km_exit.csv", row.names = FALSE)

out <- do.call(rbind, rows)
write.csv(out, "outputs/estimates/survival_estimates.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/estimates/survival_sessioninfo.txt")
cat(sprintf("wrote survival_estimates.csv (%d rows) + survival_km_exit.csv (%d rows)\n",
            nrow(out), nrow(kmd)))

hl <- out[out$term == "circ" & out$vcov == "HC1", ]
cat("\ncirc coefficient by outcome/spec (pp):\n")
print(data.frame(outcome = hl$outcome, spec = hl$spec,
                 est_pp = round(100 * hl$estimate, 2),
                 se_pp = round(100 * hl$se, 2), p = signif(hl$p, 3)),
      row.names = FALSE)
