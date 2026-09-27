# 5_robustness_windows.R: the January-convention (v1) robustness layer,
# window LPMs S0/S1/S2, W24/48, CONF, DEC, TREND, PREC v1, logit AME,
# cluster SEs and the precision KM curves. Main models: 3_time_to_event.R.
#
# Gap estimation, circular vs non-circular financing gap. Three specs per
# outcome:
#   S0  raw difference in means
#   S1  + land x cohort fixed effects            (same country, same founding year)
#   S2  S1 + sector dummies + team composition   (S1 - S2 = composition share)
# Estimator: LPM, HC1 SEs (treatment varies at firm level); land-clustered SEs and a
# logit AME reported as robustness on the main outcome. Sub-type split + cohort trend.
# Deterministic (no RNG).
#
# Inputs:  data/processed/firms_analysis.csv  (built by scripts/build_analysis.py)
# Outputs: outputs/estimates/gap_estimates.csv, gap_km_private.csv
#          (+ gap_sessioninfo.txt)
# Run:     from the repo root, after build_analysis.py (run_pipeline.sh step 6).
#
# Circular Startup Financing. Master's thesis, 2026.
###############################################################################
library(fixest)

# ---- Load + composition controls ----
if (!file.exists("data/processed/firms_analysis.csv"))
  stop("Run this script from the repository root (see run_pipeline.sh).")
df <- read.csv("data/processed/firms_analysis.csv")  # pipeline step 3 output
if (Sys.getenv("REBASE") != "1") stopifnot(nrow(df) == 22039)  # REBASE=1 lifts pins (new data export, see scripts/sanity_checks.py)

OUTCOMES <- c("fund36_private", "logfund36_private", "success36", "grant36_cb", "fund36")
IND <- grep("^ind_", names(df), value = TRUE)
if (Sys.getenv("REBASE") != "1") stopifnot(length(IND) == 20)

df$founders_cat <- cut(df$n_founders, c(0, 1, 2, 3, 4, Inf),
                       labels = c("1", "2", "3", "4", "5plus"))
df$founders_cat <- factor(ifelse(is.na(df$founders_cat), "missing",
                                 as.character(df$founders_cat)))
df$nind_cat <- cut(df$n_industries, c(0, 2, 4, 6, Inf),
                   labels = c("1-2", "3-4", "5-6", "7plus"))
df$nind_cat <- factor(ifelse(is.na(df$nind_cat), "none", as.character(df$nind_cat)))
df$circ_subtype <- relevel(factor(df$circ_subtype), ref = "None")
df$trend <- df$t_year - 2016

S2_RHS <- paste(c("founders_cat", "nind_cat", IND), collapse = " + ")

rows <- list()
grab <- function(m, run_id, outcome, spec, vcov_lab, terms = NULL) {
  ct <- coeftable(m)
  keep <- if (is.null(terms)) rownames(ct)[grepl("circ|treated|t_year", rownames(ct))] else terms
  keep <- intersect(keep, rownames(ct))
  for (tm in keep) {
    rows[[length(rows) + 1]] <<- data.frame(
      run_id = run_id, outcome = outcome, spec = spec, term = tm,
      estimate = ct[tm, 1], se = ct[tm, 2], p = ct[tm, 4],
      n = m$nobs, vcov = vcov_lab)
  }
}

# ---- S0/S1/S2 ladder on the five v1 window outcomes ----
for (y in OUTCOMES) {
  f0 <- as.formula(paste(y, "~ circ"))
  f1 <- as.formula(paste(y, "~ circ | land^t_year"))
  f2 <- as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year"))
  grab(feols(f0, df, vcov = "hetero"), paste0("S0_", y), y, "S0_raw", "HC1")
  grab(feols(f1, df, vcov = "hetero"), paste0("S1_", y), y, "S1_landxcohort", "HC1")
  grab(feols(f2, df, vcov = "hetero"), paste0("S2_", y), y, "S2_composition", "HC1")
}

# ---- SE + estimator robustness (cluster-land, logit AME) ----
# land-clustered robustness (20 clusters, few-cluster caveat applies), main outcome
grab(feols(fund36_private ~ circ | land^t_year, df, cluster = ~land),
     "S1_fund36_private_cland", "fund36_private", "S1_landxcohort", "cluster_land")
grab(feols(as.formula(paste("fund36_private ~ circ +", S2_RHS, "| land^t_year")),
           df, cluster = ~land),
     "S2_fund36_private_cland", "fund36_private", "S2_composition", "cluster_land")

# logit AME robustness, main outcome, S2 controls (FE cells without variation drop)
m_logit <- feglm(as.formula(paste("fund36_private ~ circ +", S2_RHS, "| land^t_year")),
            df, family = binomial())
d_circ1 <- df; d_circ1$circ <- 1L
d_circ0 <- df; d_circ0$circ <- 0L
ame <- mean(predict(m_logit, newdata = d_circ1, type = "response")
            - predict(m_logit, newdata = d_circ0, type = "response"), na.rm = TRUE)
rows[[length(rows) + 1]] <- data.frame(
  run_id = "S2_fund36_private_logitAME", outcome = "fund36_private",
  spec = "S2_logit_AME", term = "circ_AME", estimate = ame, se = NA, p = NA,
  n = m_logit$nobs, vcov = "point_estimate_only")

# ---- Sub-type split (v1 January layer) ----
# mechanism: five circular sub-types vs non-circular base
for (y in c("fund36_private", "grant36_cb", "success36")) {
  fs1 <- as.formula(paste(y, "~ circ_subtype | land^t_year"))
  fs2 <- as.formula(paste(y, "~ circ_subtype +", S2_RHS, "| land^t_year"))
  grab(feols(fs1, df, vcov = "hetero"), paste0("SUB1_", y), y, "S1_subtypes", "HC1",
       terms = paste0("circ_subtype", c("Design", "Platform", "Nature", "Service", "Waste")))
  grab(feols(fs2, df, vcov = "hetero"), paste0("SUB2_", y), y, "S2_subtypes", "HC1",
       terms = paste0("circ_subtype", c("Design", "Platform", "Nature", "Service", "Waste")))
}

# ---- success36 decomposition ----
# which component drives the +3.8pp flip?
# rounds3 = the ≥3-event-months margin; bigmonth = a >$10M month; ma = M&A in window.
df$success36_rounds3 <- as.integer(df$n_fund_events_36 >= 3)
df$rounds3_private <- as.integer(df$n_private_months_36 >= 3)   # grant-free variant
for (y in c("success36_rounds3", "rounds3_private", "success36_bigmonth", "success36_ma")) {
  f1 <- as.formula(paste(y, "~ circ | land^t_year"))
  f2 <- as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year"))
  grab(feols(f1, df, vcov = "hetero"), paste0("DEC1_", y), y, "S1_landxcohort", "HC1")
  grab(feols(f2, df, vcov = "hetero"), paste0("DEC2_", y), y, "S2_composition", "HC1")
}

# ---- Classifier-confidence robustness ----
# drop circular labels with cbm_confidence < 0.8 (~28% of circ); if the gap
# holds on high-confidence labels, classifier noise cannot explain it
hc <- df[!(df$circ == 1 & df$cbm_confidence < 0.8), ]
grab(feols(fund36_private ~ circ | land^t_year, hc, vcov = "hetero"),
     "CONF1_fund36_private", "fund36_private", "S1_conf_ge0.8", "HC1")
grab(feols(as.formula(paste("fund36_private ~ circ +", S2_RHS, "| land^t_year")),
           hc, vcov = "hetero"),
     "CONF2_fund36_private", "fund36_private", "S2_conf_ge0.8", "HC1")

# ---- Window robustness: 24m and 48m horizons ----
# the same gap on 24m and 48m windows, derived from first_private_offset_m (uncensored months to first private
# round; consistency checked: offset<=35 reproduces fund36_private exactly).
# 24m is complete for every firm; 48m only for cohorts 2016-2021
# (Dec 2021 + 47m = Nov 2025 <= panel end 2026-03), so W48 restricts to them.
df$fund24_private <- as.integer(!is.na(df$first_private_offset_m)
                                & df$first_private_offset_m <= 23)
df$fund48_private <- as.integer(!is.na(df$first_private_offset_m)
                                & df$first_private_offset_m <= 47)
stopifnot(all(df$fund36_private ==
              as.integer(!is.na(df$first_private_offset_m)
                         & df$first_private_offset_m <= 35)))
d48 <- df[df$t_year <= 2021, ]
grab(feols(fund24_private ~ circ | land^t_year, df, vcov = "hetero"),
     "W24_S1_fund24_private", "fund24_private", "S1_landxcohort", "HC1")
grab(feols(as.formula(paste("fund24_private ~ circ +", S2_RHS, "| land^t_year")),
           df, vcov = "hetero"),
     "W24_S2_fund24_private", "fund24_private", "S2_composition", "HC1")
grab(feols(fund48_private ~ circ | land^t_year, d48, vcov = "hetero"),
     "W48_S1_fund48_private", "fund48_private", "S1_landxcohort_coh16_21", "HC1")
grab(feols(as.formula(paste("fund48_private ~ circ +", S2_RHS, "| land^t_year")),
           d48, vcov = "hetero"),
     "W48_S2_fund48_private", "fund48_private", "S2_composition_coh16_21", "HC1")

# ---- Gap over time ----
# per-cohort gaps + one linear-trend interaction
mt <- feols(fund36_private ~ i(t_year, circ) | land^t_year, df, vcov = "hetero")
grab(mt, "TREND_percohort", "fund36_private", "S1_gap_by_cohort", "HC1",
     terms = grep("t_year", rownames(coeftable(mt)), value = TRUE))
mtl <- feols(fund36_private ~ circ + circ:trend | land^t_year, df, vcov = "hetero")
grab(mtl, "TREND_linear", "fund36_private", "S1_gap_linear_trend", "HC1",
     terms = c("circ", "circ:trend"))

# ---- Founding-date-precision split (v1 PREC) ----
# 55% of founding
# months are year-imputed to January (founding_month_imputed). The S1 gap differs
# sharply by half (audit read-only: precise -2.3pp vs imputed -8.6pp); the
# substitution result is robust in both halves. Report both splits for the
# main outcome and grant36_cb; the thesis reads this as concentration of the
# penalty in the less-documented, asset-heavier half (precise circular firms
# are 32% vs 19% Platform), stated honestly in the limitations.
for (prec in c(0, 1)) {
  dp <- df[df$founding_month_imputed == prec, ]
  lab <- if (prec == 0) "precise" else "imputed"
  for (y in c("fund36_private", "grant36_cb")) {
    grab(feols(as.formula(paste(y, "~ circ | land^t_year")), dp, vcov = "hetero"),
         paste0("PREC1_", lab, "_", y), y, paste0("S1_", lab, "_founding_dates"), "HC1")
    grab(feols(as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year")), dp,
               vcov = "hetero"),
         paste0("PREC2_", lab, "_", y), y, paste0("S2_", lab, "_founding_dates"), "HC1")
  }
}

# ---- Hazard robustness (Cox, v1 clock) ----
# the SAME gap estimand in time-to-event form: Cox PH of months to the FIRST
# private round from founding,
# censored at panel end 2026-03, baseline hazard stratified by land x cohort
# (the S1 analog). Template: Fosfuri & Nagar (2026, RP) §4.1, simplified (no
# quarterly time-varying covariates). +1 month shift because in-founding-month
# events would otherwise have zero-length intervals; rank-based Cox is unaffected.
library(survival)
CUTOFF_IDX <- 2026 * 12 + 2   # month_index("2026-03"), single definition src/cohort.py
haz_event <- as.integer(!is.na(df$first_private_offset_m))
haz_time <- ifelse(haz_event == 1, df$first_private_offset_m,
                   CUTOFF_IDX - df$founding_month_idx) + 1
stopifnot(all(haz_time > 0))
cx <- coxph(Surv(haz_time, haz_event) ~ circ + strata(land, t_year), data = df)
cs <- summary(cx)$coefficients
rows[[length(rows) + 1]] <- data.frame(
  run_id = "HAZ_S1_first_private", outcome = "time_to_first_private",
  spec = "S1_cox_strata_landxcohort", term = "circ_loghr",
  estimate = cs["circ", "coef"], se = cs["circ", "se(coef)"],
  p = cs["circ", "Pr(>|z|)"], n = cx$n, vcov = "cox_model_based")
cat(sprintf("\nhazard robustness: circ HR = %.3f (log %.3f, se %.3f, p = %.2g)\n",
            exp(cs["circ", "coef"]), cs["circ", "coef"], cs["circ", "se(coef)"],
            cs["circ", "Pr(>|z|)"]))

# ---- Precision KM curve data ----
# KM of months to the FIRST private round, circ vs non-circ, censored at
# panel end.
# Primary sample = precise founding dates (year-imputed months distort the time
# axis by up to +/-11 months); pooled written as variant. Cumulative incidence
# for the figure = 1 - surv. All 36m windows are complete by design, so KM at
# month 36 must equal the raw fund36_private rates (verification anchor).
df$haz_event <- haz_event
df$haz_time <- haz_time
km_rows <- list()
for (samp in c("precise", "pooled")) {
  ds <- if (samp == "precise") df[df$founding_month_imputed == 0, ] else df
  km <- survfit(Surv(haz_time, haz_event) ~ circ, data = ds)
  km_rows[[samp]] <- data.frame(sample = samp,
                                group = rep(names(km$strata), km$strata),
                                time_m = km$time, surv = km$surv,
                                lower = km$lower, upper = km$upper,
                                n_risk = km$n.risk)
}
kmp <- do.call(rbind, km_rows)
write.csv(kmp, "outputs/estimates/gap_km_private.csv", row.names = FALSE)
cat(sprintf("wrote gap_km_private.csv (%d rows)\n", nrow(kmp)))

# ---- Write estimates + headline ----
out <- do.call(rbind, rows)
dir.create("outputs/estimates", showWarnings = FALSE, recursive = TRUE)
write.csv(out, "outputs/estimates/gap_estimates.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/estimates/gap_sessioninfo.txt")
cat(sprintf("wrote gap_estimates.csv (%d rows)\n", nrow(out)))

hl <- out[out$outcome == "fund36_private" & out$term == "circ"
          & out$vcov == "HC1" & out$spec %in% c("S0_raw", "S1_landxcohort", "S2_composition"), ]
cat("\nheadline fund36_private gap (pp):\n")
print(data.frame(spec = hl$spec, gap_pp = round(100 * hl$estimate, 2),
                 se_pp = round(100 * hl$se, 2)), row.names = FALSE)
