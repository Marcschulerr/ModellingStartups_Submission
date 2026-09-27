# 3_time_to_event.R: MAIN MODELS, time to the first private round on the
# registry-enriched (v2) founding months. Kaplan-Meier curves, Cox proportional
# hazards (land x cohort strata), and cumulative-incidence-difference
# decompositions at 24/36/48 months (S0 raw -> S1 within land x cohort ->
# S2 + sector/team).
#
# MAIN SAMPLE = month-verified founding dates only: following Kennedy et al.
# 2024 and Fosfuri & Nagar 2026, firms without a
# month-precise founding date are excluded from the time-to-event analysis.
# The full frame (incl. January-imputed) runs as the V2FULL_* robustness layer.
#
# Inputs:  data/processed/firms_analysis.csv  (built by scripts/build_analysis.py)
#          outputs/tables/prefounding_private.csv          (flags from
#            scripts/sequencing_stats.py; that robustness skipped if absent)
#          data/scrape/registry/registry_match_final.csv   (dissolution dates;
#            the dissolution-censoring robustness skipped if absent)
# Outputs: outputs/estimates/timing_estimates.csv, timing_km_private.csv,
#          timing_schoenfeld.csv (+ timing_sessioninfo.txt)
# Run:     from the repo root, after build_analysis.py (run_pipeline.sh step 6).
#
# Circular Startup Financing. Master's thesis, 2026.
# Templates: Kennedy et al. 2024 (funded-only frame, strata, KM/Cox, exclusion
#            of firms without founding information); Fosfuri & Nagar 2026
#            (time to first round; incomplete-info exclusion); Guzman &
#            Kacperczyk 2019 (decomposition, here on the incidence scale).
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
df$circ_subtype <- relevel(factor(df$circ_subtype), ref = "None")
S2_RHS <- paste(c("founders_cat", "nind_cat", IND), collapse = " + ")
CUTOFF_IDX <- 2026 * 12 + 2  # month_index("2026-03"), single definition src/cohort.py

df$fund24_private_v2 <- as.integer(!is.na(df$first_private_offset_m_v2)
                                   & df$first_private_offset_m_v2 <= 23)
df$fund48_private_v2 <- as.integer(!is.na(df$first_private_offset_m_v2)
                                   & df$first_private_offset_m_v2 <= 47)
stopifnot(all(df$fund36_private_v2 ==
              as.integer(!is.na(df$first_private_offset_m_v2)
                         & df$first_private_offset_m_v2 <= 35)))

# the analysis sample for everything below; df stays the full-frame layer
prec <- df$founding_month_imputed_v2 == 0
d_mv <- df[prec, ]
cat(sprintf("time-to-event sample (month-verified): %d of %d firms (%.1f%%)\n",
            nrow(d_mv), nrow(df), 100 * mean(prec)))

rows <- list()
grab <- function(m, run_id, outcome, spec, vcov_lab, terms = NULL) {
  ct <- coeftable(m)
  keep <- if (is.null(terms)) rownames(ct)[grepl("circ", rownames(ct))] else terms
  for (tm in intersect(keep, rownames(ct))) {
    rows[[length(rows) + 1]] <<- data.frame(
      run_id = run_id, outcome = outcome, spec = spec, term = tm,
      estimate = ct[tm, 1], se = ct[tm, 2], p = ct[tm, 4],
      n = m$nobs, vcov = vcov_lab)
  }
}
cox_row <- function(cx, run_id, outcome, term = "circ",
                    spec = "cox_strata_landxcohort") {
  cs <- summary(cx)$coefficients
  for (tm in intersect(term, rownames(cs))) {
    rows[[length(rows) + 1]] <<- data.frame(
      run_id = run_id, outcome = outcome, spec = spec,
      term = paste0(tm, "_loghr"), estimate = cs[tm, "coef"],
      se = cs[tm, "se(coef)"], p = cs[tm, "Pr(>|z|)"], n = cx$n,
      vcov = "cox_model_based")
  }
}

# ---- Cumulative-incidence differences at fixed horizons (24/36/48m) ----
# fund{h}_private_v2 = 1[T <= h]; the S0->S1->S2 ladder is the composition
# decomposition, additive in percentage points (the Chapter-4.2 gap table).
for (y in c("fund36_private_v2", "logfund36_private_v2", "grant36_cb_v2")) {
  grab(feols(as.formula(paste(y, "~ circ")), d_mv, vcov = "hetero"),
       paste0("V2_S0_", y), y, "S0_raw", "HC1")
  grab(feols(as.formula(paste(y, "~ circ | land^t_year")), d_mv, vcov = "hetero"),
       paste0("V2_S1_", y), y, "S1_landxcohort", "HC1")
  grab(feols(as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year")),
             d_mv, vcov = "hetero"),
       paste0("V2_S2_", y), y, "S2_composition", "HC1")
}
d_w48 <- d_mv[d_mv$t_year <= 2021, ]
# S0 cells for the 24/48m horizons: completes the S0/S1/S2 ladder of the
# Chapter 4.2 gap table at every horizon
grab(feols(fund24_private_v2 ~ circ, d_mv, vcov = "hetero"),
     "V2_W24_S0", "fund24_private_v2", "S0_raw", "HC1")
grab(feols(fund48_private_v2 ~ circ, d_w48, vcov = "hetero"),
     "V2_W48_S0", "fund48_private_v2", "S0_raw_coh16_21", "HC1")
grab(feols(fund24_private_v2 ~ circ | land^t_year, d_mv, vcov = "hetero"),
     "V2_W24_S1", "fund24_private_v2", "S1_landxcohort", "HC1")
grab(feols(as.formula(paste("fund24_private_v2 ~ circ +", S2_RHS, "| land^t_year")),
           d_mv, vcov = "hetero"), "V2_W24_S2", "fund24_private_v2", "S2_composition", "HC1")
grab(feols(fund48_private_v2 ~ circ | land^t_year, d_w48, vcov = "hetero"),
     "V2_W48_S1", "fund48_private_v2", "S1_landxcohort_coh16_21", "HC1")
grab(feols(as.formula(paste("fund48_private_v2 ~ circ +", S2_RHS, "| land^t_year")),
           d_w48, vcov = "hetero"), "V2_W48_S2", "fund48_private_v2",
     "S2_composition_coh16_21", "HC1")

# ---- Subtype incidence @36m (Chapter-4.2 subtype table) ----
# feeds the subtype table together with the TIME_COX_subtypes HRs below
# (v1 subtype LPMs in 5_robustness_windows.R stay the January layer)
SUBT <- paste0("circ_subtype", c("Design", "Platform", "Nature", "Service", "Waste"))
for (y in c("fund36_private_v2", "grant36_cb_v2")) {
  grab(feols(as.formula(paste(y, "~ circ_subtype +", S2_RHS, "| land^t_year")),
             d_mv, vcov = "hetero"),
       paste0("V2_SUBTYPE_S2_", y), y, "S2_subtypes", "HC1", terms = SUBT)
}

# ---- Robustness layers 1-4 (Chapter 5.3) ----
# robustness layer 1: full frame incl. January-imputed (V2FULL_*), shows the
# month-verified exclusion does not drive the results
for (y in c("fund36_private_v2", "grant36_cb_v2")) {
  grab(feols(as.formula(paste(y, "~ circ | land^t_year")), df, vcov = "hetero"),
       paste0("V2FULL_S1_", y), y, "S1_landxcohort_full_frame", "HC1")
  grab(feols(as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year")), df,
             vcov = "hetero"), paste0("V2FULL_S2_", y), y,
       "S2_composition_full_frame", "HC1")
}
# robustness layer 2: the still-imputed side alone (documentation of the
# measurement heterogeneity; the month-verified side IS the main sample)
d_imputed <- df[!prec, ]
for (y in c("fund36_private_v2", "grant36_cb_v2")) {
  grab(feols(as.formula(paste(y, "~ circ | land^t_year")), d_imputed, vcov = "hetero"),
       paste0("V2_PREC1_imputed_", y), y, "S1_imputed_v2_founding", "HC1")
  grab(feols(as.formula(paste(y, "~ circ +", S2_RHS, "| land^t_year")), d_imputed,
             vcov = "hetero"), paste0("V2_PREC2_imputed_", y), y,
       "S2_imputed_v2_founding", "HC1")
}

# robustness layer 3: high-confidence classifier subsample ON THE MAIN SAMPLE
# (cbm_confidence >= 0.8 for circular labels); the cited variant, so the
# robustness range has one base. Plus the logit AME on the main sample
# (mirrors 5_robustness: point estimate only).
d_conf <- d_mv[d_mv$circ == 0 | d_mv$cbm_confidence >= 0.8, ]
grab(feols(fund36_private_v2 ~ circ | land^t_year, d_conf, vcov = "hetero"),
     "V2_CONF_MV_S1", "fund36_private_v2", "S1_conf08_mv", "HC1")
grab(feols(as.formula(paste("fund36_private_v2 ~ circ +", S2_RHS, "| land^t_year")),
           d_conf, vcov = "hetero"),
     "V2_CONF_MV_S2", "fund36_private_v2", "S2_conf08_mv", "HC1")
m_logit <- feglm(as.formula(paste("fund36_private_v2 ~ circ +", S2_RHS, "| land^t_year")),
            d_mv, family = binomial())
d_circ1 <- d_mv; d_circ1$circ <- 1L
d_circ0 <- d_mv; d_circ0$circ <- 0L
ame <- mean(predict(m_logit, newdata = d_circ1, type = "response")
            - predict(m_logit, newdata = d_circ0, type = "response"), na.rm = TRUE)
rows[[length(rows) + 1]] <- data.frame(
  run_id = "V2_LOGIT_MV_AME", outcome = "fund36_private_v2",
  spec = "S2_logit_AME_mv", term = "circ_AME", estimate = ame, se = NA, p = NA,
  n = m_logit$nobs, vcov = "point_estimate_only")
# robustness layer 4: with the register clock, private
# rounds BEFORE the verified founding month are real pre-registration
# financings; the window convention treats those firms as unfunded until
# their next round. Variant: count them as month-0 events (outcome = 1 at
# every horizon; Cox event at t = 1). Flags from scripts/sequencing_stats.py.
pf <- "outputs/tables/prefounding_private.csv"
if (file.exists(pf)) {
  flags <- read.csv(pf)
  d_pre <- merge(d_mv, flags, by = "firm_id", all.x = TRUE)
  d_pre$pre_private_any[is.na(d_pre$pre_private_any)] <- 0L
  d_pre$fund36_pre0 <- pmax(d_pre$fund36_private_v2, d_pre$pre_private_any)
  cat(sprintf("pre-founding-private robustness: %d flagged firms (%.1f%%)\n",
              sum(d_pre$pre_private_any), 100 * mean(d_pre$pre_private_any)))
  grab(feols(fund36_pre0 ~ circ | land^t_year, d_pre, vcov = "hetero"),
       "V2_PREFOUND0_S1", "fund36_pre0", "S1_prefound_month0", "HC1")
  grab(feols(as.formula(paste("fund36_pre0 ~ circ +", S2_RHS, "| land^t_year")),
             d_pre, vcov = "hetero"),
       "V2_PREFOUND0_S2", "fund36_pre0", "S2_prefound_month0", "HC1")
  ev0 <- as.integer(d_pre$pre_private_any == 1 | !is.na(d_pre$first_private_offset_m_v2))
  tt0 <- ifelse(d_pre$pre_private_any == 1, 1,
                ifelse(!is.na(d_pre$first_private_offset_m_v2),
                       d_pre$first_private_offset_m_v2 + 1,
                       CUTOFF_IDX - d_pre$founding_month_idx_v2 + 1))
  cx0 <- coxph(Surv(tt0, ev0) ~ circ + strata(land, t_year), data = d_pre)
  cox_row(cx0, "V2_PREFOUND0_HAZ", "time_to_first_private_pre0")
  # S2 counterpart: Chapter 5.3 reports the robustness layers like-for-like
  # on the S2 hazard form
  cx0_s2 <- coxph(as.formula(paste("Surv(tt0, ev0) ~ circ +", S2_RHS,
                                   "+ strata(land, t_year)")), data = d_pre)
  cox_row(cx0_s2, "V2_PREFOUND0_HAZ_S2", "time_to_first_private_pre0",
          spec = "cox_S2_strata_landxcohort")
  s2p <- coeftable(feols(as.formula(paste("fund36_pre0 ~ circ +", S2_RHS,
                                          "| land^t_year")), d_pre,
                         vcov = "hetero"))["circ", 1]
  cat(sprintf("  prefound-month0: S2 = %+.2fpp | HR = %.3f\n", 100 * s2p,
              exp(summary(cx0)$coefficients["circ", "coef"])))
}

q <- quantile(d_mv$first_private_offset_m_v2, c(.5, .75, .9), na.rm = TRUE)
cat(sprintf("time to first private round, mv reachers (offset months): P50=%.0f P75=%.0f P90=%.0f (n=%d)\n",
            q[1], q[2], q[3], sum(!is.na(d_mv$first_private_offset_m_v2))))

# ---- Cox proportional hazards: months to first private round ----
# baseline hazard stratified by land x founding cohort
surv_parts <- function(d) {
  ev <- as.integer(!is.na(d$first_private_offset_m_v2))
  tt <- ifelse(ev == 1, d$first_private_offset_m_v2,
               CUTOFF_IDX - d$founding_month_idx_v2) + 1
  stopifnot(all(tt > 0))
  list(ev = ev, tt = tt)
}
sm <- surv_parts(d_mv)
cx_main <- coxph(Surv(sm$tt, sm$ev) ~ circ + strata(land, t_year), data = d_mv)
cox_row(cx_main, "V2_HAZ_S1_first_private", "time_to_first_private_v2")
sf <- surv_parts(df)
cx_full <- coxph(Surv(sf$tt, sf$ev) ~ circ + strata(land, t_year), data = df)
cox_row(cx_full, "V2FULL_HAZ_first_private", "time_to_first_private_v2")
# S2 counterpart: like-for-like S2 hazard form for Chapter 5.3
cx_full_s2 <- coxph(as.formula(paste("Surv(sf$tt, sf$ev) ~ circ +", S2_RHS,
                                     "+ strata(land, t_year)")), data = df)
cox_row(cx_full_s2, "V2FULL_HAZ_S2_first_private", "time_to_first_private_v2",
        spec = "cox_S2_strata_landxcohort")
cx_sub <- coxph(Surv(sm$tt, sm$ev) ~ circ_subtype + strata(land, t_year), data = d_mv)
cox_row(cx_sub, "TIME_COX_subtypes", "time_to_first_private_v2",
        term = paste0("circ_subtype", c("Design", "Platform", "Nature",
                                        "Service", "Waste")))
# Cox with the S2 composition controls: the S2 counterpart of the incidence
# decomposition in hazard form; the S1-strata Cox above stays the headline
# time form
cx_main_s2 <- coxph(as.formula(paste("Surv(sm$tt, sm$ev) ~ circ +", S2_RHS,
                                     "+ strata(land, t_year)")), data = d_mv)
cox_row(cx_main_s2, "V2_HAZ_S2_first_private", "time_to_first_private_v2",
        spec = "cox_S2_strata_landxcohort")
cx_sub_s2 <- coxph(as.formula(paste("Surv(sm$tt, sm$ev) ~ circ_subtype +",
                                    S2_RHS, "+ strata(land, t_year)")), data = d_mv)
cox_row(cx_sub_s2, "TIME_COX_subtypes_S2", "time_to_first_private_v2",
        term = paste0("circ_subtype", c("Design", "Platform", "Nature",
                                        "Service", "Waste")),
        spec = "cox_S2_strata_landxcohort")

# ---- Schoenfeld PH diagnostics (appendix) ----
# Schoenfeld residual tests per term + global, following Kennedy et al. 2024
# ("Schoenfeld residuals were used to verify the proportionality of
# hazards"). Reading: a small p REJECTS a constant HR for that term, which is
# why the piecewise HRs below, not a single HR, are the reported time form.
zrows <- list()
zph_rows <- function(cx, run_id) {
  z <- cox.zph(cx)$table
  for (r in rownames(z)) {
    zrows[[length(zrows) + 1]] <<- data.frame(
      run_id = run_id, term = r, chisq = z[r, "chisq"], df = z[r, "df"],
      p = z[r, "p"], n = cx$n)
  }
}
zph_rows(cx_main, "V2_HAZ_S1_first_private")
zph_rows(cx_full, "V2FULL_HAZ_first_private")
zph_rows(cx_sub, "TIME_COX_subtypes")
zph_rows(cx_main_s2, "V2_HAZ_S2_first_private")  # the chapter cites the S2 test
zout <- do.call(rbind, zrows)
write.csv(zout, "outputs/estimates/timing_schoenfeld.csv", row.names = FALSE)

# ---- Piecewise-constant HRs (0-24 / 25-48 / 49+ months) ----
# Period-specific hazard ratios (appendix; the PH test rejects a constant HR):
# piecewise-constant circ effect over follow-up months 0-24 / 25-48 / 49+,
# following the piecewise specification of Fosfuri & Nagar 2026 (their Table 2,
# column (5)); baseline still stratified land x cohort. Direction check for
# the W24 > W36 > W48 incidence gradient (delay-dominant gap).
dpw <- data.frame(tt = sm$tt, ev = sm$ev, circ = d_mv$circ, land = d_mv$land,
                  t_year = d_mv$t_year)
dpw <- survSplit(Surv(tt, ev) ~ ., data = dpw, cut = c(24, 48), episode = "tgroup")
cx_pw <- coxph(Surv(tstart, tt, ev) ~ circ:strata(tgroup) + strata(land, t_year),
               data = dpw)
cs_pw <- summary(cx_pw)$coefficients
for (k in 1:3) {
  tm <- grep(paste0("tgroup=", k), rownames(cs_pw), value = TRUE)
  rows[[length(rows) + 1]] <- data.frame(
    run_id = paste0("V2_HAZ_PIECEWISE_", c("m0_24", "m25_48", "m49plus")[k]),
    outcome = "time_to_first_private_v2", spec = "cox_piecewise_strata_landxcohort",
    term = "circ_loghr", estimate = cs_pw[tm, "coef"], se = cs_pw[tm, "se(coef)"],
    p = cs_pw[tm, "Pr(>|z|)"], n = cx_main$n, vcov = "cox_model_based")
}
cat(sprintf("piecewise HR (circ): 0-24m %.3f | 25-48m %.3f | 49m+ %.3f\n",
            exp(cs_pw[1, "coef"]), exp(cs_pw[2, "coef"]), exp(cs_pw[3, "coef"])))

# Piecewise with the S2 composition controls: the ADJUSTED piecewise is the
# reported main form; the S1 piecewise above is the appendix variant.
dpw2 <- data.frame(tt = sm$tt, ev = sm$ev, circ = d_mv$circ, land = d_mv$land,
                   t_year = d_mv$t_year, founders_cat = d_mv$founders_cat,
                   nind_cat = d_mv$nind_cat, d_mv[, IND])
dpw2 <- survSplit(Surv(tt, ev) ~ ., data = dpw2, cut = c(24, 48), episode = "tgroup")
cx_pw2 <- coxph(as.formula(paste("Surv(tstart, tt, ev) ~ circ:strata(tgroup) +",
                                 S2_RHS, "+ strata(land, t_year)")), data = dpw2)
cs_pw2 <- summary(cx_pw2)$coefficients
for (k in 1:3) {
  tm <- grep(paste0("tgroup=", k), rownames(cs_pw2), value = TRUE)
  rows[[length(rows) + 1]] <- data.frame(
    run_id = paste0("V2_HAZ_PIECEWISE_S2_", c("m0_24", "m25_48", "m49plus")[k]),
    outcome = "time_to_first_private_v2", spec = "cox_piecewise_S2_strata_landxcohort",
    term = "circ_loghr", estimate = cs_pw2[tm, "coef"], se = cs_pw2[tm, "se(coef)"],
    p = cs_pw2[tm, "Pr(>|z|)"], n = cx_main$n, vcov = "cox_model_based")
}
cat(sprintf("piecewise HR S2 (circ): 0-24m %.3f | 25-48m %.3f | 49m+ %.3f\n",
            exp(cs_pw2[grep("tgroup=1", rownames(cs_pw2)), "coef"]),
            exp(cs_pw2[grep("tgroup=2", rownames(cs_pw2)), "coef"]),
            exp(cs_pw2[grep("tgroup=3", rownames(cs_pw2)), "coef"])))

# ---- Log-rank test (KM contrast) ----
lr <- survdiff(Surv(sm$tt, sm$ev) ~ circ, data = d_mv)
lr_p <- pchisq(lr$chisq, df = 1, lower.tail = FALSE)
rows[[length(rows) + 1]] <- data.frame(
  run_id = "V2_LOGRANK_first_private", outcome = "time_to_first_private_v2",
  spec = "logrank_unstratified", term = "circ_chisq", estimate = lr$chisq,
  se = NA, p = lr_p, n = nrow(d_mv), vcov = "logrank")
cat(sprintf("log-rank (main sample): chisq(1) = %.1f, p = %.3g\n", lr$chisq, lr_p))
cat("Schoenfeld (PH) tests -> timing_schoenfeld.csv:\n")
print(data.frame(run = zout$run_id, term = zout$term, chisq = round(zout$chisq, 2),
                 df = zout$df, p = signif(zout$p, 2)), row.names = FALSE)

# ---- Appendix 6: pre-specified robustness programme ----
# the pre-specified Section 3.6 robustness rows; additive, all prior run_ids
# unchanged.
# (a) country-clustered SEs for the main S2 pair (20 clusters)
grab(feols(as.formula(paste("fund36_private_v2 ~ circ +", S2_RHS, "| land^t_year")),
           d_mv, cluster = ~land),
     "V2_S2_CLAND", "fund36_private_v2", "S2_composition", "cluster_land")
cx_cl <- coxph(as.formula(paste("Surv(sm$tt, sm$ev) ~ circ +", S2_RHS,
                                "+ strata(land, t_year)")),
               data = d_mv, cluster = land)
cs_cl <- summary(cx_cl)$coefficients  # robust se column
rows[[length(rows) + 1]] <- data.frame(
  run_id = "V2_HAZ_S2_CLAND", outcome = "time_to_first_private_v2",
  spec = "cox_S2_strata_landxcohort", term = "circ_loghr",
  estimate = cs_cl["circ", "coef"], se = cs_cl["circ", "robust se"],
  p = cs_cl["circ", "Pr(>|z|)"], n = cx_cl$n, vcov = "cluster_land")

# (b) S2 hazard form for the high-confidence subsample (LPMs exist above)
sconf <- surv_parts(d_conf)
cx_conf <- coxph(as.formula(paste("Surv(sconf$tt, sconf$ev) ~ circ +", S2_RHS,
                                  "+ strata(land, t_year)")), data = d_conf)
cox_row(cx_conf, "V2_CONF_MV_HAZ_S2", "time_to_first_private_v2",
        spec = "cox_S2_strata_landxcohort_conf08")

# (c) random month within the founding year (full frame; the January
# convention replaced by a seeded uniform draw for the imputed firms;
# events falling before the drawn month are censored, mirroring the
# register-clock convention). Seed = the project sampling seed.
set.seed(20260711)
shift <- ifelse(df$founding_month_imputed_v2 == 1,
                sample(0:11, nrow(df), replace = TRUE), 0L)
dr <- df
dr$fidx_rand <- dr$founding_month_idx_v2 + shift
off_r <- dr$first_private_offset_m_v2 - shift
ev_r <- as.integer(!is.na(off_r) & off_r >= 0)
dr$fund36_rand <- as.integer(ev_r == 1 & off_r <= 35)
grab(feols(as.formula(paste("fund36_rand ~ circ +", S2_RHS, "| land^t_year")),
           dr, vcov = "hetero"),
     "V2_RANDMONTH_S2", "fund36_rand", "S2_random_month_full_frame", "HC1")
tt_r <- ifelse(ev_r == 1, off_r, CUTOFF_IDX - dr$fidx_rand) + 1
stopifnot(all(tt_r > 0))
cx_r <- coxph(as.formula(paste("Surv(tt_r, ev_r) ~ circ +", S2_RHS,
                               "+ strata(land, t_year)")), data = dr)
cox_row(cx_r, "V2_RANDMONTH_HAZ_S2", "time_to_first_private_rand",
        spec = "cox_S2_strata_landxcohort_randmonth")

# (d) yearly discrete-time model (cloglog), imputation-free: duration in
# CALENDAR years from the founding year to the first private round's
# calendar year; no month information enters. Last complete panel year
# = 2025 (cutoff 2026-03); later events censored. Person-year expansion.
ev_year <- (df$founding_month_idx_v2 + df$first_private_offset_m_v2) %/% 12
dur_ev <- ev_year - df$t_year
cens <- 2025 - df$t_year
has_ev <- !is.na(dur_ev) & dur_ev <= cens
dur <- ifelse(has_ev, dur_ev, cens)
py <- data.frame(idx = rep(seq_len(nrow(df)), dur + 1))
py$period <- unlist(lapply(dur + 1, seq_len)) - 1
py$ev <- as.integer(has_ev[py$idx] & py$period == dur[py$idx])
py <- cbind(py, df[py$idx, c("circ", "land", "t_year", "founders_cat",
                             "nind_cat", IND)])
py$period_cat <- factor(pmin(py$period, 6))
cat(sprintf("yearly discrete-time panel: %d firm-years (%d firms, %d events)\n",
            nrow(py), nrow(df), sum(py$ev)))
for (sp in c("S1", "S2")) {
  rhs <- if (sp == "S1") "circ + period_cat" else
    paste("circ + period_cat +", S2_RHS)
  my <- feglm(as.formula(paste("ev ~", rhs, "| land^t_year")), py,
              family = binomial(link = "cloglog"))
  ct <- coeftable(my)
  rows[[length(rows) + 1]] <- data.frame(
    run_id = paste0("V2_YEARLY_CLOGLOG_", sp), outcome = "first_private_yearly",
    spec = paste0(sp, "_discrete_time_cloglog_full_frame"), term = "circ_loghr",
    estimate = ct["circ", 1], se = ct["circ", 2], p = ct["circ", 4],
    n = my$nobs, vcov = "cloglog_model_based")
}
cat(sprintf("appendix runs: cland S2 %.2fpp | CONF HR %.3f | rand-month S2 %.2fpp, HR %.3f | yearly cloglog HR S1 %.3f S2 %.3f\n",
            100 * rows[[which(sapply(rows, function(r) r$run_id[1]) == "V2_S2_CLAND")]]$estimate,
            exp(summary(cx_conf)$coefficients["circ", "coef"]),
            100 * rows[[which(sapply(rows, function(r) r$run_id[1]) == "V2_RANDMONTH_S2")]]$estimate,
            exp(summary(cx_r)$coefficients["circ", "coef"]),
            exp(rows[[which(sapply(rows, function(r) r$run_id[1]) == "V2_YEARLY_CLOGLOG_S1")]]$estimate),
            exp(rows[[which(sapply(rows, function(r) r$run_id[1]) == "V2_YEARLY_CLOGLOG_S2")]]$estimate)))

# (e) cohort-specific S2 gaps (Appendix 7; within cohort => land FE only)
for (cy in 2016:2022) {
  dc <- d_mv[d_mv$t_year == cy, ]
  grab(feols(as.formula(paste("fund36_private_v2 ~ circ +", S2_RHS, "| land")),
             dc, vcov = "hetero"),
       paste0("V2_COHORT_S2_", cy), "fund36_private_v2",
       paste0("S2_within_cohort_", cy), "HC1")
}

# (f) censoring at register dissolution: 1,287 mv firms are
# dissolved in the register, 527 of them carry neither a Crunchbase closure
# nor a dead-website signal and would otherwise stay in the risk set until
# the cutoff. Variant: firms leave the risk set at their register dissolution
# month (verified matches, data/scrape/registry/registry_match_final.csv;
# min date per firm; the ~30 CB rounds recorded after a dissolution date are
# censored at the dissolution). Direction expected conservative: non-circular
# firms dissolve more often.
rmf <- "data/scrape/registry/registry_match_final.csv"
if (file.exists(rmf)) {
  rg <- read.csv(rmf)
  rg <- rg[!is.na(rg$dissolution_date) & rg$dissolution_date != "", ]
  dd <- as.Date(rg$dissolution_date, format = "%Y-%m-%d")
  dd[is.na(dd)] <- as.Date(rg$dissolution_date[is.na(dd)], format = "%d/%m/%Y")
  stopifnot(all(!is.na(dd)))
  rg$diss_midx <- as.integer(format(dd, "%Y")) * 12 +
    as.integer(format(dd, "%m")) - 1
  dmin <- aggregate(diss_midx ~ firm_id, rg, min)
  d_diss <- merge(d_mv, dmin, by = "firm_id", all.x = TRUE)
  diss_t <- pmin(ifelse(is.na(d_diss$diss_midx), CUTOFF_IDX, d_diss$diss_midx),
                 CUTOFF_IDX) - d_diss$founding_month_idx_v2 + 1
  base_ev <- as.integer(!is.na(d_diss$first_private_offset_m_v2))
  base_t <- ifelse(base_ev == 1, d_diss$first_private_offset_m_v2 + 1,
                   CUTOFF_IDX - d_diss$founding_month_idx_v2 + 1)
  ev_d <- as.integer(base_ev == 1 & base_t <= diss_t)
  tt_d <- pmin(base_t, pmax(diss_t, 1))
  cat(sprintf("register-dissolution censoring: %d mv firms dissolved, %d events after dissolution censored\n",
              sum(!is.na(d_diss$diss_midx)), sum(base_ev == 1 & ev_d == 0)))
  cx_d <- coxph(as.formula(paste("Surv(tt_d, ev_d) ~ circ +", S2_RHS,
                                 "+ strata(land, t_year)")), data = d_diss)
  cox_row(cx_d, "V2_REGDISS_CENSOR_HAZ_S2", "time_to_first_private_v2",
          spec = "cox_S2_strata_landxcohort_regdiss_censor")
  cat(sprintf("  regdiss-censored HR S2 = %.3f (p = %.2g)\n",
              exp(summary(cx_d)$coefficients["circ", "coef"]),
              summary(cx_d)$coefficients["circ", "Pr(>|z|)"]))
}

# ---- Kaplan-Meier curve data (source of fig2_time_to_private) ----
# sample 'main' = month-verified (the fig2 source), 'full' = incl.
# January-imputed (frame-comparison companion, appendix layer).
# KM at month 36 must equal the raw v2 incidence rates, the anchor.
km_rows <- list()
for (samp in c("main", "full")) {
  ds <- if (samp == "main") d_mv else df
  sp <- surv_parts(ds)
  km <- survfit(Surv(sp$tt, sp$ev) ~ circ, data = ds)
  km_rows[[samp]] <- data.frame(sample = samp,
                                group = rep(names(km$strata), km$strata),
                                time_m = km$time, surv = km$surv,
                                lower = km$lower, upper = km$upper,
                                n_risk = km$n.risk)
}
kmp <- do.call(rbind, km_rows)
write.csv(kmp, "outputs/estimates/timing_km_private.csv", row.names = FALSE)
for (cc in 0:1) {  # anchor: 1 - S(36) == mean(fund36_private_v2) per group (main)
  g <- kmp[kmp$sample == "main" & kmp$group == paste0("circ=", cc)
           & kmp$time_m <= 36, ]
  km36 <- 1 - tail(g$surv, 1)
  raw <- mean(d_mv$fund36_private_v2[d_mv$circ == cc])
  stopifnot(abs(km36 - raw) < 0.005)
}

# ---- Write estimates + headline ----
out <- do.call(rbind, rows)
dir.create("outputs/estimates", showWarnings = FALSE, recursive = TRUE)
write.csv(out, "outputs/estimates/timing_estimates.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/estimates/timing_sessioninfo.txt")
hr <- exp(out[out$run_id == "V2_HAZ_S1_first_private", "estimate"])
hr2 <- exp(out[out$run_id == "V2_HAZ_S2_first_private", "estimate"])
s2 <- 100 * out[out$run_id == "V2_S2_fund36_private_v2", "estimate"]
cat(sprintf("wrote timing_estimates.csv (%d rows) + timing_km_private.csv\n", nrow(out)))
cat(sprintf("headline (month-verified sample, n=%d): incidence gap S2 @36m = %.2f pp | Cox HR S1 = %.3f | S2 = %.3f\n",
            nrow(d_mv), s2, hr, hr2))
