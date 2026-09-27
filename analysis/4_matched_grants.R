# 4_matched_grants.R: matched grant comparison, set-FE LPMs, mechanism pack,
# blended-finance split, post-grant KM and the TRANS transition Cox.
#
# Design: matched comparison with pseudo award dates; sample from
# scripts/build_grants_sample.py (seed 20260711; matched on cohort x circ x
# land x pre-grant funding history incl. recent-12m momentum).
#   M0  difference in means (cluster: matched set)
#   M1  within matched set (set FE)                      <- main
#   M2  M1 + founding_month_imputed + n_founders          (residual-imbalance guard)
# Main outcome: any private round within 24m of (pseudo-)grant signature.
# Splits: EIC blended finance (-BF) in/out; consortium-R&D as separate treatment;
# treated x circ interaction. Placebo: pre-grant round count.
# Deterministic (no RNG here; the sampling seed lives in build_grants_sample.py).
#
# Inputs:  data/processed/grants_matched.csv     (v1 layer; pipeline step 5)
#          data/processed/grants_matched_mv.csv  (month-verified sample, the
#            reported treatment definition; the V2 block is skipped if this
#            file is absent)
# Outputs: outputs/estimates/grant_estimates.csv, grant_schoenfeld.csv,
#          grants_km_postgrant.csv, grants_km_bridge.csv,
#          grants_fourgroup_km.csv (+ grant_sessioninfo.txt)
# Run:     from the repo root, after build_grants_sample.py (run_pipeline.sh
#          step 6).
#
# Circular Startup Financing. Master's thesis, 2026.
###############################################################################
library(fixest)

# ---- Load v1 sample ----
if (!file.exists("data/processed/grants_matched.csv"))
  stop("Run this script from the repository root (see run_pipeline.sh).")
m <- read.csv("data/processed/grants_matched.csv")  # built by pipeline step 5
if (Sys.getenv("REBASE") != "1") {  # REBASE=1 lifts pins (new data export, see scripts/sanity_checks.py)
  # Two documented pin updates, no silent loosening:
  # 1. Musa <-> MUSA scarl rejected as a wrong entity (append-only verdict):
  #    one consortium set fewer; rows 5953 to 5948, consortium-treated 540 to 539.
  # 2. The DE4.csv relabel (DE to CH) plus matcher extensions and the
  #    short-name review lift treated firms 1,553 to 1,661; sample 5948 to
  #    6410 rows, sme_eic 683 to 732, consortium 539 to 583 (seed unchanged).
  stopifnot(nrow(m) == 6410,
            sum(m$treated == 1 & m$group == "sme_eic") == 732,
            sum(m$treated == 1 & m$group == "consortium") == 583)
}

m$founders_fill <- ifelse(is.na(m$n_founders), median(m$n_founders, na.rm = TRUE),
                          m$n_founders)
m$founders_miss <- as.integer(is.na(m$n_founders))

rows <- list()
grab <- function(mod, run_id, group, outcome, spec, terms = "treated") {
  ct <- coeftable(mod)
  for (tm in intersect(terms, rownames(ct))) {
    rows[[length(rows) + 1]] <<- data.frame(
      run_id = run_id, group = group, outcome = outcome, spec = spec, term = tm,
      estimate = ct[tm, 1], se = ct[tm, 2], p = ct[tm, 4], n = mod$nobs,
      n_sets = length(unique(mod$fixef_id$set_id %||% NA)))
  }
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- V1 layer: M0/M1/M2 + placebo per treatment group ----
# the v1 January-imputation sample; the reported numbers live in the V2 block
# below and v1 stays as the robustness layer
OUTS <- c("post24_any_private", "log_post24_usd", "post24_n_private")

for (g in c("sme_eic", "consortium")) {
  s <- m[m$group == g, ]
  grab(feols(post24_any_private ~ treated, s, cluster = ~set_id),
       paste0(g, "_M0"), g, "post24_any_private", "M0_means")
  for (y in OUTS) {
    grab(feols(as.formula(paste(y, "~ treated | set_id")), s, cluster = ~set_id),
         paste0(g, "_M1_", y), g, y, "M1_set_FE")
  }
  grab(feols(post24_any_private ~ treated + founding_month_imputed + founders_fill +
               founders_miss | set_id, s, cluster = ~set_id),
       paste0(g, "_M2"), g, "post24_any_private", "M2_plus_covariates")
  grab(feols(pre_n_private ~ treated | set_id, s, cluster = ~set_id),
       paste0(g, "_placebo"), g, "pre_n_private", "PLACEBO_pre_rounds")
}

# ---- Blended-finance split (v1) ----
# the EIC-Fund equity component looks like a VC round; main group only
s <- m[m$group == "sme_eic", ]
grab(feols(post24_any_private ~ treated | set_id, s[s$bf_set == 0, ], cluster = ~set_id),
     "sme_eic_M1_noBF", "sme_eic", "post24_any_private", "M1_grant_only_sets")
grab(feols(post24_any_private ~ treated | set_id, s[s$bf_set == 1, ], cluster = ~set_id),
     "sme_eic_M1_BF", "sme_eic", "post24_any_private", "M1_blended_finance_sets")

# ---- SE robustness: two-way clustering (set + firm) ----
# two-way clustered SEs: control firms can serve in multiple matched sets and
# set-only clustering ignores that correlation. Robustness on the three
# headline ATTs; reported as one appendix footnote in the thesis.
grab(feols(post24_any_private ~ treated | set_id, s, cluster = ~set_id + firm_id),
     "sme_eic_M1_tw", "sme_eic", "post24_any_private", "M1_twoway_set_firm")
grab(feols(post24_any_private ~ treated | set_id, s[s$bf_set == 0, ],
           cluster = ~set_id + firm_id),
     "sme_eic_M1_noBF_tw", "sme_eic", "post24_any_private", "M1_grant_only_twoway_set_firm")
grab(feols(post24_any_private ~ treated | set_id, m[m$group == "consortium", ],
           cluster = ~set_id + firm_id),
     "consortium_M1_tw", "consortium", "post24_any_private", "M1_twoway_set_firm")

# treated x circ (210 treated circular firms, MDE ~12-14pp)
grab(feols(post24_any_private ~ treated + treated:circ | set_id, s, cluster = ~set_id),
     "sme_eic_M1_circint", "sme_eic", "post24_any_private", "M1_circ_interaction",
     terms = c("treated", "treated:circ"))

# ---- Runway-mechanism pack (v1, grant-only sets) ----
# runs on the grant-only sets, where the negative effect lives.
# Predictions if "grant buys runway, round comes later": small grants ~0 / large negative;
# dip early (months 0-11) fading late (12-23); catch-up at 36/48m; no extra grant rounds.
go <- s[s$bf_set == 0, ]
grab(feols(post24_any_private ~ treated | set_id, go[go$size_class == "small", ],
           cluster = ~set_id),
     "run_dose_small", "sme_eic", "post24_any_private", "RUN_dose_small_50k")
grab(feols(post24_any_private ~ treated | set_id, go[go$size_class == "large", ],
           cluster = ~set_id),
     "run_dose_large", "sme_eic", "post24_any_private", "RUN_dose_large_0.5-2.5M")
grab(feols(post12_any_private ~ treated | set_id, go, cluster = ~set_id),
     "run_timing_0_11", "sme_eic", "post12_any_private", "RUN_timing_months_0-11")
grab(feols(post13_24_any_private ~ treated | set_id, go, cluster = ~set_id),
     "run_timing_12_23", "sme_eic", "post13_24_any_private", "RUN_timing_months_12-23")
grab(feols(post36_any_private ~ treated | set_id, go[go$win36_complete == 1, ],
           cluster = ~set_id),
     "run_catchup_36m", "sme_eic", "post36_any_private", "RUN_catchup_36m")
grab(feols(post48_any_private ~ treated | set_id, go[go$win48_complete == 1, ],
           cluster = ~set_id),
     "run_catchup_48m", "sme_eic", "post48_any_private", "RUN_catchup_48m")
grab(feols(post24_n_grant_months ~ treated | set_id, go, cluster = ~set_id),
     "run_treadmill", "sme_eic", "post24_n_grant_months", "RUN_treadmill_grant_rounds")
grab(feols(post24_n_grant_months_ex ~ treated | set_id, go, cluster = ~set_id),
     "run_treadmill_ex", "sme_eic", "post24_n_grant_months_ex",
     "RUN_treadmill_excl_own_award_window")

cat("\nrunway pack set counts: grant-only", length(unique(go$set_id)),
    "| small", length(unique(go$set_id[go$size_class == "small"])),
    "| large", length(unique(go$set_id[go$size_class == "large"])),
    "| 36m-complete", length(unique(go$set_id[go$win36_complete == 1 & go$treated == 1])),
    "| 48m-complete", length(unique(go$set_id[go$win48_complete == 1 & go$treated == 1])), "\n")

# ---- Success margins (v1) ----
# success-relevant margins on the post-grant window.
# post24_success = the success composite (≥3 private rounds ∨ >$10M month ∨ M&A);
# components reported separately so "success" is not just the rounds result
# relabeled; investor accumulation as an additional margin.
for (dset in list(list("go", go), list("go_small", go[go$size_class == "small", ]),
                  list("go_large", go[go$size_class == "large", ]))) {
  nm <- dset[[1]]; d <- dset[[2]]
  grab(feols(post24_success ~ treated | set_id, d, cluster = ~set_id),
       paste0("bridge_success_", nm), "sme_eic", "post24_success",
       paste0("BRIDGE_success_", nm))
  grab(feols(log_post24_investors ~ treated | set_id, d, cluster = ~set_id),
       paste0("bridge_loginv_", nm), "sme_eic", "log_post24_investors",
       paste0("BRIDGE_log_investors_", nm))
}
grab(feols(post24_any_multi_inv ~ treated | set_id, go, cluster = ~set_id),
     "bridge_multiinv_go", "sme_eic", "post24_any_multi_inv", "BRIDGE_any_multi_investor")
grab(feols(post24_ma ~ treated | set_id, go, cluster = ~set_id),
     "bridge_ma_go", "sme_eic", "post24_ma", "BRIDGE_component_MA")
grab(feols(post24_success ~ treated | set_id, m[m$group == "consortium", ],
           cluster = ~set_id),
     "bridge_success_consortium", "consortium", "post24_success", "BRIDGE_success_consortium")

# ---- Post-grant KM curve data (v1; overwritten by the V2 block below) ----
# KM of months from (pseudo-)grant month
# to the first subsequent private round, treated vs matched controls, GRANT-ONLY
# sets (the -BF equity component would mechanically look private). Censoring at
# panel end 2026-03; time convention offset+1 (offset 0 = round in the grant
# month itself, same >= g convention as post24_any_private; KM at time 24 must
# reproduce the 40.4/46.0% rates). The treated-control deficit grows
# monotonically (~-5.4pp @24m, ~-10.9pp @36m), the grant-track pattern in
# time form.
library(survival)
CUTOFF_IDX <- 2026 * 12 + 2
ev6 <- as.integer(!is.na(go$post_first_private_offset_m))
t6 <- ifelse(ev6 == 1, go$post_first_private_offset_m, CUTOFF_IDX - go$g_midx) + 1
stopifnot(all(t6 > 0))
km6 <- survfit(Surv(t6, ev6) ~ treated, data = go)
km6d <- data.frame(group = rep(names(km6$strata), km6$strata),
                   time_m = km6$time, surv = km6$surv,
                   lower = km6$lower, upper = km6$upper, n_risk = km6$n.risk)
write.csv(km6d, "outputs/estimates/grants_km_postgrant.csv", row.names = FALSE)
cat(sprintf("wrote grants_km_postgrant.csv (%d rows)\n", nrow(km6d)))

# ---- TRANS: the matched comparison in hazard form ----
# Stratified Cox by matched set: treated = log HR of transitioning to private
# capital after the (pseudo-)grant. Same identification as M1, expressed in
# time. Templates: Croce et al. 2013 and Kennedy et al. 2024.
zrows <- list()
trans_row <- function(d, run_id, grp) {
  ev <- as.integer(!is.na(d$post_first_private_offset_m))
  tt <- ifelse(ev == 1, d$post_first_private_offset_m,
               CUTOFF_IDX - d$g_midx) + 1
  cx <- coxph(Surv(tt, ev) ~ treated + strata(set_id), data = d)
  cs <- summary(cx)$coefficients
  rows[[length(rows) + 1]] <<- data.frame(
    run_id = run_id, group = grp, outcome = "time_to_private_post_grant",
    spec = "TRANS_cox_strata_set", term = "treated_loghr",
    estimate = cs["treated", "coef"], se = cs["treated", "se(coef)"],
    p = cs["treated", "Pr(>|z|)"], n = cx$n, n_sets = length(unique(d$set_id)))
  # Schoenfeld PH test (appendix, following Kennedy et al. 2024)
  z <- cox.zph(cx)$table
  for (r in rownames(z)) {
    zrows[[length(zrows) + 1]] <<- data.frame(
      run_id = run_id, term = r, chisq = z[r, "chisq"], df = z[r, "df"],
      p = z[r, "p"], n = cx$n)
  }
  cat(sprintf("  TRANS %-22s HR = %.3f  (p = %.2g, n = %d)\n",
              run_id, exp(cs["treated", "coef"]), cs["treated", "Pr(>|z|)"], cx$n))
}
trans_row(go, "TRANS_grant_only", "sme_eic_grant_only")
trans_row(s[s$bf_set == 1, ], "TRANS_bf", "sme_eic_bf")
trans_row(m[m$group == "consortium", ], "TRANS_consortium", "consortium")

# ---- V2 block: month-verified sample = the REPORTED treatment ----
# Both set sides are restricted to month-verified founding dates
# (build_grants_sample.py
# --fidx founding_month_idx_v2 --mv-only --out grants_matched_mv.csv). The
# January imputation is measurement error in the time axis (pseudo-award =
# founding + treated age), which a covariate cannot fix; one sample for both
# RQs. The earlier full-frame v2 sample (grants_matched_v2.csv, 550 E4 sets,
# ATT -5.9*/M2 -8.6***/TRANS 0.684***) stays on disk as the named robustness
# layer. founders remain the only M2 covariates (founding_month_imputed is
# constant 0 here).
if (file.exists("data/processed/grants_matched_mv.csv")) {
  d_mv <- read.csv("data/processed/grants_matched_mv.csv")
  cat(sprintf("\nV2 matched sample: %d rows; sme_eic treated %d, consortium %d\n",
              nrow(d_mv), sum(d_mv$treated == 1 & d_mv$group == "sme_eic"),
              sum(d_mv$treated == 1 & d_mv$group == "consortium")))

  # eligibility filter: UK left Horizon/EIC eligibility with Brexit and CH
  # lost association in 2021, so their post-2020 controls are "never granted
  # by construction". Sets whose (pseudo-)grant date falls in an ineligible
  # land x period are excluded from ALL V2 runs.
  tl2 <- d_mv[d_mv$treated == 1, c("set_id", "land", "g_midx")]
  tl2$set_gyear <- tl2$g_midx %/% 12
  names(tl2)[2] <- "set_land"
  d_mv <- merge(d_mv, tl2[, c("set_id", "set_land", "set_gyear")], by = "set_id")
  inelig <- (d_mv$set_land %in% c("UK", "CH")) & (d_mv$set_gyear >= 2021)
  cat(sprintf("  E4 eligibility filter: %d sets excluded (UK/CH with grant year >= 2021)\n",
              length(unique(d_mv$set_id[inelig]))))
  d_mv <- d_mv[!inelig, ]
  # treatment definition: competitively awarded, firm-targeted PURE grants;
  # the grant-only runs below are the reported treatment and the pooled, BF
  # and consortium rows stay computed as appendix/audit layers.
  for (g in c("sme_eic", "consortium")) {
    d_sme <- d_mv[d_mv$group == g, ]
    grab(feols(post24_any_private ~ treated | set_id, d_sme, cluster = ~set_id),
         paste0("V2_M1_", g), g, "post24_any_private", "set_FE")
    grab(feols(pre_n_private ~ treated | set_id, d_sme, cluster = ~set_id),
         paste0("V2_placebo_", g), g, "pre_n_private", "set_FE")
  }
  d_sme <- d_mv[d_mv$group == "sme_eic", ]
  grab(feols(post24_any_private ~ treated | set_id, d_sme[d_sme$bf_set == 0, ],
             cluster = ~set_id), "V2_M1_grant_only", "sme_eic_grant_only",
       "post24_any_private", "set_FE")
  grab(feols(post24_any_private ~ treated | set_id, d_sme[d_sme$bf_set == 1, ],
             cluster = ~set_id), "V2_M1_bf", "sme_eic_bf",
       "post24_any_private", "set_FE")

  # v2 = the headline numbers: mechanism pack, TRANS and the post-grant KM
  # curve run on the v2 sample; the v1 runs above stay as the
  # January-imputation robustness layer. Mirrors the v1 specifications 1:1.
  d_mv$founders_fill <- ifelse(is.na(d_mv$n_founders),
                             median(d_mv$n_founders, na.rm = TRUE), d_mv$n_founders)
  d_mv$founders_miss <- as.integer(is.na(d_mv$n_founders))
  d_sme <- d_mv[d_mv$group == "sme_eic", ]
  d_pure <- d_sme[d_sme$bf_set == 0, ]
  grab(feols(post24_any_private ~ treated + founders_fill +
               founders_miss | set_id, d_sme, cluster = ~set_id),
       "V2_M2_sme_eic", "sme_eic", "post24_any_private", "M2_plus_covariates")
  # M2 guard on the MAIN treatment definition (grant-only); in the mv-only
  # sample the founders are the only M2 covariates
  grab(feols(post24_any_private ~ treated + founders_fill +
               founders_miss | set_id, d_pure, cluster = ~set_id),
       "V2_M2_grant_only", "grant_only", "post24_any_private", "M2_plus_covariates")
  grab(feols(post24_any_private ~ treated | set_id, d_sme,
             cluster = ~set_id + firm_id),
       "V2_M1_tw", "sme_eic", "post24_any_private", "M1_twoway_set_firm")
  grab(feols(post24_any_private ~ treated + treated:circ | set_id, d_sme,
             cluster = ~set_id),
       "V2_M1_circint", "sme_eic", "post24_any_private", "M1_circ_interaction",
       terms = c("treated", "treated:circ"))
  # circ interaction on the PURE-grant sets: the reported interaction matches
  # the reported treatment definition; the pooled 600-set run above stays as
  # the robustness layer
  grab(feols(post24_any_private ~ treated + treated:circ | set_id, d_pure,
             cluster = ~set_id),
       "V2_M1_circint_pure", "sme_eic_grant_only", "post24_any_private",
       "M1_circ_interaction_pure", terms = c("treated", "treated:circ"))
  grab(feols(post24_any_private ~ treated | set_id,
             d_pure[d_pure$size_class == "small", ], cluster = ~set_id),
       "V2_run_dose_small", "sme_eic", "post24_any_private", "RUN_dose_small_50k")
  grab(feols(post24_any_private ~ treated | set_id,
             d_pure[d_pure$size_class == "large", ], cluster = ~set_id),
       "V2_run_dose_large", "sme_eic", "post24_any_private", "RUN_dose_large_0.5-2.5M")
  grab(feols(post12_any_private ~ treated | set_id, d_pure, cluster = ~set_id),
       "V2_run_timing_0_11", "sme_eic", "post12_any_private", "RUN_timing_months_0-11")
  grab(feols(post13_24_any_private ~ treated | set_id, d_pure, cluster = ~set_id),
       "V2_run_timing_12_23", "sme_eic", "post13_24_any_private", "RUN_timing_months_12-23")
  grab(feols(post36_any_private ~ treated | set_id, d_pure[d_pure$win36_complete == 1, ],
             cluster = ~set_id),
       "V2_run_catchup_36m", "sme_eic", "post36_any_private", "RUN_catchup_36m")
  grab(feols(post48_any_private ~ treated | set_id, d_pure[d_pure$win48_complete == 1, ],
             cluster = ~set_id),
       "V2_run_catchup_48m", "sme_eic", "post48_any_private", "RUN_catchup_48m")
  grab(feols(post24_n_grant_months ~ treated | set_id, d_pure, cluster = ~set_id),
       "V2_run_treadmill", "sme_eic", "post24_n_grant_months",
       "RUN_treadmill_grant_rounds")
  grab(feols(post24_n_grant_months_ex ~ treated | set_id, d_pure, cluster = ~set_id),
       "V2_run_treadmill_ex", "sme_eic", "post24_n_grant_months_ex",
       "RUN_treadmill_excl_own_award_window")
  trans_row(d_pure, "V2_TRANS_grant_only", "sme_eic_grant_only")
  trans_row(d_sme[d_sme$bf_set == 1, ], "V2_TRANS_bf", "sme_eic_bf")
  trans_row(d_mv[d_mv$group == "consortium", ], "V2_TRANS_consortium", "consortium")
  # TRANS with the two M2 covariates: the transition Cox guarded against the
  # residual imbalances, symmetric to V2_M2_grant_only; same identification,
  # expressed in time.
  ev_m2 <- as.integer(!is.na(d_pure$post_first_private_offset_m))
  tt_m2 <- ifelse(ev_m2 == 1, d_pure$post_first_private_offset_m,
                  CUTOFF_IDX - d_pure$g_midx) + 1
  cx_m2 <- coxph(Surv(tt_m2, ev_m2) ~ treated + founders_fill +
                   founders_miss + strata(set_id), data = d_pure)
  cs_m2 <- summary(cx_m2)$coefficients
  rows[[length(rows) + 1]] <- data.frame(
    run_id = "V2_TRANS_M2_grant_only", group = "sme_eic_grant_only",
    outcome = "time_to_private_post_grant", spec = "TRANS_cox_M2_strata_set",
    term = "treated_loghr", estimate = cs_m2["treated", "coef"],
    se = cs_m2["treated", "se(coef)"], p = cs_m2["treated", "Pr(>|z|)"],
    n = cx_m2$n, n_sets = length(unique(d_pure$set_id)))
  cat(sprintf("  TRANS M2 (+covariates)  HR = %.3f  (p = %.2g, n = %d)\n",
              exp(cs_m2["treated", "coef"]), cs_m2["treated", "Pr(>|z|)"], cx_m2$n))
  # PH test for the covariate-adjusted TRANS (the chapter cites this model)
  zm2 <- cox.zph(cx_m2)$table
  for (r in rownames(zm2)) {
    zrows[[length(zrows) + 1]] <- data.frame(
      run_id = "V2_TRANS_M2_grant_only", term = r, chisq = zm2[r, "chisq"],
      df = zm2[r, "df"], p = zm2[r, "p"], n = cx_m2$n)
  }
  cat(sprintf("  TRANS M2 zph (treated): chisq = %.1f, p = %.2g\n",
              zm2["treated", "chisq"], zm2["treated", "p"]))
  # Bridge split: the any-private-before-award margin is an EXACT matching
  # cell, so sets are homogeneous in it and splitting them breaks nothing.
  # no-prior = the first-entry ("bridge") question; prior = the acceleration
  # question. Homogeneity asserted.
  d_pure$pre_any <- as.integer(d_pure$pre_n_private > 0)
  stopifnot(all(tapply(d_pure$pre_any, d_pure$set_id, function(x) length(unique(x))) == 1))
  for (bs in c("noprior", "prior")) {
    d_b <- d_pure[d_pure$pre_any == as.integer(bs == "prior"), ]
    cat(sprintf("  BRIDGE %s: %d sets\n", bs, length(unique(d_b$set_id))))
    grab(feols(post24_any_private ~ treated | set_id, d_b, cluster = ~set_id),
         paste0("V2_run_bridge_", bs), "sme_eic_grant_only", "post24_any_private",
         paste0("RUN_bridge_", bs, "_pre_award_private"))
    trans_row(d_b, paste0("V2_TRANS_bridge_", bs), "sme_eic_grant_only")
  }
  # Bridge KM curves per subgroup (Figure 4: KM by pre-award status)
  km_b <- list()
  for (bs in c("noprior", "prior")) {
    d_b <- d_pure[d_pure$pre_any == as.integer(bs == "prior"), ]
    ev_b <- as.integer(!is.na(d_b$post_first_private_offset_m))
    tt_b <- ifelse(ev_b == 1, d_b$post_first_private_offset_m,
                   CUTOFF_IDX - d_b$g_midx) + 1
    kb <- survfit(Surv(tt_b, ev_b) ~ treated, data = d_b)
    km_b[[bs]] <- data.frame(split = bs,
                             group = rep(names(kb$strata), kb$strata),
                             time_m = kb$time, surv = kb$surv,
                             lower = kb$lower, upper = kb$upper,
                             n_risk = kb$n.risk)
  }
  write.csv(do.call(rbind, km_b), "outputs/estimates/grants_km_bridge.csv",
            row.names = FALSE)
  cat(sprintf("wrote grants_km_bridge.csv (%d rows)\n",
              nrow(do.call(rbind, km_b))))
  # Formal subgroup-difference test for the bridge split (same logic as the
  # dose test below: sig-in-A + ns-in-B is not a difference test)
  grab(feols(post24_any_private ~ treated + treated:pre_any | set_id, d_pure,
             cluster = ~set_id),
       "V2_run_bridge_diff", "sme_eic_grant_only", "post24_any_private",
       "RUN_bridge_difference_test", terms = c("treated", "treated:pre_any"))
  # Dose-difference test: a formal test of small vs large treatment effects,
  # treated:small on all pure-grant sets; the size main effect is absorbed by
  # the set FE (size_class is a set attribute). Guards the "concentrated in
  # small awards" wording.
  d_pure$small <- as.integer(d_pure$size_class == "small")
  grab(feols(post24_any_private ~ treated + treated:small | set_id, d_pure,
             cluster = ~set_id),
       "V2_run_dose_diff", "sme_eic_grant_only", "post24_any_private",
       "RUN_dose_difference_test", terms = c("treated", "treated:small"))
  # Period-specific transition HRs (appendix; PH test rejects a constant HR for
  # the pure-grant TRANS): piecewise-constant treated effect over months 0-12 /
  # 13-24 / 25+ after the (pseudo-)award, following Fosfuri & Nagar 2026.
  ev_pw <- as.integer(!is.na(d_pure$post_first_private_offset_m))
  tt_pw <- ifelse(ev_pw == 1, d_pure$post_first_private_offset_m, CUTOFF_IDX - d_pure$g_midx) + 1
  dpw <- data.frame(tt = tt_pw, ev = ev_pw, treated = d_pure$treated, set_id = d_pure$set_id)
  dpw <- survSplit(Surv(tt, ev) ~ ., data = dpw, cut = c(12, 24), episode = "tgroup")
  cx_pw <- coxph(Surv(tstart, tt, ev) ~ treated:strata(tgroup) + strata(set_id), data = dpw)
  cs_pw <- summary(cx_pw)$coefficients
  for (k in 1:3) {
    tm <- grep(paste0("tgroup=", k), rownames(cs_pw), value = TRUE)
    rows[[length(rows) + 1]] <- data.frame(
      run_id = paste0("V2_TRANS_PIECEWISE_", c("m0_12", "m13_24", "m25plus")[k]),
      group = "sme_eic_grant_only", outcome = "time_to_private_post_grant",
      spec = "TRANS_cox_piecewise_strata_set", term = "treated_loghr",
      estimate = cs_pw[tm, "coef"], se = cs_pw[tm, "se(coef)"],
      p = cs_pw[tm, "Pr(>|z|)"], n = cx_pw$n, n_sets = length(unique(d_pure$set_id)))
  }
  cat(sprintf("piecewise TRANS HR (pure grants): 0-12m %.3f | 13-24m %.3f | 25m+ %.3f\n",
              exp(cs_pw[1, "coef"]), exp(cs_pw[2, "coef"]), exp(cs_pw[3, "coef"])))
  # Piecewise TRANS with the M2 covariates: the covariate-adjusted piecewise
  # is the reported main form; the unadjusted piecewise above is the appendix
  # variant.
  dpwm <- data.frame(tt = tt_pw, ev = ev_pw, treated = d_pure$treated,
                     set_id = d_pure$set_id,
                     founders_fill = d_pure$founders_fill,
                     founders_miss = d_pure$founders_miss)
  dpwm <- survSplit(Surv(tt, ev) ~ ., data = dpwm, cut = c(12, 24), episode = "tgroup")
  cx_pwm <- coxph(Surv(tstart, tt, ev) ~ treated:strata(tgroup) +
                    founders_fill + founders_miss +
                    strata(set_id), data = dpwm)
  cs_pwm <- summary(cx_pwm)$coefficients
  for (k in 1:3) {
    tm <- grep(paste0("tgroup=", k), rownames(cs_pwm), value = TRUE)
    rows[[length(rows) + 1]] <- data.frame(
      run_id = paste0("V2_TRANS_PIECEWISE_M2_", c("m0_12", "m13_24", "m25plus")[k]),
      group = "sme_eic_grant_only", outcome = "time_to_private_post_grant",
      spec = "TRANS_cox_piecewise_M2_strata_set", term = "treated_loghr",
      estimate = cs_pwm[tm, "coef"], se = cs_pwm[tm, "se(coef)"],
      p = cs_pwm[tm, "Pr(>|z|)"], n = cx_pwm$n, n_sets = length(unique(d_pure$set_id)))
  }
  cat(sprintf("piecewise TRANS HR M2: 0-12m %.3f | 13-24m %.3f | 25m+ %.3f\n",
              exp(cs_pwm[grep("tgroup=1", rownames(cs_pwm)), "coef"]),
              exp(cs_pwm[grep("tgroup=2", rownames(cs_pwm)), "coef"]),
              exp(cs_pwm[grep("tgroup=3", rownames(cs_pwm)), "coef"])))
  # Set-stratified log-rank
  lr2 <- survdiff(Surv(tt_pw, ev_pw) ~ treated + strata(set_id), data = d_pure)
  lr2_p <- pchisq(lr2$chisq, df = 1, lower.tail = FALSE)
  rows[[length(rows) + 1]] <- data.frame(
    run_id = "V2_LOGRANK_trans", group = "sme_eic_grant_only",
    outcome = "time_to_private_post_grant", spec = "logrank_strata_set",
    term = "treated_chisq", estimate = lr2$chisq, se = NA, p = lr2_p,
    n = length(ev_pw), n_sets = length(unique(d_pure$set_id)))
  cat(sprintf("log-rank (strata=set): chisq(1) = %.1f, p = %.3g\n", lr2$chisq, lr2_p))
  # M2 mirrors of the horizon LPMs (table 4.3 Panel A: each horizon with
  # and without the founder covariates)
  grab(feols(post12_any_private ~ treated + founders_fill +
               founders_miss | set_id, d_pure, cluster = ~set_id),
       "V2_run_timing_0_11_M2", "sme_eic", "post12_any_private",
       "RUN_timing_months_0-11_M2")
  grab(feols(post36_any_private ~ treated + founders_fill +
               founders_miss | set_id, d_pure[d_pure$win36_complete == 1, ],
             cluster = ~set_id),
       "V2_run_catchup_36m_M2", "sme_eic", "post36_any_private", "RUN_catchup_36m_M2")
  grab(feols(post48_any_private ~ treated + founders_fill +
               founders_miss | set_id, d_pure[d_pure$win48_complete == 1, ],
             cluster = ~set_id),
       "V2_run_catchup_48m_M2", "sme_eic", "post48_any_private", "RUN_catchup_48m_M2")
  # Any further grant within 24m (share version of the treadmill outcome,
  # table 4.4 Panel B; a 36m variant would need a sample rebuild, so 24m only)
  d_pure$any24_grant_ex <- as.integer(d_pure$post24_n_grant_months_ex > 0)
  grab(feols(any24_grant_ex ~ treated | set_id, d_pure, cluster = ~set_id),
       "V2_run_anygrant24_ex", "sme_eic", "any24_grant_ex",
       "RUN_any_further_grant_24m")
  cat(sprintf("any further grant within 24m: treated %.3f vs control %.3f\n",
              mean(d_pure$any24_grant_ex[d_pure$treated == 1]),
              mean(d_pure$any24_grant_ex[d_pure$treated == 0])))
  # ERA split (appendix), the analog to the Cleantech 1.0/2.0 period split of
  # Kennedy et al. 2024: H2020 SME-Instrument era (SME-1/2/2b, awards
  # 2016-2021) vs Horizon Europe EIC era (HORIZON-EIC pure grants, awards
  # 2022-2024) on the pure-grant sets. Era is a set-level attribute (scheme of
  # the treated firm). The EIC era holds few sets, sensitivity only.
  er <- d_pure[d_pure$treated == 1, c("set_id", "set_scheme")]
  er$era <- ifelse(grepl("^HORIZON-EIC", er$set_scheme), "eic", "sme")
  d_pure$era <- er$era[match(d_pure$set_id, er$set_id)]
  for (e in c("sme", "eic")) {
    d_e <- d_pure[d_pure$era == e, ]
    cat(sprintf("  ERA %s: %d sets\n", e, length(unique(d_e$set_id))))
    grab(feols(post24_any_private ~ treated | set_id, d_e, cluster = ~set_id),
         paste0("V2_ERA_M1_", e), "sme_eic_grant_only", "post24_any_private",
         paste0("ERA_", e, "_set_FE"))
    trans_row(d_e, paste0("V2_TRANS_ERA_", e), "sme_eic_grant_only")
  }
  # post-grant KM curve from the v2 grant-only sets: overwrites the v1 write
  # above (the exhibit runs on v2; one rerun regenerates either variant)
  ev6 <- as.integer(!is.na(d_pure$post_first_private_offset_m))
  t6 <- ifelse(ev6 == 1, d_pure$post_first_private_offset_m,
               CUTOFF_IDX - d_pure$g_midx) + 1
  stopifnot(all(t6 > 0))
  km6 <- survfit(Surv(t6, ev6) ~ treated, data = d_pure)
  km6d <- data.frame(group = rep(names(km6$strata), km6$strata),
                     time_m = km6$time, surv = km6$surv,
                     lower = km6$lower, upper = km6$upper, n_risk = km6$n.risk)
  write.csv(km6d, "outputs/estimates/grants_km_postgrant.csv", row.names = FALSE)
  cat(sprintf("wrote grants_km_postgrant.csv from V2 sample (%d rows)\n", nrow(km6d)))

  # ---- Appendix 9-11 runs (additive) ----
  # (a) SE variants of the main ATT on the PURE-grant sets (Appendix 10 Panel C;
  # the existing V2_M1_tw runs on the pooled 600-set group)
  grab(feols(post24_any_private ~ treated | set_id, d_pure,
             cluster = ~set_id + firm_id),
       "V2_M1_grant_only_tw", "sme_eic_grant_only", "post24_any_private",
       "M1_twoway_set_firm")
  grab(feols(post24_any_private ~ treated | set_id, d_pure, vcov = "hetero"),
       "V2_M1_grant_only_hc", "sme_eic_grant_only", "post24_any_private",
       "M1_hetero_robust")
  # (b) bridge split at every horizon (Appendix 9): M1 + M2 mirrors at
  # 12/36/48 months; 24m rows exist above (V2_run_bridge_*). Horizon windows
  # must be complete (win36/win48), as in the catch-up runs.
  for (bs in c("noprior", "prior")) {
    d_b <- d_pure[d_pure$pre_any == as.integer(bs == "prior"), ]
    cat(sprintf("  BRIDGE %s horizon sets: 36m-complete %d | 48m-complete %d\n",
                bs, length(unique(d_b$set_id[d_b$win36_complete == 1])),
                length(unique(d_b$set_id[d_b$win48_complete == 1]))))
    grab(feols(post12_any_private ~ treated | set_id, d_b, cluster = ~set_id),
         paste0("V2_run_bridge_", bs, "_12m"), "sme_eic_grant_only",
         "post12_any_private", paste0("RUN_bridge_", bs, "_12m"))
    grab(feols(post36_any_private ~ treated | set_id,
               d_b[d_b$win36_complete == 1, ], cluster = ~set_id),
         paste0("V2_run_bridge_", bs, "_36m"), "sme_eic_grant_only",
         "post36_any_private", paste0("RUN_bridge_", bs, "_36m"))
    grab(feols(post48_any_private ~ treated | set_id,
               d_b[d_b$win48_complete == 1, ], cluster = ~set_id),
         paste0("V2_run_bridge_", bs, "_48m"), "sme_eic_grant_only",
         "post48_any_private", paste0("RUN_bridge_", bs, "_48m"))
    for (h in c("12", "24", "36", "48")) {
      y <- c("12" = "post12_any_private", "24" = "post24_any_private",
             "36" = "post36_any_private", "48" = "post48_any_private")[h]
      d_h <- if (h %in% c("36", "48"))
        d_b[d_b[[paste0("win", h, "_complete")]] == 1, ] else d_b
      grab(feols(as.formula(paste(y, "~ treated + founders_fill + founders_miss",
                                  "| set_id")), d_h, cluster = ~set_id),
           paste0("V2_run_bridge_", bs, "_", h, "m_M2"), "sme_eic_grant_only",
           y, paste0("RUN_bridge_", bs, "_", h, "m_M2"))
    }
  }
  # (c) post-award incidence for the four circ x treated groups (Appendix 11
  # Panel D; circ is an exact matching cell, so it is set-homogeneous). KM
  # read-offs at the horizon months reproduce the post{h}_any_private shares.
  stopifnot(all(tapply(d_pure$circ, d_pure$set_id, function(x) length(unique(x))) == 1))
  km4 <- survfit(Surv(t6, ev6) ~ circ + treated, data = d_pure)
  s4 <- summary(km4, times = c(12, 24, 36, 48), extend = TRUE)
  fg <- data.frame(group = as.character(s4$strata), horizon_m = s4$time,
                   incidence = 1 - s4$surv, n_risk = s4$n.risk,
                   lower = 1 - s4$upper, upper = 1 - s4$lower)
  write.csv(fg, "outputs/estimates/grants_fourgroup_km.csv", row.names = FALSE)
  cat(sprintf("wrote grants_fourgroup_km.csv (%d rows)\n", nrow(fg)))
}

# ---- Write estimates + headline ----
out <- do.call(rbind, rows)
dir.create("outputs/estimates", showWarnings = FALSE, recursive = TRUE)
write.csv(out, "outputs/estimates/grant_estimates.csv", row.names = FALSE)
zout <- do.call(rbind, zrows)
write.csv(zout, "outputs/estimates/grant_schoenfeld.csv", row.names = FALSE)
cat("Schoenfeld (PH) tests -> grant_schoenfeld.csv:\n")
print(data.frame(run = zout$run_id, term = zout$term, chisq = round(zout$chisq, 2),
                 df = zout$df, p = signif(zout$p, 2)), row.names = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/estimates/grant_sessioninfo.txt")
cat(sprintf("wrote grant_estimates.csv (%d rows)\n", nrow(out)))

hl <- out[out$term == "treated" & out$outcome == "post24_any_private"
          & out$spec %in% c("M0_means", "M1_set_FE", "M2_plus_covariates"), ]
cat("\nheadline P(private round within 24m) effects (pp):\n")
print(data.frame(group = hl$group, spec = hl$spec,
                 att_pp = round(100 * hl$estimate, 2), se_pp = round(100 * hl$se, 2),
                 p = signif(hl$p, 3)), row.names = FALSE)
