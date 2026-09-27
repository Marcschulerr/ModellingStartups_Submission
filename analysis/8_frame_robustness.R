# 8_frame_robustness.R: frame-selection robustness for the grants RQ plus
# one RQ1 frame check.
#
# The analysis frame requires >= 1 recorded financing event. A control with no
# grant and no private round by its pseudo-award is usually in the frame only
# through a LATER event, often the outcome itself, while recipients enter
# through the award. Five blocks:
#   T  entry tabulation: share of units whose FIRST recorded event lies after
#      the (pseudo-)award, by treatment x prior-funding stratum
#   A/B restricted re-estimation: controls (A) resp. both sides (B) must have
#      entered the frame before the award; full model pack re-run
#   G  follow-on grants on NON-EU grant months only (controls cannot hold an
#      EU follow-on by construction; new count from frame_robustness_build.py)
#   F  future-winner arm: later H2020 winners as controls (entered the frame
#      through their own award AND passed the same selection); set gate >= 150
#   R1 RQ1 S2 gap re-estimated on ever-private firms only (timing component
#      is free of the frame-entry margin)
#
# Inputs:  data/processed/grants_matched_mv.csv, frame_entry.csv,
#          grants_future_winner.csv (scripts/frame_robustness_build.py),
#          data/processed/firms_analysis.csv (R1 block)
# Outputs: outputs/estimates/frame_robustness.csv,
#          frame_entry_tabulation.csv (+ frame_sessioninfo.txt)
# Run:     from the repo root, after scripts/frame_robustness_build.py.
#
# Circular Startup Financing. Master's thesis, 2026.
###############################################################################
suppressMessages({library(fixest); library(survival)})

if (!file.exists("data/processed/grants_matched_mv.csv"))
  stop("Run this script from the repository root (see run_pipeline.sh).")
m  <- read.csv("data/processed/grants_matched_mv.csv")   # pipeline step 5 (mv)
fe <- read.csv("data/processed/frame_entry.csv")          # frame_robustness_build.py
fw <- read.csv("data/processed/grants_future_winner.csv") # frame_robustness_build.py

CUTOFF_IDX <- 2026 * 12 + 2  # month_index("2026-03"), single definition src/cohort.py
PIN <- Sys.getenv("REBASE") != "1"  # REBASE=1 lifts pins (new data export, see scripts/sanity_checks.py)

# ---- Pure SME/EIC sets + eligibility filter, exactly as in 4_matched_grants.R ----
tl <- m[m$treated == 1, c("set_id", "land", "g_midx")]
names(tl)[2:3] <- c("set_land", "set_g")
m <- merge(m, tl, by = "set_id")
# eligibility: UK/CH awards >= 2021 fall in the CORDIS-unobservable period, excluded
m <- m[!((m$set_land %in% c("UK", "CH")) & (m$set_g %/% 12 >= 2021)), ]
s <- m[m$group == "sme_eic" & m$bf_set == 0, ]
s <- merge(s, fe[, c("set_id", "firm_id", "first_ev", "first_grant_ev",
                     "post24_n_grant_months_noneu")],
           by = c("set_id", "firm_id"), all.x = TRUE)
s$pre_any <- as.integer(s$pre_n_private > 0)
stopifnot(all(tapply(s$pre_any, s$set_id, function(x) length(unique(x))) == 1))
if (PIN) stopifnot(length(unique(s$set_id)) == 453)
cat(sprintf("canonical: %d sets, %d rows (%d treated)\n",
            length(unique(s$set_id)), nrow(s), sum(s$treated == 1)))

rows <- list()
put <- function(run_id, arm, outcome, term, est, se, p, n, n_sets, scale) {
  rows[[length(rows) + 1]] <<- data.frame(
    run_id = run_id, arm = arm, outcome = outcome, term = term,
    estimate = est, se = se, p = p, n = n, n_sets = n_sets, scale = scale)
}

# ---- Block T: frame-entry tabulation ------------------------------------
s$late_entry <- as.integer(is.na(s$first_ev) | s$first_ev > s$g_midx)
tab <- list()
for (pa in 0:1) for (tr in 0:1) {
  d <- s[s$pre_any == pa & s$treated == tr, ]
  tab[[length(tab) + 1]] <- data.frame(
    stratum = ifelse(pa == 1, "prior", "noprior"),
    group = ifelse(tr == 1, "treated", "control"), n = nrow(d),
    share_late_entry = mean(d$late_entry),
    p_private_24m = mean(d$post24_any_private))
}
d0 <- s[s$pre_any == 0 & s$treated == 0, ]
tab[[length(tab) + 1]] <- data.frame(
  stratum = "noprior", group = "control_late_entry",
  n = sum(d0$late_entry == 1),
  share_late_entry = 1, p_private_24m = mean(d0$post24_any_private[d0$late_entry == 1]))
tab[[length(tab) + 1]] <- data.frame(
  stratum = "noprior", group = "control_early_entry",
  n = sum(d0$late_entry == 0),
  share_late_entry = 0, p_private_24m = mean(d0$post24_any_private[d0$late_entry == 0]))
# what lets an early-entry no-prior control into the frame: a CB grant month
e0 <- d0[d0$late_entry == 0, ]
tab[[length(tab) + 1]] <- data.frame(
  stratum = "noprior", group = "control_early_via_cb_grant",
  n = sum(!is.na(e0$first_grant_ev) & e0$first_grant_ev <= e0$g_midx),
  share_late_entry = mean(!is.na(e0$first_grant_ev) & e0$first_grant_ev <= e0$g_midx),
  p_private_24m = NA)
tab <- do.call(rbind, tab)
print(tab, row.names = FALSE)
write.csv(tab, "outputs/estimates/frame_entry_tabulation.csv", row.names = FALSE)

# ---- Blocks A/B: restricted re-estimation --------------------------------
run_models <- function(d, arm) {
  d$founders_fill <- ifelse(is.na(d$n_founders),
                            median(d$n_founders, na.rm = TRUE), d$n_founders)
  d$founders_miss <- as.integer(is.na(d$n_founders))
  grab <- function(mod, id, term = "treated") {
    ct <- coeftable(mod)
    if (!term %in% rownames(ct)) return(invisible(NULL))
    put(id, arm, "post24_any_private", term, ct[term, 1], ct[term, 2],
        ct[term, 4], mod$nobs, length(unique(mod$fixef_id$set_id)), "prop")
  }
  trans <- function(dd, id) {
    ev <- as.integer(!is.na(dd$post_first_private_offset_m))
    tt <- ifelse(ev == 1, dd$post_first_private_offset_m,
                 CUTOFF_IDX - dd$g_midx) + 1
    cx <- coxph(Surv(tt, ev) ~ treated + strata(set_id), data = dd)
    cs <- summary(cx)$coefficients
    put(id, arm, "time_to_private", "treated_HR", exp(cs["treated", "coef"]),
        cs["treated", "se(coef)"], cs["treated", "Pr(>|z|)"], cx$n,
        length(unique(dd$set_id)), "HR")
  }
  grab(feols(post24_any_private ~ treated | set_id, d, cluster = ~set_id), "M1")
  grab(feols(post24_any_private ~ treated + founders_fill + founders_miss |
               set_id, d, cluster = ~set_id), "M2")
  grab(feols(pre_n_private ~ treated | set_id, d, cluster = ~set_id),
       "placebo_pre_rounds")
  for (bs in c(0, 1)) {
    db <- d[d$pre_any == bs, ]
    if (length(unique(db$set_id)) >= 10) {
      lab <- ifelse(bs == 1, "prior", "noprior")
      grab(feols(post24_any_private ~ treated | set_id, db, cluster = ~set_id),
           paste0("bridge_", lab))
      trans(db, paste0("TRANS_", lab))
    }
  }
  grab(feols(post24_any_private ~ treated + treated:pre_any | set_id, d,
             cluster = ~set_id), "bridge_diff", term = "treated:pre_any")
  d$small <- as.integer(d$size_class == "small")
  grab(feols(post24_any_private ~ treated | set_id, d[d$small == 1, ],
             cluster = ~set_id), "dose_small")
  grab(feols(post24_any_private ~ treated | set_id, d[d$small == 0, ],
             cluster = ~set_id), "dose_large")
  grab(feols(post24_any_private ~ treated + treated:small | set_id, d,
             cluster = ~set_id), "dose_diff", term = "treated:small")
  trans(d, "TRANS_pooled")
}
filter_sets <- function(d) {
  keep <- tapply(d$treated, d$set_id, function(x) any(x == 1) & any(x == 0))
  d[d$set_id %in% names(keep)[keep], ]
}
# Variant A: controls entered the frame BEFORE the award
d_varA <- filter_sets(s[s$treated == 1 |
                      (!is.na(s$first_ev) & s$first_ev < s$g_midx), ])
# Variant B (symmetric): treated too, the award itself counting as entry
tr_ok <- s$treated == 1 & (
  (!is.na(s$first_ev) & s$first_ev <= s$g_midx) |
  (!is.na(s$first_grant_ev) & s$first_grant_ev > s$g_midx &
     s$first_grant_ev <= s$g_midx + 3))
d_varB <- filter_sets(s[tr_ok | (s$treated == 0 & !is.na(s$first_ev) &
                               s$first_ev < s$g_midx), ])
if (PIN) stopifnot(length(unique(d_varA$set_id)) == 267,
                   length(unique(d_varB$set_id)) == 255)
cat(sprintf("Variant A: %d sets | Variant B: %d sets\n",
            length(unique(d_varA$set_id)), length(unique(d_varB$set_id))))
run_models(d_varA, "A_control_entry")
run_models(d_varB, "B_symmetric_entry")
# internal validity check: the prior stratum has 0% late entry on both sides,
# so its restricted estimate must reproduce the canonical +9.3pp exactly
pa <- do.call(rbind, rows)
chk <- pa[pa$run_id == "bridge_prior" & pa$arm == "A_control_entry", "estimate"]
if (PIN) stopifnot(abs(chk - 0.0927) < 0.001)

# ---- Block G: follow-on grants on non-EU grant months only ----------------
s$any24_noneu <- as.integer(s$post24_n_grant_months_noneu > 0)
cat(sprintf("any further NON-EU grant within 24m: treated %.3f vs control %.3f\n",
            mean(s$any24_noneu[s$treated == 1]),
            mean(s$any24_noneu[s$treated == 0])))
put("share_treated", "G_noneu_grants", "any24_grant_noneu", "share",
    mean(s$any24_noneu[s$treated == 1]), NA, NA, sum(s$treated == 1),
    length(unique(s$set_id)), "prop")
put("share_control", "G_noneu_grants", "any24_grant_noneu", "share",
    mean(s$any24_noneu[s$treated == 0]), NA, NA, sum(s$treated == 0),
    length(unique(s$set_id)), "prop")
g1 <- feols(any24_noneu ~ treated | set_id, s, cluster = ~set_id)
put("anygrant24_noneu", "G_noneu_grants", "any24_grant_noneu", "treated",
    coeftable(g1)["treated", 1], coeftable(g1)["treated", 2],
    coeftable(g1)["treated", 4], g1$nobs, length(unique(s$set_id)), "prop")
g2 <- feols(post24_n_grant_months_noneu ~ treated | set_id, s, cluster = ~set_id)
put("treadmill_noneu", "G_noneu_grants", "post24_n_grant_months_noneu", "treated",
    coeftable(g2)["treated", 1], coeftable(g2)["treated", 2],
    coeftable(g2)["treated", 4], g2$nobs, length(unique(s$set_id)), "months")
# group means of the non-EU grant-month count: Table 7 Panel B reports
# Difference/Treated/Controls, so the means need a canonical source next to
# the set-FE difference above
put("treadmill_noneu_mean_treated", "G_noneu_grants",
    "post24_n_grant_months_noneu", "mean",
    mean(s$post24_n_grant_months_noneu[s$treated == 1]), NA, NA,
    sum(s$treated == 1), length(unique(s$set_id)), "months")
put("treadmill_noneu_mean_control", "G_noneu_grants",
    "post24_n_grant_months_noneu", "mean",
    mean(s$post24_n_grant_months_noneu[s$treated == 0]), NA, NA,
    sum(s$treated == 0), length(unique(s$set_id)), "months")

# ---- Block F: future-winner arm -------------------------------------------
if (PIN) stopifnot(length(unique(fw$fw_set_id)) == 169)
cat(sprintf("future-winner arm: %d sets, %d rows\n",
            length(unique(fw$fw_set_id)), nrow(fw)))
fgrab <- function(mod, id) {
  ct <- coeftable(mod)
  put(id, "F_future_winner", "post24_any_private", "treated", ct["treated", 1],
      ct["treated", 2], ct["treated", 4], mod$nobs,
      length(unique(mod$fixef_id$fw_set_id)), "prop")
}
fgrab(feols(post24_any_private ~ treated | fw_set_id, fw, cluster = ~fw_set_id),
      "FW_M1")
p1 <- feols(pre_n_private ~ treated | fw_set_id, fw, cluster = ~fw_set_id)
put("FW_placebo_pre_rounds", "F_future_winner", "pre_n_private", "treated",
    coeftable(p1)["treated", 1], coeftable(p1)["treated", 2],
    coeftable(p1)["treated", 4], p1$nobs, length(unique(fw$fw_set_id)), "count")
for (bs in c(0, 1)) {
  db <- fw[fw$prior == bs, ]
  if (length(unique(db$fw_set_id)) >= 10)
    fgrab(feols(post24_any_private ~ treated | fw_set_id, db,
                cluster = ~fw_set_id),
          paste0("FW_bridge_", ifelse(bs == 1, "prior", "noprior")))
}
# transition Cox; future winners are censored at their OWN award month (they
# become treated there), recipients at the panel end
ev <- as.integer(!is.na(fw$post_first_private_offset_m))
cens <- ifelse(fw$treated == 1, CUTOFF_IDX - fw$eval_midx,
               fw$own_award_midx - fw$eval_midx)
ev[ev == 1 & fw$post_first_private_offset_m >= cens] <- 0
tt <- ifelse(ev == 1, fw$post_first_private_offset_m, cens) + 1
cx <- coxph(Surv(tt, ev) ~ treated + strata(fw_set_id), data = fw)
cs <- summary(cx)$coefficients
put("FW_TRANS_pooled", "F_future_winner", "time_to_private", "treated_HR",
    exp(cs["treated", "coef"]), cs["treated", "se(coef)"],
    cs["treated", "Pr(>|z|)"], cx$n, length(unique(fw$fw_set_id)), "HR")

# ---- Block R1: RQ1 S2 gap on ever-private firms ---------------------------
df <- read.csv("data/processed/firms_analysis.csv")  # pipeline step 3 output
if (PIN) stopifnot(nrow(df) == 22039)
IND <- grep("^ind_", names(df), value = TRUE)
df$founders_cat <- cut(df$n_founders, c(0, 1, 2, 3, 4, Inf),
                       labels = c("1", "2", "3", "4", "5plus"))
df$founders_cat <- factor(ifelse(is.na(df$founders_cat), "missing",
                                 as.character(df$founders_cat)))
df$nind_cat <- cut(df$n_industries, c(0, 2, 4, 6, Inf),
                   labels = c("1-2", "3-4", "5-6", "7plus"))
df$nind_cat <- factor(ifelse(is.na(df$nind_cat), "none", as.character(df$nind_cat)))
S2_RHS <- paste(c("founders_cat", "nind_cat", IND), collapse = " + ")
# month-verified sample AND >= 1 private round by the panel end: within
# ever-funded-privately firms the 36m gap is a pure timing quantity, free of
# the frame-entry margin
d_mv <- df[df$founding_month_imputed_v2 == 0 &
           !is.na(df$first_private_offset_m_v2), ]
cat(sprintf("R1 ever-private sample: %d firms (circ %d)\n",
            nrow(d_mv), sum(d_mv$circ)))
r0 <- feols(fund36_private_v2 ~ circ, d_mv, vcov = "hetero")
put("R1_everpriv_S0", "R1_rq1_frame", "fund36_private_v2", "circ",
    coeftable(r0)["circ", 1], coeftable(r0)["circ", 2],
    coeftable(r0)["circ", 4], r0$nobs, NA, "prop")
r2 <- feols(as.formula(paste("fund36_private_v2 ~ circ +", S2_RHS,
                             "| land^t_year")), d_mv, vcov = "hetero")
put("R1_everpriv_S2", "R1_rq1_frame", "fund36_private_v2", "circ",
    coeftable(r2)["circ", 1], coeftable(r2)["circ", 2],
    coeftable(r2)["circ", 4], r2$nobs, NA, "prop")

# ---- Write ----------------------------------------------------------------
out <- do.call(rbind, rows)
print(cbind(out[, c("run_id", "arm", "term")],
            est = round(out$estimate, 4), se = round(out$se, 4),
            p = signif(out$p, 3), n = out$n, n_sets = out$n_sets),
      row.names = FALSE)
write.csv(out, "outputs/estimates/frame_robustness.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()),
           "outputs/estimates/frame_sessioninfo.txt")
cat("written: outputs/estimates/frame_robustness.csv + frame_entry_tabulation.csv\n")
