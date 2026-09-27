# 2_descriptive_statistics.R: summary of the model populations (Kennedy et
# al. 2024, Table 1 analog). The MAIN table runs on the month-verified sample
# (n = 16,615, the estimation sample of every Chapter 4 analysis); the full
# frame (22,039, January-imputed founding months) is written as the appendix
# robustness variant.
# Four columns circular / non-circular / EU grant / no grant; rows per the
# chapter spec: n, founded 2020+, 24m+36m incidence, median time among
# reachers, grant rates, patents, founders, age, status (exits/closed/dead).
# Chi-square for shares, Welch t for means, Mann-Whitney for medians. IP rows
# are DESCRIPTIVE ONLY (end-state stock, selective coverage; banned from all
# models).
#
# Inputs:  data/processed/firms_analysis.csv  (built by scripts/build_analysis.py)
#          data/raw/companies_fat.csv         (IP columns only, the single
#                                              raw-data dependency on the R side)
# Outputs: outputs/estimates/descriptives_t1.csv        (main, month-verified)
#          outputs/estimates/descriptives_t1_frame.csv  (appendix, full frame)
# Run:     from the repo root, after build_analysis.py (run_pipeline.sh step 6).
#
# Circular Startup Financing. Master's thesis, 2026.
###############################################################################

# ---- Load inputs ----
if (!file.exists("data/processed/firms_analysis.csv"))
  stop("Run this script from the repository root (see run_pipeline.sh).")
df <- read.csv("data/processed/firms_analysis.csv")  # pipeline step 3 output
if (Sys.getenv("REBASE") != "1") stopifnot(nrow(df) == 22039)  # REBASE=1 lifts pins (new data export, see scripts/sanity_checks.py)
CUTOFF_IDX <- 2026 * 12 + 2  # panel end 2026-03 (month_index = year*12 + month - 1)

# IP columns from the raw firm table (selective read via colClasses)
hdr <- names(read.csv("data/raw/companies_fat.csv", nrows = 1))
cc <- rep("NULL", length(hdr))
cc[hdr %in% c("crunchbase_url", "cbcsv_patents_granted")] <- "character"
ip <- read.csv("data/raw/companies_fat.csv", colClasses = cc)
names(ip)[names(ip) == "crunchbase_url"] <- "firm_id"
df <- merge(df, ip, by = "firm_id", all.x = TRUE)

# ---- Derived columns for the table rows ----
# dead by any signal: CB Closed status OR dead website at the 2026-03 scrape
# (register dissolutions only DATE deaths, they do not define this flag)
df$dead_any <- as.integer(df$status_closed == 1 |
                          (!is.na(df$site_dead_2026_03) & df$site_dead_2026_03 == 1))
df$m2p <- df$first_private_offset_m_v2
df$patents <- suppressWarnings(as.numeric(df$cbcsv_patents_granted))
df$fund24_private_v2 <- as.integer(!is.na(df$first_private_offset_m_v2)
                                   & df$first_private_offset_m_v2 <= 23)
df$founded_2020p <- as.integer(df$founding_month_idx %/% 12 >= 2020)

# ---- Table builder: four populations x row spec ----
stars <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**",
                     ifelse(p < .05, "*", "")))

run_t1 <- function(d, out_file, label) {
  pops <- list(
    "M1 Circular" = d[d$circ == 1, ],
    "M2 Non-circular" = d[d$circ == 0, ],
    "M3 EU grant" = d[d$has_grant == 1, ],
    "M4 No EU grant" = d[d$has_grant == 0 & d$has_grant_uncertain == 0, ])
  PAIRS <- list(c("M1 Circular", "M2 Non-circular"),
                c("M3 EU grant", "M4 No EU grant"))
  p_share <- function(col, pair) {
    a <- pops[[pair[1]]][[col]]; b <- pops[[pair[2]]][[col]]
    suppressWarnings(chisq.test(matrix(c(sum(a), length(a) - sum(a),
                                         sum(b), length(b) - sum(b)), 2)))$p.value
  }
  p_num <- function(col, pair, median = FALSE) {
    a <- pops[[pair[1]]][[col]]; b <- pops[[pair[2]]][[col]]
    if (median) wilcox.test(a, b)$p.value else t.test(a, b)$p.value
  }
  ROWS <- list(
    list("Number of startups", "n", NA, NA),
    list("Founded 2020 or later", "share", "founded_2020p", FALSE),
    list("First private round within 24 months", "share", "fund24_private_v2", FALSE),
    list("First private round within 36 months", "share", "fund36_private_v2", FALSE),
    list("Months to first private round (median, among firms reaching one)",
         "median", "m2p", TRUE),
    # has_grant is definitional for the grant/no-grant split -> pair not tested
    list("Grant round within 36 months (Crunchbase)", "share", "grant36_cb_v2", FALSE),
    list("Verified EU grant (CORDIS)", "share", "has_grant", FALSE, TRUE),
    list("Mean patents granted (among IP-covered)", "mean", "patents", FALSE),
    list("Mean number of founders", "mean", "n_founders", FALSE),
    list("Mean firm age at cutoff (years)", "mean", "age_y", FALSE),
    list("Exits (acquired/IPO, status at cutoff)", "share", "exit_any", FALSE),
    list("Failures — closed (CB status)", "share", "status_closed", FALSE),
    list("Failures — dead by any signal", "share", "dead_any", FALSE))
  recs <- list()
  for (r in ROWS) {
    lbl <- r[[1]]; kind <- r[[2]]; col <- r[[3]]; med <- r[[4]]
    ps <- c(NA, NA)
    if (kind != "n") {
      ps <- sapply(PAIRS, function(pr)
        if (kind == "share") p_share(col, pr) else p_num(col, pr, med))
      if (length(r) >= 5 && isTRUE(r[[5]])) ps[2] <- NA  # definitional grant pair
    }
    cells <- sapply(names(pops), function(nm) {
      dd <- pops[[nm]]
      pi <- if (nm %in% PAIRS[[1]]) ps[1] else if (nm %in% PAIRS[[2]]) ps[2] else NA
      tag <- if (!is.na(pi)) sprintf("%s p=%.3g", stars(pi), pi) else ""
      if (kind == "n") format(nrow(dd), big.mark = ",")
      else if (kind == "share") sprintf("%s (%.1f%%)%s",
        format(sum(dd[[col]], na.rm = TRUE), big.mark = ","),
        100 * mean(dd[[col]], na.rm = TRUE), tag)
      else if (kind == "median") sprintf("%.0f%s", median(dd[[col]], na.rm = TRUE), tag)
      else sprintf("%.2f%s", mean(dd[[col]], na.rm = TRUE), tag)
    })
    recs[[length(recs) + 1]] <- c(row = lbl, cells)
  }
  t1 <- do.call(rbind, lapply(recs, function(x) as.data.frame(t(x))))
  write.csv(t1, out_file, row.names = FALSE)
  cov <- sapply(pops, function(dd) sprintf("n=%d (%.0f%%)",
                sum(!is.na(dd$patents)), 100 * mean(!is.na(dd$patents))))
  cat(sprintf("wrote %s (%s), IP coverage: %s\n", out_file, label,
              paste(names(cov), cov, collapse = " | ")))
}

# ---- Table 1 (main): month-verified sample ----
# age from the verified v2 founding month
d_mv <- df[df$founding_month_imputed_v2 == 0, ]
d_mv$age_y <- (CUTOFF_IDX - d_mv$founding_month_idx_v2) / 12
run_t1(d_mv, "outputs/estimates/descriptives_t1.csv",
       sprintf("month-verified, n=%d", nrow(d_mv)))

# ---- Table 1 (appendix variant): full frame ----
# incl. January-imputed founding months (age from the CB founding month,
# as before the mv switch)
df$age_y <- (CUTOFF_IDX - df$founding_month_idx) / 12
run_t1(df, "outputs/estimates/descriptives_t1_frame.csv",
       sprintf("full frame, n=%d", nrow(df)))
