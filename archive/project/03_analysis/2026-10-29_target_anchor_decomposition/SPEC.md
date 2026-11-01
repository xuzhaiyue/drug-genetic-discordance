# Target-anchor decomposition and matched non-target negative control

Written 2026-10-29 before any result of this analysis was computed. Scope: GDSC2, frozen N3 universe
(43 included compounds, six focal targets EGFR/MTOR/ESR1/TOP1/ATR/CHEK1). Inputs, response orientation,
lineage basis, MIN_N=200 and MIN_PAIR_N=30 are inherited unchanged from `2026-09-28_phase1b_target_expansion/run_n3.py`.

## Question

How much of same-target response sharing is removed by adjusting for the measured dependency of the
annotated target, and is that removal larger than adjusting for a dependency-matched non-target gene?

## Part A — staged decomposition

Per compound, residual response under each adjustment (OLS with intercept, cells = frozen per-drug cohort,
i.e. response and true-target Chronos both observed, so every stage uses identical cells):

| Stage | Model |
|---|---|
| RAW | R (no adjustment) |
| D | R ~ D |
| L | R ~ lineage |
| DL | R ~ D + lineage (frozen primary; must reproduce frozen W_t, C_t to 1e-10 or the run stops) |
| LG | R ~ lineage + G |
| DLG | R ~ D + lineage + G |

G (general sensitivity) is pair-specific: for pair (a, b), G = per-cell median of SD-standardized DL
residuals of all included compounds excluding those assigned to target(a) or target(b); MTOR and FKBP1A
(rapalog overlap) are treated as one group for this exclusion. A cell needs >=80% of background compounds
observed. For each stage report W_t, C_t, Delta_t per target and the six-target median Delta_t.

Derived quantities (descriptive):
- Target-attributable attenuation of within-target sharing: A_t = W_t(L) - W_t(DL); with G: W_t(LG) - W_t(DLG).
- Fraction: f_t = A_t / W_t(L).
- Per-drug partial R^2 of D given lineage.

## Part B — matched non-target negative control

For each focal target t: candidate genes = Chronos genes fully observed on the union of t's compound cohorts,
excluding all 30 annotated target labels of the included compounds. Matching features = mean and SD of
D = -Chronos on that union, each standardized across candidates; the 100 nearest genes (Euclidean) form the
null set. No correlation-based exclusion.

For null gene k, every compound in t is re-residualized as R ~ gene_k + lineage; compounds outside t keep
their frozen DL residuals. Recompute W_t^k, C_t^k, Delta_t^k and A_t^k = W_t(L) - W_t^k.

Per target: one-sided empirical P = (1 + #{A_t^k >= A_t}) / (1 + 100); percentile of Delta_t(DL) within
{Delta_t^k}. Aggregate: 10,000 draws picking one null gene per target uniformly (seed 20261029); null of the
six-target median A; P as above with plus-one correction. Same for per-drug partial R^2 (true vs null).

## Pre-stated reading

- Target specificity supported for t if P <= 0.05 for A_t; aggregate supported if aggregate P <= 0.05.
- Size: report f_t as is. No threshold is used to call the residual "large" or "small".
- Outcomes are reported regardless of direction. If the target anchor is no better than matched genes,
  the manuscript must say that nominal-target dependency adjustment is non-specific in this space.
- No changes to the frozen N3 results, eligibility, or groups. PRISM is out of scope for this run.

## PRISM addendum (written 2026-10-29 after the GDSC2 results, before any PRISM result)

Same Part A/B definitions applied to the frozen PRISM universe: the 30 LEGAL compounds of
`n3_prism_compound_coverage.tsv`, four focal targets EGFR/MTOR/ESR1/TOP1. Loading copies `load_prism_coverage`
and `prism_analysis` exactly: PRISM Secondary AUC averaged by compound core BRD × ACH ID, cells restricted to the
GDSC2∩Chronos∩lineage universe, R = -z(AUC) on that set, OLS on the union of PRISM cells. G background, null-gene
pool (excluding all 30 GDSC2 target labels), matching, 100 null genes, seeds and P definitions are unchanged.
Reproduction gate: DL stage W_t, C_t vs frozen `n3_prism_targets.tsv` within 1e-4. Outputs go to `prism/`.
ESR1 and TOP1 have one within-target pair each in PRISM, so their W_t is a single correlation.
GDSC2 outputs must be byte-identical after the refactor (checked by re-running GDSC2).

## Deviations (recorded before any Part A/B result was seen)

1. Reproduction gate relaxed from 1e-10 to 1e-4. The first run stopped at the gate with max |diff| 3.1e-7
   (target level). Residual vectors are bit-identical to `rev_common.linear_ols` in the current environment
   (NumPy 2.4.3 / SciPy 1.17.1 vs frozen NumPy 2.0.2 / SciPy 1.13.1); 730/903 pair rho differ from the frozen
   pair table by <= 5.2e-5 because a few compounds have near-tied residuals (e.g. temsirolimus: 457 unique
   values in 460 cells at 1e-12) that tie in one build and split in another. No stage result was computed
   before this change.
