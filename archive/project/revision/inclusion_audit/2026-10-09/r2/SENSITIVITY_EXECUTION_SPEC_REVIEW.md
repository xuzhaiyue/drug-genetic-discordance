# N3 pharmacology sensitivity execution specification

Status: **AWAITING EXTERNAL REVIEW; NOT AUTHORIZED TO RUN**. Version 2, 2026-10-09. Addresses independent review M1–M5; version 1 is preserved in the parent directory.

This is a result-aware amendment prepared after inspection of historical N3 results. It is not retrospective preregistration. Approval and a content hash must precede any new response calculation. The historical 43-drug universe, original target labels, claim ledger and figures remain intact. This document proposes an exact execution design for review; it does not approve promotion of N3 to the main discovery cohort.

## 1. Inputs and evidence boundary

Use `PROJECT_ROOT` and the relative paths in this package's `run_config.yaml`. Hash the original eligibility/scaffold/group tables, original N3 specification/configuration, original pair/target tables, raw-input manifest, this document, the 43-row review, and the overlap mask before execution. Resolve symlinks and verify original manifest hashes. Do not modify raw files or historical outputs.

Annotation has separate fields for frozen target, binding/functional entities, mechanism, broad pathway, active-moiety relation and source URLs. A mechanism-supported entry is not evidence of exclusive selectivity at screening concentrations. Broad pathway overlap is a reviewer-defined classification; it is not evidence that two drugs bind the same protein. No target relabeling, additional compound, annotation chosen by response, new gene-anchor model or expansion of unresolved drugs is permitted here.

Focal target candidates remain EGFR, MTOR, ESR1, TOP1, ATR and CHEK1. PRISM uses discovery-derived identities and the historical LEGAL coverage flags; its results cannot select GDSC2 membership. No RAS drugs are added in this amendment.

## 2. Ordered variants — report all

| ID | Intervention |
|---|---|
| S0 | Reproduce original 43/30-drug results and all original pair/target outputs. |
| S1 | Restrict within-target pairs to frozen distinct-scaffold pairs (Morgan scaffold Tanimoto <0.70); cross controls remain as in S0. |
| S2 | Exclude BEZ235 from focal and control sets. |
| S3 | Exclude AZD7762 from focal and control sets. |
| S4 | Remove only the irinotecan–SN-38 within-target pair; retain both drugs' other comparisons. |
| S5 | Exclude irinotecan globally; retain SN-38. |
| S6 | Exclude tamoxifen globally. |
| S7 | Exclude BEZ235, AZD7762 and irinotecan globally. |
| S8 | Globally exclude all 12 drugs with reviewed multiple-target, multiple-mechanism, family-inhibitor or unresolved-isoform flags; exact IDs are in `run_config.yaml`. This is an evidence-defined restriction, not a fully selective-drug universe. |
| S9 | Remove cross-target controls sharing narrow reviewed mechanism tokens, including MTOR–FKBP1A rapalog comparisons. Keep focal pairs. |
| S10 | Remove cross-target controls sharing broad reviewed pathway tokens. This deliberately broader definition is a separate stress test. |
| S11 | Combine S8 + irinotecan/tamoxifen exclusions + distinct-scaffold within-pair restriction + parent/metabolite-pair restriction + S9 narrow-control restriction. |

S9/S10 masks are fixed on compound identities, not inferred from residual correlations. Never promote FKBP1A to a new MTOR discovery group in this exercise. S11 does not include S10; their estimands differ and are reported separately. Retain the full 43-row annotation even when a scenario excludes a compound.

## 3. Eligibility, target loss and control reconstruction

Retain historical per-drug complete-case minimum 200 and per-pair minimum 30. Use the original residual definition and actual original fitting implementation: per-drug OLS of response on the assigned gene dependency and lineage. This N3 residual is not silently replaced by the old Arena cross-fitted model.

After each global exclusion, rebuild each target's membership and all cross-target controls from the retained universe. A target needs at least two retained compounds, at least two different frozen scaffolds, at least one allowed within pair with finite rho and n >= 30, and at least one allowed cross pair with finite rho and n >= 30. Otherwise mark it lost with the exact reason; do not borrow another target or substitute a zero. S4 removes a pair without deleting its drugs. S1/S11 use the exact distinct_pairs IDs frozen in PHASE1B_TARGET_GROUPS.tsv (Tanimoto <0.70), not merely unequal scaffold SMILES. They do not restrict arbitrary cross-pair scaffold similarity.

For each surviving target t, W_t is median allowed within-target rho; C_t is median allowed pairs with exactly one drug in t and one outside t; delta_t = W_t - C_t. The overall estimand is the unweighted median of delta_t over surviving targets. Preserve every underlying pair, mask and denominator.

Minimum groups for a population-level sensitivity status: **GDSC2 >= 4, PRISM >= 4**. Four is a proposed GDSC2 floor; the PRISM floor preserves the historical four-group requirement. Below the floor, report per-target effects and an explicitly descriptive median, status `COVERAGE_INSUFFICIENT`; do not declare PASS or relaxed replication. Keep targets lost through pharmacology exclusions visible even when the remaining median increases.

Report both the surviving-set aggregate and S0 restricted to exactly the same surviving targets. Compare interventions only on this common target set. A changed-target median is not an estimate of effect retention for the original six/four targets. Baseline-common median <= 0 makes any relative retention undefined; never divide by zero or a negative baseline and interpret a favorable ratio.

## 4. Inference and null implementation

### 4.1 Baseline gate

S0 must reproduce historical drug/pair/target identifiers, n, scaffold eligibility, residual conventions and deterministic W_t/C_t/delta_t. Use absolute rho tolerance 1e-8 for this amendment's direct deterministic reproduction, identical row order and numeric environment; failure pauses for review and must not be fixed by widening the tolerance after seeing the discrepancy. Also run the previously approved revision regression gate using its approved tolerances. If either gate fails, stop before sensitivities.

Preserve the original 10,000 target-label permutations and 10,000 target bootstrap draws, seed 20260927 and resource offsets from the original script. The original null must refit residuals under each permuted gene assignment, or select mathematically identical cached fits. Permuting pair labels or retaining observed-anchor residuals is not equivalent. Count shared-drug dependence; do not treat drug pairs as independent replicates.

### 4.2 Sensitivity inference

All eligible S1–S11 variants report 10,000 percentile target-bootstrap draws of the median delta_t (2.5th/97.5th percentiles), using NumPy default linear quantiles and a recorded fixed seed `20260927 + 1000*scenario_index + resource_offset`, where GDSC2 offset 0 and PRISM offset 100. Targets, not pairs, are the resampling unit. Report the discrete/low-group limitations and leave-one-target-out effects. This bootstrap is a descriptive uncertainty summary for this finite target set, not proof of a population sampling model.

**Do not generate new sensitivity permutation P values in this specification.** The original label-null is preserved for S0 only. With identity-fixed pharmacology masks, outcome-informed annotation, scaffold requirements and target losses, unrestricted label exchangeability is not established. Reusing S0's P, pair-label shuffling or excluding invalid permutations until significant would be invalid. Sensitivity variants use effect, target consistency, coverage and bootstrap summaries; any new conditional/stratified permutation model requires a separate reviewed amendment before execution. This resolves the null ambiguity conservatively.

No pair-level Mann–Whitney P is a replacement for a drug-dependent null. No cell/pair bootstrap silently replaces the target bootstrap. No significance selection among variants; no multiplicity-adjusted confirmatory assertion is made from sensitivity CIs.

## 5. Decision and stopping rules

Technical STOP: baseline/regression mismatch; source/hash mismatch; inconsistent compound identities; missing scaffold; inconsistent masks; unapproved annotation changes; nonfinite eligible W_t/C_t; implementation cannot reproduce stated estimand. Save a failure report and stop, without changing claims or figures.

Scientific unfavorable results do **not** terminate the remaining ordered variants. Complete all technically valid scenarios and report failures. For a scenario with >=4 groups, use the historical descriptive rule CI lower bound >0 and fraction of positive targets >=0.70 to label `DESCRIPTIVELY_CONSISTENT`; otherwise label `DESCRIPTIVELY_INCONSISTENT`. Never call this a confirmatory replication P value. Low-coverage variants remain `COVERAGE_INSUFFICIENT` regardless of effect size.

GDSC2 discovery-universe recommendation remains HOLD for external outcome review if an eligible restrictive GDSC2 scenario is descriptively inconsistent or a common-set aggregate changes sign. PRISM strict-universe coverage loss does not automatically veto GDSC2 discovery-universe validity. Evaluate four distinct decisions: GDSC2 discovery validity; S0 broad-universe PRISM replication; strict-universe PRISM replication; manuscript claims/figure placement. Strict PRISM with fewer than four groups is COVERAGE_INSUFFICIENT regardless of effect and cannot support a strict cross-resource claim. No automatic PASS-to-main-paper rule: even all favorable results require external review of annotation, estimands and interpretation. Coverage loss is a design limitation, not evidence against the biological effect and not evidence of robustness. The specific strict PRISM variants already predict coverage loss; report that openly.

No magnitude-retention threshold is introduced: there is no independently justified 50%/60%/80% cutoff. Report common-set absolute change and ratio when mathematically defined, including all large attenuations, without threshold tuning.

## 6. Required outputs after approval

Separate dated analysis directory; README, run_config, input/output SHA, software versions, timing and logs. Each variant/resource produces retained drugs, excluded drugs/reasons, complete pair table with masks, per-target W_t/C_t/delta_t, target eligibility/loss reasons, common-target baseline comparisons, target bootstrap draws and leave-one-target-out summaries. Save masks and failures. Main report includes all twelve variants, all six candidate targets, PRISM coverage losses, unresolved selectivity questions and deviations. Preserve S0 and all historical files.

Before any later Figure 1 replacement, independently review the original N3 pair/target results and the sensitivity report. Do not transfer old Arena Figure 2 numbers to a new universe. No writing/figure/claim edits or git operations are authorized by this specification.

## 7. Reviewer decisions required

Approve or amend: all43 flag definitions and sources; exact S8 exclusion IDs; narrow/broad overlap token mapping; proposed GDSC2 minimum four-group floor; retained PRISM four-group floor; no new sensitivity permutation P values; support/coverage labels. Changes must be dated and hashed before new calculations. Until then, only metadata/source auditing is permitted.

## 8. Version 2 implementation clarifications

- Token matching: split semicolon-delimited values, strip whitespace, remove empty tokens, deduplicate; match exact case-sensitive tokens only. Missing mechanism/pathway tokens stop mask construction for review rather than becoming proof of non-overlap. Multiple shared tokens create one unordered drug-pair mask, with every shared token retained. Mask only different frozen target labels. S9 is narrow mechanism intersection; S10 is broad pathway intersection. The accompanying reconciliation checks 12/48 all-drug masks and 10/31 masks touching focal labels. These are metadata counts only.
- Upload scenario_drug_membership.tsv, scenario_pair_masks.tsv, scenario_pair_counts.tsv and sensitivity_target_losses.tsv. Every retained drug and every within/cross pair has a scenario/resource/target record. Removed drugs have explicit reasons; identity-deleted cross pairs are recovered by joining S0 pairs to scenario membership. At execution, report removed controls by identity, overlap, and cell-N/nonfinite gate as separate sequential counts without double counting.
- For every surviving target report W_t(S0), W_t(S), change_W; C_t(S0), C_t(S), change_C; delta_t(S0), delta_t(S), change_delta; exact pair denominators and removal reasons. Identity removal first, mechanism removal second, cell gate third. S9/S10 must keep W_t unchanged relative to S0; stop on mismatch. Their delta increases may reflect lower C_t only and are labeled contrast-definition stress tests.
- Rapamycin/temsirolimus have pharmacological_family=RAPALOG and an explicit possible active-moiety relationship note. FDA human metabolism evidence does not establish conversion in these in-vitro screens. Keep active_moiety_family reserved for the operational irinotecan–SN38 pair restriction; the new RAPALOG field alone does not introduce additional S4 exclusions.
- Four to six selected target groups support descriptive uncertainty only. Ten thousand resamples are numerical draws, not independent targets. Four-group floors are coverage/reporting rules, not representativeness proofs. The 0.70 threshold is documented in revision/STAGE3_N3_SPEC.md section 1 and run_n3.py summarize_effect; applying it to the new sensitivities is explicitly a current result-aware descriptive rule.
- Review code excerpts and full original script before S0. Static inspection alone does not validate execution or cached fit equivalence for all permuted anchors. No original code change is authorized.
