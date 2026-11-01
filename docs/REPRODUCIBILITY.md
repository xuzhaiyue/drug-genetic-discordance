# Reproduction guide

## Scientific implementation map

| Component | Executed source under archive/project | Accessible core |
| --- | --- | --- |
| Eligibility | discordance/build_phase1b_chembl36_eligibility.py | frozen data/phase1b metadata |
| Discovery and PRISM | 03_analysis/2026-09-28_phase1b_target_expansion/run_n3.py | residuals.py; sharing.py |
| Pharmacological sensitivities | 03_analysis/2026-10-09_n3_pharmacology_sensitivity/run_sensitivity.py | scenario masks and component outputs |
| Nonlinear and dependency-source comparisons | run_n3.py; matched Chronos recovery script | archived fits/results |
| Response-assisted prediction | corrected 03_analysis/2026-10-17_prediction_execution/run_predictions.py; runtime_preflight.py | prediction.py |
| Prediction evaluation | 03_analysis/2026-10-17_prediction_execution/evaluate_sealed_predictions.py | frozen task/target summaries |
| Baseline Hallmark | 03_analysis/2026-10-19_figure4_v1_2_associations/run_associations.py | molecular.py |
| Target-anchor decomposition | 03_analysis/2026-10-29_target_anchor_decomposition/run_decomposition.py | executed TSV/JSON summaries; Supplementary Tables S8–S10 in manuscript_tables/ |
| Figure 2 including panel e | 03_analysis/2026-10-29_figure2_panel_e/draw_figure2_with_e.py | saved decomposition summaries; no new fits |
| Figure 3 and Supplementary Figure S1 | 03_analysis/2026-10-29_figure3_s1_label/draw_figure3.py | frozen prediction tables; no new models |

## Read-only figure replay

Run `python scripts/reproduce_figures.py --output-directory outputs/figure_reproduction` from the repository root. Figure replay consumes the included frozen pair, target and cell-level example tables. It creates output in an isolated workspace; it does not recompute residuals, fit prediction models, perform association screening, run the decomposition or run inferential tests. Release v0.2.0 draws Figure 2, including panel e, and Figure 3 / Supplementary Figure S1 with the 2026-10-29 scripts. Earlier drawing scripts remain in the archive. Arial was used in the original artwork; another system font can change layout.

The Figure 3 public source manifest is generated for the included, privacy-redacted publication inputs. Its original local manifest was broader and also covered editorial material. `provenance/EXPORT_MANIFEST.json` records original and exported file hashes; historical hash-linked execution manifests remain historical records and should not be represented as current public execution certificates.

## Full computation prerequisites

1. Retrieve exactly the GDSC2 AUC, PRISM Secondary AUC, DepMap 26Q1 Model/Chronos, DEMETER2 and RNA releases described in `data/EXTERNAL_INPUTS.json` and the archived run configurations. The eligibility pipeline additionally requires the original ChEMBL 36 source database/API artifacts.
2. Preserve compound identities, sample masks and task folds. The first code release supplies the chosen annotation universe and metadata, not a claim that all 168 unresolved compounds are biologically ineligible.
3. Set local input paths using `config.example.yaml`; historical generic `/path/to/...` constants require adaptation before executing the archived standalone runners. The configuration template documents the required inputs; it is not automatically wired into every historical runner.
4. Restore the original gates, auxiliary assets and provider data needed by each historical runner. Some full-run artifacts (raw matrices, expression scores, all residual vectors, all fit-parameter JSONs and bootstrap arrays) are not included in this Git repository. The historical runners retain checks for them. Do not delete assertions or relabel old gate reports as newly executed.
5. Use new output directories and record the actual environment. This repository upload performed no new scientific execution or clean-environment reproduction. Archived pass records describe the prior study runs only.

## Core APIs

`residuals.response_vector`, `ols_residual`, `partial_corr` and `spearman_pair` are extracted from the original implementation. `sharing.pair_table` and `target_estimands` preserve the exact pair and target definitions.

For prediction, call `prediction.configure_context(meta, gdsc2_auc, chronos, lineages)` with compound metadata indexed by compound_id and a target column, raw GDSC2 AUC columns keyed by compound_id, Chronos columns keyed by gene symbol and lineage indexed by ModelID. Then call the original `predict_only(task_row, training_label_series, task_cell_fold_table, benchmark=False)`. The function partitions fold==0 versus fold!=0, as in the accepted preflight; the production adapter maps each outer fold to that interface. Measured test-cell GDSC2 drug features are required. Retain original task definitions for reproducibility.

The molecular routines use HC3/t approximation, pair-level same-direction support and BY over exactly 300 hypotheses. Specificity remains descriptive. No exploratory explanatory-boundary analysis, BH replacement or TOP1 candidate-gene screen is included.
