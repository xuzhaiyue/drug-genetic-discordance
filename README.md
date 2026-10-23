# Drug genetic discordance

Analysis code, frozen summary results and figure source data for **Shared drug-response patterns persist after adjustment for measured target-gene dependency and lineage**.

This study examines drug-response similarity after adjustment for measured annotated-target dependency and lineage. It uses a systematically assessed GDSC2 collection and PRISM pharmacology with a shared DepMap Chronos reference.

## Repository contents

- `src/drug_genetic_discordance/`: original residualization, pair/target estimands, accepted train-only prediction fitter and molecular inference routines extracted from the executed implementations.
- `archive/project/`: curated executed analysis implementations, frozen compound eligibility, prediction task definitions, pharmacological masks, summary results and plotting programs, preserving the original relative project layout.
- `data/EXTERNAL_INPUTS.json`: original input release identifiers, provider URLs and available checksums. Raw response/dependency/expression matrices are not bundled.
- `scripts/reproduce_figures.py`: redraws Figures 1–3 and the same-drug benchmark from supplied frozen results in a new output workspace, without fitting models.
- `provenance/`: export hashes and original-to-public file mapping. Workstation paths are replaced with generic placeholders; those edits are documented.
- `docs/REPRODUCIBILITY.md`: entry points, external-data requirements and release scope.

## Study summary

The 286-compound eligibility assessment yielded 43 included, 75 excluded and 168 unresolved compounds. Eighteen focal compounds span six target groups. The target-median within-minus-cross residual correlation is 0.326 in GDSC2 and 0.296 in four qualifying PRISM groups. The response-assisted prediction experiment used 22 directed tasks and 990 recipient-model fits; its frozen scientific classification is **MODEST_OR_UNCERTAIN**. The fixed baseline Hallmark screen yielded **0/300 BY q ≤ 0.05**. Original numerical tables and negative findings are retained.

## Install and inspect

The original numerical environment was Python 3.9.6 with the pinned packages in `requirements.txt`. RDKit versions involved in chemistry qualification and later execution are recorded in the archived files; do not infer chemical metadata by regenerating it with a different release.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/reproduce_figures.py --output-directory outputs/figure_reproduction
```

The figure command creates an isolated copy of the included project capsule. It refuses an existing destination and never writes into the frozen source-data directory. Arial was used in the original artwork; another system font can change layout. The export command has not been independently rerun in a clean environment for this release.

## Raw-data computation

The extracted scientific functions can be imported without triggering an analysis. Full historical runners depend on raw input matrices, original gates and hash-linked assets. They are retained for methodological inspection; **they are not advertised as a single-command raw-data pipeline in this first code-availability release**. See `docs/REPRODUCIBILITY.md` before attempting historical execution. Do not run archived top-level runners in an evidence directory: some create files on import.

## Data rights and citation

Download source data from their providers under the applicable terms. ChEMBL-derived annotation/chemistry tables, MSigDB gene sets and upstream public datasets retain their providers' terms and attribution requirements; public access here does not relicense them. The full Hallmark GMT, raw matrices and third-party FORGE implementation are not redistributed. See `docs/DATA_SOURCES.md`.

Use `CITATION.cff` for the versioned code citation. Paper authorship/funding details are not inferred from the repository owner. A DOI has not yet been assigned. Code licensing is pending author confirmation; no permissive license is assumed in this initial deposit.
