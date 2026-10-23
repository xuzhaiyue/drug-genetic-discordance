# Data sources and redistribution

Provider releases and original recorded URLs/checksums are in `data/EXTERNAL_INPUTS.json`.

- GDSC: https://www.cancerrxgene.org/ ; the study uses the archived portal-harmonized GDSC2 AUC matrix and mapped DPC identities.
- DepMap Public 26Q1 Chronos, lineage and RNA: https://depmap.org/portal/download/ . Do not substitute the NextGen 3D resource or another release.
- PRISM Secondary dose-response parameters: https://depmap.org/repurposing/ . Primary LFC and secondary AUC are different endpoints.
- DEMETER2 RNAi dependency: provider source identified in the original input manifest.
- ChEMBL 36: https://www.ebi.ac.uk/chembl/ ; selected compound annotation and scaffold tables are derived from this release. Original provider terms and attribution apply to those data.
- MSigDB Human Hallmark 2026.1.Hs: https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp ; terms: https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp . Retrieve the GMT from the provider; it is not bundled here.

Hallmark reference: Liberzon A et al. The Molecular Signatures Database (MSigDB) hallmark gene set collection. Cell Systems. 2015;1(6):417–425. https://doi.org/10.1016/j.cels.2015.12.004 . Scores in this study are means of standardized expression, not GSEA enrichment scores.

No private patient data, raw drug-response/dependency/expression matrices or third-party FORGE source are bundled. The included scientific tables are study-derived summaries and metadata, with the Figure 1 example supplied as derived residual coordinates. This code-availability deposit does not override provider terms.
