#!/usr/bin/env python3
"""Independent PRISM replication of frozen Atlas v2.

Source locked by coverage audit: PRISM Secondary 20Q2 AUC (Gate A LEGAL; Gate B FAIL — MAPK1 absent).
Same PANEL, MIN_N=200, 26Q1 Chronos, lineage-adjusted E. No extra drugs. No NextGen.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_prism_coverage import PANEL, MULTI_CHEMO, MIN_N, COMPOUNDS, PRISM_DIR, URLS, brd_core, dpc_brds
from run_discordance_atlas_gdsc2 import (
    CONC_HIGH,
    CRISPR_PATH,
    MODEL_PATH,
    N_PERM,
    REPRO_HIGH,
    classify,
    load_chronos,
    ols_residual,
    partial_corr,
    response_vector,
    spearman_pair,
)

OUT = Path("/path/to/project/discordance")
COVERAGE = json.loads((OUT / "prism_coverage.json").read_text())


def load_secondary_auc(legal_drugs: set[str]) -> dict[str, pd.Series]:
    compounds = pd.read_csv(COMPOUNDS).set_index("CompoundID")
    dpc_to_brd = {dpc: dpc_brds(compounds.loc[dpc, "SampleIDs"]) for _, _, dpc in PANEL if dpc in compounds.index}
    sec = pd.read_csv(
        PRISM_DIR / URLS["secondary_curves"][0],
        usecols=["broad_id", "depmap_id", "auc", "name"],
        low_memory=False,
    )
    sec["core"] = sec["broad_id"].astype(str).map(brd_core)
    sec["ach"] = sec["depmap_id"].astype(str)
    sec["auc"] = pd.to_numeric(sec["auc"], errors="coerce")
    out: dict[str, pd.Series] = {}
    for name, _, dpc in PANEL:
        if name not in legal_drugs:
            continue
        brds = dpc_to_brd.get(dpc, [])
        sub = sec[sec["core"].isin(brds) & sec["ach"].str.startswith("ACH-")].dropna(subset=["auc"])
        if sub.empty:
            continue
        s = sub.groupby("ach")["auc"].mean()
        out[name] = s
    return out


def main() -> None:
    cov = COVERAGE["secondary_20Q2_auc"]
    if cov["verdict"] != "LEGAL":
        raise SystemExit("PRISM secondary is not LEGAL; do not run")
    legal_drugs = {r["drug"] for r in cov["drugs"] if r["legal"]}
    legal_multi = {t: ds for t, ds in cov["legal_multi_chemotype_targets"].items()}

    chronos = load_chronos()
    models = pd.read_csv(MODEL_PATH, usecols=["ModelID", "OncotreeLineage"])
    models["ModelID"] = models["ModelID"].astype(str)
    lin = pd.get_dummies(
        models.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"],
        prefix="lin",
        dummy_na=False,
    )
    auc_map = load_secondary_auc(legal_drugs)

    residuals: dict[str, pd.Series] = {}
    pair_rows: list[dict[str, Any]] = []
    drug_target = {name: t for name, t, _ in PANEL}

    for name, target, dpc in PANEL:
        if name not in auc_map:
            continue
        if target not in chronos.columns:
            continue
        cells = auc_map[name].index.intersection(chronos.index).intersection(lin.index)
        R = response_vector(auc_map[name].reindex(cells))
        D = -chronos.loc[cells, target]
        L = lin.reindex(cells)
        conc = partial_corr(R, D, L)
        E = ols_residual(R, D, L)
        n = int(E.notna().sum())
        if n < MIN_N:
            continue
        residuals[name] = E
        pair_rows.append(
            {
                "drug": name,
                "target": target,
                "compound_id": dpc,
                "n": n,
                "concordance_partial": round(conc, 3),
            }
        )
        print(f"{name}→{target} n={n} conc|lin={conc:.3f}")

    target_rows: list[dict[str, Any]] = []
    repro_pairs: list[dict[str, Any]] = []
    for target, drugs in MULTI_CHEMO.items():
        avail = [d for d in drugs if d in residuals]
        rhos = []
        for a, b in combinations(avail, 2):
            r, n = spearman_pair(residuals[a], residuals[b])
            repro_pairs.append({"target": target, "drug_a": a, "drug_b": b, "n": n, "rho": None if not np.isfinite(r) else round(r, 3)})
            if np.isfinite(r):
                rhos.append(r)
        concs = [p["concordance_partial"] for p in pair_rows if p["target"] == target]
        mean_conc = float(np.nanmean(concs)) if concs else float("nan")
        med_repro = float(np.median(rhos)) if rhos else float("nan")
        target_rows.append(
            {
                "target": target,
                "n_drugs": len(avail),
                "mean_concordance_partial": None if not np.isfinite(mean_conc) else round(mean_conc, 3),
                "median_chemotype_residual_rho": None if not np.isfinite(med_repro) else round(med_repro, 3),
                "atlas_class": classify(mean_conc, med_repro if np.isfinite(med_repro) else None),
                "coverage": "legal" if target in legal_multi else "BLOCKED_BY_COVERAGE",
            }
        )

    for target in sorted({p["target"] for p in pair_rows} - set(MULTI_CHEMO)):
        concs = [p["concordance_partial"] for p in pair_rows if p["target"] == target]
        mean_conc = float(np.mean(concs))
        target_rows.append(
            {
                "target": target,
                "n_drugs": len(concs),
                "mean_concordance_partial": round(mean_conc, 3),
                "median_chemotype_residual_rho": None,
                "atlas_class": "I" if mean_conc >= CONC_HIGH else "IV",
                "coverage": "n=1_rail",
            }
        )

    same_rhos, cross_rhos = [], []
    drugs = list(residuals.keys())
    for a, b in combinations(drugs, 2):
        r, _ = spearman_pair(residuals[a], residuals[b])
        if not np.isfinite(r):
            continue
        if drug_target[a] == drug_target[b]:
            same_rhos.append(r)
        else:
            cross_rhos.append(r)
    null_cross = {
        "median_same_target": float(np.median(same_rhos)) if same_rhos else None,
        "median_cross_target": float(np.median(cross_rhos)) if cross_rhos else None,
        "n_same_pairs": len(same_rhos),
        "n_cross_pairs": len(cross_rhos),
        "mw_p_same_gt_cross": float(stats.mannwhitneyu(same_rhos, cross_rhos, alternative="greater").pvalue)
        if same_rhos and cross_rhos
        else None,
    }

    rng = np.random.default_rng(20260830)
    e1, e2 = residuals.get("ERLOTINIB"), residuals.get("GEFITINIB")
    null_lin = None
    if e1 is not None and e2 is not None:
        common = e1.dropna().index.intersection(e2.dropna().index)
        obs, _ = spearman_pair(e1, e2)
        lineage = models.set_index("ModelID").loc[common, "OncotreeLineage"]
        nulls = []
        for _ in range(N_PERM):
            e1s = e1.copy()
            for lin_name in lineage.unique():
                idx = lineage.index[lineage == lin_name]
                vals = e1.loc[idx].dropna()
                if len(vals) < 2:
                    continue
                e1s.loc[vals.index] = rng.permutation(vals.to_numpy())
            r, _ = spearman_pair(e1s, e2)
            if np.isfinite(r):
                nulls.append(r)
        null_lin = {
            "drug_a": "ERLOTINIB",
            "drug_b": "GEFITINIB",
            "obs_rho": float(obs),
            "median_perm_rho": float(np.median(nulls)) if nulls else None,
            "p_obs_ge_perm": float((np.sum(np.array(nulls) >= obs) + 1) / (len(nulls) + 1)) if nulls else None,
        }

    same_gt_cross = bool(
        null_cross["median_same_target"] is not None
        and null_cross["median_cross_target"] is not None
        and null_cross["median_same_target"] > null_cross["median_cross_target"]
        and (null_cross["mw_p_same_gt_cross"] or 1) < 0.05
    )
    atlas = {
        "module": "II-B PRISM independent replication (secondary 20Q2 AUC + 26Q1 Chronos)",
        "prereg": "PRISM_REPLICATION_PREREG.md",
        "source": "secondary_20Q2_auc",
        "metric": "AUC (mean across screens if duplicated)",
        "gate_B_four_poles": False,
        "mapk1": "BLOCKED_BY_COVERAGE on preferred AUC source; not substituted",
        "pairs": pair_rows,
        "targets": target_rows,
        "chemotype_pairs": repro_pairs,
        "nulls": {"cross_target": null_cross, "within_lineage_EGFR": null_lin},
        "primary_claim_same_gt_cross": same_gt_cross,
    }
    (OUT / "atlas_prism.json").write_text(json.dumps(atlas, indent=2))
    pd.DataFrame(pair_rows).to_csv(OUT / "atlas_prism_pairs.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(OUT / "atlas_prism_targets.tsv", sep="\t", index=False)
    print(json.dumps({"null_cross": null_cross, "same_gt_cross": same_gt_cross, "null_lin": null_lin, "targets": target_rows}, indent=2))
    print("wrote", OUT / "atlas_prism.json")


if __name__ == "__main__":
    main()
