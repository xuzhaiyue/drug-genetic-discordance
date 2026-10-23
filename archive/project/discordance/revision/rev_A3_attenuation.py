#!/usr/bin/env python3
"""Stage 2A A3 Chronos attenuation, retest reliability and replacement-D."""
from __future__ import annotations

import itertools
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from rev_common import (REVISION, compute_E, linear_ols, load_resource,
                        package_versions, pair_table, stage_metadata, summary)

ROOT = REVISION.parents[1]
DEPMAP = Path("/path/to/depmap")
import sys
sys.path.insert(0, str(ROOT / "discordance"))
import run_discordance_atlas_gdsc2 as gdsc  # noqa: E402

SEED = 20260927
N_SIM = 2000
N_BOOT = 1000
TARGETS = ("EGFR", "AKT1", "MET", "MAPK1")
PANEL_TARGETS = sorted({target for _, target, _ in gdsc.PANEL})
WRONG_GENES_EGFR = [g for g in PANEL_TARGETS if g != "EGFR"]
GENES = sorted(set(TARGETS) | set(WRONG_GENES_EGFR))

SCREEN_PATH = ROOT / "raw" / "revision" / "ScreenGeneEffect.csv"
SEQUENCE_PATH = ROOT / "raw" / "revision" / "ScreenSequenceMap.csv"
D2_PATH = ROOT / "raw" / "revision" / "D2_combined_gene_dep_scores.csv"
MODEL_PATH = DEPMAP / "26Q1" / "Model.csv"
EXPR_PATH = DEPMAP / "26Q1" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
FIG_DIR = ROOT / "revision_figs"


def gene_column(columns, gene: str) -> str:
    hits = [c for c in columns if str(c).split(" ")[0].upper() == gene.upper()]
    if not hits:
        raise KeyError(f"No matrix column found for {gene}")
    return hits[0]


def atlas_lambda_star(atlas: dict) -> list[dict]:
    pairs = atlas["chemotype_pairs"]
    concordance = {row["drug"]: row.get("concordance_partial") for row in atlas["pairs"]}
    output = []
    for row in pairs:
        ra = float(concordance[row["drug_a"]])
        rb = float(concordance[row["drug_b"]])
        rho = float(row["rho"])
        ab = ra * rb
        if ab <= 0:
            lam, reason = None, "r_a*r_b<=0"
        else:
            den = ab + rho * np.sqrt(max(0.0, (1 - ra**2) * (1 - rb**2)))
            if den <= 0:
                lam, reason = None, "nonpositive denominator"
            else:
                lam, reason = float(ab / den), None
        output.append({"target": row["target"], "drug_a": row["drug_a"],
                       "drug_b": row["drug_b"], "r_a": ra, "r_b": rb,
                       "rho_frozen": rho, "n": int(row["n"]),
                       "lambda_star": lam, "na_reason": reason})
    return output


def monte_carlo_pair(resource: dict, E: pd.DataFrame, row: dict,
                     n_sim: int = N_SIM, seed: int = SEED) -> dict:
    a, b = row["drug_a"], row["drug_b"]
    lam = row["lambda_star"]
    ra, rb, rho_obs = row["r_a"], row["r_b"], row["rho_frozen"]
    if lam is None or lam <= 0 or lam > 1 or lam < max(ra**2, rb**2):
        return {"n_sim": 0, "simulated_mean_rho": None, "observed_rho": rho_obs,
                "absolute_difference": None, "analytic_approx_poor": None,
                "status": "invalid_lambda_for_simulation"}
    cells = E.index[E[a].notna() & E[b].notna()]
    lineage = resource["lineage"].reindex(cells)
    n = len(cells)
    ca, cb = ra / np.sqrt(lam), rb / np.sqrt(lam)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        t = rng.normal(size=n)
        da = np.sqrt(lam) * t + np.sqrt(1 - lam) * rng.normal(size=n)
        db = np.sqrt(lam) * t + np.sqrt(1 - lam) * rng.normal(size=n)
        ya = ca * t + np.sqrt(max(0.0, 1 - ca**2)) * rng.normal(size=n)
        yb = cb * t + np.sqrt(max(0.0, 1 - cb**2)) * rng.normal(size=n)
        ea = linear_ols(pd.Series(ya, index=cells), pd.Series(da, index=cells), lineage)
        eb = linear_ols(pd.Series(yb, index=cells), pd.Series(db, index=cells), lineage)
        rhos[i], _ = gdsc.spearman_pair(ea, eb)
    mean = float(np.nanmean(rhos))
    diff = abs(mean - rho_obs)
    return {"n_sim": n_sim, "n_cells": int(n), "simulated_mean_rho": mean,
            "simulated_sd_rho": float(np.nanstd(rhos, ddof=1)),
            "observed_rho": rho_obs, "absolute_difference": float(diff),
            "analytic_approx_poor": bool(diff > 0.03), "status": "ok",
            "lambda_star_used": lam}


def solve_simulated_lambda(resource: dict, E: pd.DataFrame, row: dict,
                           seed: int, iterations: int = 12) -> dict:
    """Bisection λ* using the same frozen OLS residualization and common random draws."""
    a, b = row["drug_a"], row["drug_b"]
    ra, rb, target_rho = row["r_a"], row["r_b"], row["rho_frozen"]
    cells = E.index[E[a].notna() & E[b].notna()]
    lineage = resource["lineage"].reindex(cells)
    n = len(cells)
    lo, hi = max(ra**2, rb**2) + 1e-8, 1.0

    def objective(lam: float) -> float:
        ca, cb = ra / np.sqrt(lam), rb / np.sqrt(lam)
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(N_SIM):
            t = rng.normal(size=n)
            da = np.sqrt(lam) * t + np.sqrt(1 - lam) * rng.normal(size=n)
            db = np.sqrt(lam) * t + np.sqrt(1 - lam) * rng.normal(size=n)
            ya = ca * t + np.sqrt(max(0.0, 1 - ca**2)) * rng.normal(size=n)
            yb = cb * t + np.sqrt(max(0.0, 1 - cb**2)) * rng.normal(size=n)
            ea = linear_ols(pd.Series(ya, index=cells), pd.Series(da, index=cells), lineage)
            eb = linear_ols(pd.Series(yb, index=cells), pd.Series(db, index=cells), lineage)
            rho, _ = gdsc.spearman_pair(ea, eb)
            if np.isfinite(rho):
                values.append(rho)
        return float(np.mean(values)) - target_rho

    flo, fhi = objective(lo), objective(hi)
    if flo == 0:
        return {"lambda_star_simulated": float(lo), "iterations": 0, "bracketed": True,
                "objective_at_root": float(flo), "n_sim_per_evaluation": N_SIM}
    if fhi == 0:
        return {"lambda_star_simulated": float(hi), "iterations": 0, "bracketed": True,
                "objective_at_root": float(fhi), "n_sim_per_evaluation": N_SIM}
    if flo * fhi > 0:
        best = (lo, flo) if abs(flo) < abs(fhi) else (hi, fhi)
        return {"lambda_star_simulated": float(best[0]), "iterations": 0, "bracketed": False,
                "objective_at_endpoint": float(best[1]), "n_sim_per_evaluation": N_SIM}
    mid, fmid = (lo + hi) / 2, float("nan")
    for iteration in range(1, iterations + 1):
        mid = (lo + hi) / 2
        fmid = objective(mid)
        if abs(fmid) < 0.001:
            break
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return {"lambda_star_simulated": float(mid), "iterations": iteration,
            "bracketed": True, "objective_at_root": float(fmid),
            "n_sim_per_evaluation": N_SIM}


def choose_retest_pairs(sequence: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keep = (sequence["PassesQC"].astype(str).str.lower().eq("true") &
            sequence["ScreenType"].eq("2DS") &
            sequence["DrugTreated"].astype(str).str.lower().eq("false") &
            sequence["IsEngineered"].astype(str).str.lower().eq("false"))
    eligible = sequence.loc[keep].dropna(subset=["ModelID", "ScreenID", "Library"]).copy()
    eligible["ModelID"] = eligible["ModelID"].astype(str)
    eligible["ScreenID"] = eligible["ScreenID"].astype(str)
    eligible["Library"] = eligible["Library"].astype(str)
    screen_level = eligible.drop_duplicates(["ModelID", "ScreenID", "Library"])
    selected = {}
    for model, group in screen_level.groupby("ModelID", sort=True):
        items = sorted(group[["ScreenID", "Library"]].drop_duplicates().itertuples(index=False, name=None))
        if len(items) < 2:
            continue
        pairs = list(itertools.combinations(items, 2))
        cross_library = [p for p in pairs if p[0][1] != p[1][1]]
        chosen = sorted(cross_library or pairs, key=lambda p: (p[0][0], p[1][0]))[0]
        selected[model] = (chosen[0][0], chosen[1][0])
    return selected, eligible


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def bootstrap_corr(x, y, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = pearson(x[idx], y[idx])
        if np.isfinite(val):
            vals.append(val)
    return ((float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))
            if vals else (float("nan"), float("nan")))


def retest_reliability(data: dict, atlas_cohort: set[str]) -> tuple[dict, dict]:
    seq = pd.read_csv(SEQUENCE_PATH, dtype=str)
    selected, eligible = choose_retest_pairs(seq)
    # The 9 frozen EGFR wrong-gene alternatives are the other annotated panel targets.
    screen_header = pd.read_csv(SCREEN_PATH, nrows=0).columns
    index_col = str(screen_header[0])
    needed_cols = [gene_column(screen_header, g) for g in GENES]
    screen = pd.read_csv(SCREEN_PATH, index_col=index_col,
                         usecols=[index_col] + needed_cols, low_memory=False)
    screen.index = screen.index.astype(str)
    screen = screen.apply(pd.to_numeric, errors="coerce")
    if screen.index.has_duplicates:
        screen = screen.groupby(level=0, sort=False).mean()
    eligible_screen_counts = (eligible.loc[eligible["ExcludeFromCRISPRCombined"].astype(str).str.lower().eq("false")]
                              .drop_duplicates(["ModelID", "ScreenID"])
                              .groupby("ModelID")["ScreenID"].nunique())
    model = pd.read_csv(MODEL_PATH, usecols=["ModelID"])
    chronos_genes = data["chronos"]
    results = {}
    cohort_order = set(map(str, atlas_cohort))
    for gene in GENES:
        col = gene_column(screen.columns, gene)
        rows = []
        for model_id, (sid1, sid2) in selected.items():
            if sid1 not in screen.index or sid2 not in screen.index:
                continue
            x, y = screen.loc[sid1, col], screen.loc[sid2, col]
            if np.isfinite(x) and np.isfinite(y):
                rows.append({"ModelID": model_id, "screen1": sid1, "screen2": sid2,
                             "score1": float(x), "score2": float(y)})
        d = pd.DataFrame(rows)
        retest_n = len(d)
        lam = pearson(d.get("score1", []), d.get("score2", [])) if retest_n >= 3 else float("nan")
        ci = bootstrap_corr(d["score1"], d["score2"], seed=SEED + GENES.index(gene)) if retest_n >= 3 else (None, None)
        dco = d[d["ModelID"].astype(str).isin(cohort_order)] if retest_n else d
        cohort_n = len(dco)
        lam_cohort = pearson(dco.get("score1", []), dco.get("score2", [])) if cohort_n >= 3 else float("nan")
        if cohort_n < 50:
            lam_cohort = float("nan")
        cohort_models = dco["ModelID"].astype(str).tolist() if cohort_n else []
        ks = [int(eligible_screen_counts.get(mid, 0)) for mid in cohort_models]
        sb = [k * lam_cohort / (1 + (k - 1) * lam_cohort)
              for k in ks if k >= 1 and np.isfinite(lam_cohort) and 1 + (k - 1) * lam_cohort != 0]
        lambda_combined = float(np.mean(sb)) if sb else None
        retest_sd = float(np.std(np.r_[d["score1"].to_numpy(), d["score2"].to_numpy()], ddof=1)) if retest_n > 1 else None
        target_chronos = gene_column(chronos_genes.columns, gene)
        cohort_values = pd.to_numeric(chronos_genes[target_chronos].reindex(cohort_models), errors="coerce").dropna()
        cohort_sd = float(cohort_values.std(ddof=1)) if len(cohort_values) > 1 else None
        ratio = (max(retest_sd, cohort_sd) / min(retest_sd, cohort_sd)
                 if retest_sd is not None and cohort_sd is not None and min(retest_sd, cohort_sd) > 0 else None)
        results[gene] = {
            "role": "true_target" if gene in TARGETS else "wrong_gene_reference_for_EGFR",
            "lambda_single": lam if np.isfinite(lam) else None,
            "lambda_single_ci95": list(ci) if ci[0] is not None else None,
            "n_retest_models": int(retest_n),
            "lambda_single_atlas_cohort": lam_cohort if np.isfinite(lam_cohort) else None,
            "n_atlas_cohort_models": int(cohort_n),
            "lambda_combined_atlas_cohort": lambda_combined,
            "mean_k_atlas_cohort": float(np.mean([k for k in ks if k > 0])) if any(k > 0 for k in ks) else None,
            "sd_retest_pooled": retest_sd,
            "sd_atlas_cohort_chronos": cohort_sd,
            "sd_ratio_max_over_min": ratio,
            "sd_range_flag_gt_1_5x": bool(ratio > 1.5) if ratio is not None else None,
        }
    diagnostics = {"n_eligible_screens": int(eligible["ScreenID"].nunique()),
                   "n_models_with_two_screens": int(len(selected)),
                   "n_models_used_by_gene": {g: results[g]["n_retest_models"] for g in GENES},
                   "selected_screen_pairs": selected,
                   "wrong_gene_reference_for_EGFR": WRONG_GENES_EGFR}
    return results, diagnostics


def map_external_matrix(path: Path, model_path: Path, genes: list[str],
                        delimiter=",") -> tuple[dict[str, pd.Series], dict]:
    mat = pd.read_csv(path, index_col=0, low_memory=False)
    model = pd.read_csv(model_path, usecols=["ModelID", "CCLEName"], dtype=str).dropna(subset=["CCLEName"])
    by_ccle = {}
    duplicate_ccle = {}
    for ccle, group in model.groupby("CCLEName", sort=True):
        ids = sorted(set(group["ModelID"].astype(str)))
        by_ccle[str(ccle)] = ids[0]
        if len(ids) > 1:
            duplicate_ccle[str(ccle)] = ids
    mapped_cols = {str(c): by_ccle[str(c)] for c in mat.columns if str(c) in by_ccle}
    unmapped = [str(c) for c in mat.columns if str(c) not in by_ccle]
    out = {}
    for gene in genes:
        rowname = gene_column(mat.index, gene)
        vals = pd.to_numeric(mat.loc[rowname], errors="coerce")
        s = pd.Series({mapped_cols[str(c)]: float(vals[c]) for c in mat.columns if str(c) in mapped_cols})
        s = s.groupby(level=0).mean()
        out[gene] = s
    return out, {"n_columns": int(len(mat.columns)), "n_mapped_columns": int(len(mapped_cols)),
                 "n_unmapped_columns": int(len(unmapped)), "unmapped_columns": unmapped,
                 "duplicate_ccle_modelid_mappings": duplicate_ccle}


def load_target_expression(path: Path, genes: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    header = pd.read_csv(path, nrows=0).columns
    cols = {g: gene_column(header, g) for g in genes}
    usecols = list(dict.fromkeys(["ModelID", "IsDefaultEntryForModel", *cols.values()]))
    expr = pd.read_csv(path, usecols=usecols, low_memory=False)
    expr["ModelID"] = expr["ModelID"].astype(str)
    default = expr["IsDefaultEntryForModel"].astype(str).str.strip().str.lower().isin(
        {"true", "yes", "1"}
    )
    expr = expr.loc[default].dropna(subset=["ModelID"])
    out = {}
    for gene, col in cols.items():
        temp = expr[["ModelID", col]].copy()
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        temp = temp.groupby("ModelID", sort=False)[col].mean()
        out[gene] = temp
    return out, list(cols.values())


def ols_residual_multi(R: pd.Series, predictors: pd.DataFrame,
                       L: pd.DataFrame, min_n: int = 200) -> tuple[pd.Series, int]:
    df = pd.concat([R.rename("R"), predictors, L], axis=1).dropna()
    out = pd.Series(np.nan, index=R.index, dtype=float)
    if len(df) < min_n:
        return out, int(len(df))
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df[predictors.columns], df[L.columns]], axis=1).astype(float)
    fit = sm.OLS(df["R"].astype(float), X).fit()
    out.loc[df.index] = fit.resid.to_numpy()
    return out, int(len(df))


def replacement_analysis(resource: dict, D2: dict[str, pd.Series],
                          expression: dict[str, pd.Series],
                          E_chronos: pd.DataFrame) -> tuple[dict, list[dict]]:
    results, pair_output = {}, []
    for method in ("c1_D2_only", "c2_Chronos_D2_RNA"):
        new_residuals, baseline = {}, {}
        n_by_drug, excluded = {}, []
        for meta in resource["panel"]:
            drug, target = meta["drug"], meta["target"]
            R = resource["R_source"][drug]
            L = resource["lineage"]
            d2 = D2.get(target, pd.Series(dtype=float))
            dch = resource["D_source"][drug]
            if method == "c1_D2_only":
                predictors = pd.DataFrame({"D2": d2})
            else:
                mRNA = expression.get(target, pd.Series(dtype=float))
                predictors = pd.DataFrame({"Chronos": dch, "D2": d2, "target_mRNA": mRNA})
            e_new, n = ols_residual_multi(R, predictors, L, min_n=resource["min_n"])
            n_by_drug[drug] = n
            if n < resource["min_n"]:
                excluded.append({"drug": drug, "target": target, "n": n,
                                 "reason": "n<MIN_N=200"})
                new_residuals[drug] = pd.Series(np.nan, index=R.index)
                baseline[drug] = pd.Series(np.nan, index=R.index)
                continue
            # Denominator is the Chronos-only frozen linear residual on this exact subset.
            valid = e_new.notna()
            idx = R.index[R.index.isin(e_new.index[valid])]
            base = linear_ols(R.reindex(idx), dch.reindex(idx), L.reindex(idx))
            new_residuals[drug] = e_new
            baseline[drug] = base
        new_E, base_E = pd.DataFrame(new_residuals), pd.DataFrame(baseline)
        pnew = pair_table(new_E, resource["panel"])
        pbase = pair_table(base_E, resource["panel"])
        pnew = pnew.rename(columns={"n": "n_new", "rho": "rho_new"})
        pbase = pbase.rename(columns={"n": "n_baseline", "rho": "rho_baseline"})
        keys = ["drug_a", "drug_b", "target_a", "target_b", "same_target"]
        joined = pnew.merge(pbase, on=keys, how="outer")
        joined["resource"] = resource["resource"]
        joined["method"] = method
        joined["pair_retention"] = joined["rho_new"] / joined["rho_baseline"]
        pair_output.extend(joined.to_dict(orient="records"))
        sn, sb = summary(pnew.rename(columns={"n_new": "n", "rho_new": "rho"})), summary(
            pbase.rename(columns={"n_baseline": "n", "rho_baseline": "rho"}))
        target_retention = {}
        for target in sorted(set(x["target"] for x in resource["panel"])):
            qn = pnew.loc[pnew.same_target & pnew.target_a.eq(target), "rho_new"]
            qb = pbase.loc[pbase.same_target & pbase.target_a.eq(target), "rho_baseline"]
            med_new = float(qn.median()) if len(qn) else None
            med_base = float(qb.median()) if len(qb) else None
            target_retention[target] = {
                "n_pairs_new": int(len(qn)), "same_median_new": med_new,
                "same_median_baseline_same_subset": med_base,
                "retention": med_new / med_base if med_new is not None and med_base not in (None, 0) else None,
            }
        egfr_ratios = joined.loc[joined.same_target & joined.target_a.eq("EGFR"), "pair_retention"].dropna()
        # Enumerate all unavailable panel pairs so min-N exclusions cannot disappear silently.
        available_pair_keys = set(zip(pnew.drug_a, pnew.drug_b))
        absent_pairs = []
        drugs = [x["drug"] for x in resource["panel"]]
        tmap = {x["drug"]: x["target"] for x in resource["panel"]}
        for a, b in itertools.combinations(drugs, 2):
            if (a, b) not in available_pair_keys:
                absent_pairs.append({"drug_a": a, "drug_b": b, "target_a": tmap[a], "target_b": tmap[b],
                                     "reason": "at least one residual missing or pair n<30"})
        results[method] = {
            "global_new": sn, "global_chronos_only_same_subset": sb,
            "target_retention": target_retention,
            "egfr_pair_median_retention": float(egfr_ratios.median()) if len(egfr_ratios) else None,
            "n_by_drug": n_by_drug, "excluded_drugs": excluded,
            "missing_pairs": absent_pairs,
        }
    return results, pair_output


def replacement_only() -> None:
    """Repair A3c only while reusing already completed A3a/A3b outputs."""
    out_path = REVISION / "rev_A3_attenuation.json"
    output = json.loads(out_path.read_text())
    resources, E_by_resource = {}, {}
    inputs = [REVISION / "rev_common.py", Path(__file__).resolve(), out_path,
              REVISION / "rev_common_regression.json", REVISION / "rev_A1a_drugperm.json",
              REVISION / "rev_A2_nonlinear.json", SCREEN_PATH, SEQUENCE_PATH, D2_PATH,
              MODEL_PATH, EXPR_PATH, ROOT / "data" / "INPUT_MANIFEST.json",
              ROOT / "data" / "INPUT_MANIFEST.sha256"]
    for name in ("GDSC2", "PRISM"):
        data = load_resource(name)
        resources[name] = data
        E_by_resource[name] = compute_E(data)
        inputs.extend(data["input_paths"])
    d2, d2_map_diag = map_external_matrix(D2_PATH, MODEL_PATH, PANEL_TARGETS)
    expr, expr_cols = load_target_expression(EXPR_PATH, PANEL_TARGETS)
    output["replacement_D_mapping"] = {
        "DEMETER2": d2_map_diag,
        "target_mRNA_columns": {g: c for g, c in zip(PANEL_TARGETS, expr_cols)},
        "RNA_manifest_sha256": next((x.split()[0] for x in (ROOT / "data" / "INPUT_MANIFEST.sha256").read_text().splitlines()
                                      if "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv" in x), None),
        "expression_default_entry_filter": "IsDefaultEntryForModel in {Yes, True, 1}",
    }
    replace_pairs = []
    for name in ("GDSC2", "PRISM"):
        rep, rows = replacement_analysis(resources[name], d2, expr, E_by_resource[name])
        output["replacement_D"][name] = rep
        replace_pairs.extend(rows)
    rel = output["retest_reliability"]["genes"]
    gdsc2 = output["lambda_star_by_resource"]["GDSC2"]
    egfr_lam = [x.get("lambda_for_decision", x["lambda_star"]) for x in gdsc2
                if x["target"] == "EGFR" and x.get("lambda_for_decision", x["lambda_star"]) is not None]
    akt_lam = [x["lambda_star"] for x in gdsc2 if x["target"] == "AKT1" and x["lambda_star"] is not None]
    egfr_cut = float(np.median(egfr_lam)) if egfr_lam else None
    akt_cut = float(np.median(akt_lam)) if akt_lam else None
    egfr_c1 = output["replacement_D"]["GDSC2"]["c1_D2_only"]["egfr_pair_median_retention"]
    egfr_c2 = output["replacement_D"]["GDSC2"]["c2_Chronos_D2_RNA"]["egfr_pair_median_retention"]
    egfr_single, akt_single = rel.get("EGFR", {}).get("lambda_single"), rel.get("AKT1", {}).get("lambda_single")
    egfr_pass = (egfr_single is not None and egfr_cut is not None and egfr_single > egfr_cut + 0.1 and
                 egfr_c1 is not None and egfr_c2 is not None and egfr_c1 >= 0.70 and egfr_c2 >= 0.70)
    output["decision"] = {
        "EGFR_GDSC2_excludes_attenuation_artifact": bool(egfr_pass),
        "EGFR_lambda_single": egfr_single, "EGFR_median_lambda_star": egfr_cut,
        "EGFR_c1_pair_median_retention": egfr_c1, "EGFR_c2_pair_median_retention": egfr_c2,
        "EGFR_triggers_core_claim_rewrite_stop_stage3": not bool(egfr_pass),
        "AKT1_lambda_single": akt_single, "AKT1_median_lambda_star": akt_cut,
        "AKT1_downgrade": bool(akt_single is not None and akt_cut is not None and akt_single <= akt_cut),
        "PRISM_reported_separately": True,
    }
    inputs.extend([ROOT / "discordance" / "atlas_gdsc2.json", ROOT / "discordance" / "atlas_prism.json",
                   ROOT / "discordance" / "run_discordance_atlas_gdsc2.py",
                   ROOT / "discordance" / "run_discordance_atlas_prism.py",
                   ROOT / "discordance" / "audit_prism_coverage.py"])
    output.update(stage_metadata(list(dict.fromkeys(inputs)), seed=SEED, n_perm=0))
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(replace_pairs).to_csv(REVISION / "rev_A3_replaceD_pairs.tsv", sep="\t", index=False)
    print(json.dumps({"decision": output["decision"],
                      "replacement": {r: {m: output["replacement_D"][r][m]["egfr_pair_median_retention"]
                                          for m in ("c1_D2_only", "c2_Chronos_D2_RNA")}
                                     for r in ("GDSC2", "PRISM")}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement-only", action="store_true",
                        help="Recompute A3c while reusing the stored A3a/A3b results.")
    args = parser.parse_args()
    if args.replacement_only:
        replacement_only()
        return
    frozen = {r: json.loads((ROOT / "discordance" / f"atlas_{r.lower()}.json").read_text())
              for r in ("GDSC2", "PRISM")}
    output = {"analysis": "A3 Chronos measurement-error attenuation",
              "lambda_star_by_resource": {}, "monte_carlo_egfr": {},
              "retest_reliability": {}, "replacement_D": {}, "decision": {}}
    lambda_rows = []
    inputs = [REVISION / "rev_common.py", Path(__file__).resolve(),
              REVISION / "rev_common_regression.json", REVISION / "rev_A1a_drugperm.json",
              REVISION / "rev_A2_nonlinear.json", SCREEN_PATH, SEQUENCE_PATH, D2_PATH,
              MODEL_PATH, EXPR_PATH, ROOT / "data" / "INPUT_MANIFEST.json"]
    resources = {}
    E_by_resource = {}
    for name in ("GDSC2", "PRISM"):
        data = load_resource(name)
        resources[name] = data
        E = compute_E(data)
        E_by_resource[name] = E
        lam_rows = atlas_lambda_star(frozen[name])
        output["lambda_star_by_resource"][name] = lam_rows
        for row in lam_rows:
            if row["lambda_star"] is not None:
                lambda_rows.append({"resource": name, "kind": "lambda_star", **row})
        mc_results = []
        egfr_rows = [x for x in lam_rows if x["target"] == "EGFR"]
        for i, row in enumerate(egfr_rows):
            mc = monte_carlo_pair(data, E, row, seed=SEED + i + (0 if name == "GDSC2" else 10000))
            mc_results.append({**row, **mc})
        output["monte_carlo_egfr"][name] = mc_results

    # Reliability retest from screen-level 26Q1 primary gene effects.
    atlas_cohort = set(map(str, resources["GDSC2"]["atlas_valid_cells"]))
    reliability, reliability_diag = retest_reliability(resources["GDSC2"], atlas_cohort)
    output["retest_reliability"] = {"genes": reliability, "selection_diagnostics": reliability_diag,
                                    "bootstrap_n": N_BOOT, "bootstrap_seed": SEED,
                                    "wrong_gene_reference_set": WRONG_GENES_EGFR}
    for gene, values in reliability.items():
        lambda_rows.append({"resource": "DepMap26Q1", "kind": "lambda_single", "target": gene,
                            "lambda_single": values["lambda_single"],
                            "ci_low": values["lambda_single_ci95"][0] if values["lambda_single_ci95"] else None,
                            "ci_high": values["lambda_single_ci95"][1] if values["lambda_single_ci95"] else None,
                            "n": values["n_retest_models"]})

    # Replacement D: map DEMETER2 columns through the frozen 26Q1 CCLEName field.
    d2, d2_map_diag = map_external_matrix(D2_PATH, MODEL_PATH, PANEL_TARGETS)
    expr, expr_cols = load_target_expression(EXPR_PATH, PANEL_TARGETS)
    output["replacement_D_mapping"] = {
        "DEMETER2": d2_map_diag,
        "target_mRNA_columns": {g: c for g, c in zip(PANEL_TARGETS, expr_cols)},
        "RNA_manifest_sha256": next((x.split()[0] for x in (ROOT / "data" / "INPUT_MANIFEST.sha256").read_text().splitlines()
                                      if "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv" in x), None),
    }
    replace_pairs = []
    for name in ("GDSC2", "PRISM"):
        rep, rows = replacement_analysis(resources[name], d2, expr, E_by_resource[name])
        output["replacement_D"][name] = rep
        replace_pairs.extend(rows)
    egfr_lam = [x["lambda_star"] for x in output["lambda_star_by_resource"]["GDSC2"]
                if x["target"] == "EGFR" and x["lambda_star"] is not None]
    akt_lam = [x["lambda_star"] for x in output["lambda_star_by_resource"]["GDSC2"]
               if x["target"] == "AKT1" and x["lambda_star"] is not None]
    egfr_single = reliability.get("EGFR", {}).get("lambda_single")
    akt_single = reliability.get("AKT1", {}).get("lambda_single")
    egfr_r_c1 = output["replacement_D"]["GDSC2"]["c1_D2_only"]["egfr_pair_median_retention"]
    egfr_r_c2 = output["replacement_D"]["GDSC2"]["c2_Chronos_D2_RNA"]["egfr_pair_median_retention"]
    # If the analytic Spearman approximation is poor, resolve λ* by bisection as
    # prespecified, then use that simulated value for the EGFR decision statistic.
    for name in ("GDSC2", "PRISM"):
        data = resources[name]
        E = E_by_resource[name]
        for i, row in enumerate([x for x in output["lambda_star_by_resource"][name] if x["target"] == "EGFR"]):
            mc_row = output["monte_carlo_egfr"][name][i]
            if mc_row.get("analytic_approx_poor") is True:
                solved = solve_simulated_lambda(data, E, row,
                                                seed=SEED + i + (0 if name == "GDSC2" else 10000))
                mc_row.update(solved)
                row["lambda_star_simulated"] = solved["lambda_star_simulated"]
                row["lambda_for_decision"] = solved["lambda_star_simulated"]
            else:
                row["lambda_for_decision"] = row["lambda_star"]
    egfr_lam_decision = [x.get("lambda_for_decision", x["lambda_star"])
                         for x in output["lambda_star_by_resource"]["GDSC2"]
                         if x["target"] == "EGFR" and x.get("lambda_for_decision", x["lambda_star"]) is not None]
    egfr_lam = egfr_lam_decision
    egfr_cut = float(np.median(egfr_lam)) if egfr_lam else None
    akt_cut = float(np.median(akt_lam)) if akt_lam else None
    egfr_pass = (egfr_single is not None and egfr_cut is not None and
                 egfr_single > egfr_cut + 0.1 and egfr_r_c1 is not None and egfr_r_c2 is not None and
                 egfr_r_c1 >= 0.70 and egfr_r_c2 >= 0.70)
    akt_downgrade = akt_single is not None and akt_cut is not None and akt_single <= akt_cut
    output["decision"] = {
        "EGFR_GDSC2_excludes_attenuation_artifact": bool(egfr_pass),
        "EGFR_lambda_single": egfr_single, "EGFR_median_lambda_star": egfr_cut,
        "EGFR_c1_pair_median_retention": egfr_r_c1,
        "EGFR_c2_pair_median_retention": egfr_r_c2,
        "EGFR_triggers_core_claim_rewrite_stop_stage3": not bool(egfr_pass),
        "AKT1_lambda_single": akt_single, "AKT1_median_lambda_star": akt_cut,
        "AKT1_downgrade": bool(akt_downgrade),
        "PRISM_reported_separately": True,
    }
    # Hash only files used in the added A3 branches; Stage 1 atlas hashes are reused.
    inputs.extend([
        ROOT / "discordance" / "atlas_gdsc2.json", ROOT / "discordance" / "atlas_prism.json",
        ROOT / "discordance" / "run_discordance_atlas_gdsc2.py",
        ROOT / "discordance" / "run_discordance_atlas_prism.py",
        ROOT / "discordance" / "audit_prism_coverage.py",
        ROOT / "data" / "INPUT_MANIFEST.sha256",
    ])
    output.update(stage_metadata(list(dict.fromkeys(inputs)), seed=SEED, n_perm=0))
    (REVISION / "rev_A3_attenuation.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(lambda_rows).to_csv(REVISION / "rev_A3_lambda_by_gene.tsv", sep="\t", index=False)
    pd.DataFrame(replace_pairs).to_csv(REVISION / "rev_A3_replaceD_pairs.tsv", sep="\t", index=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    genes_plot = GENES
    y = np.arange(len(genes_plot))
    for i, gene in enumerate(genes_plot):
        val = reliability.get(gene, {})
        lam = val.get("lambda_single")
        ci = val.get("lambda_single_ci95")
        if lam is not None:
            low, high = ci if ci else (lam, lam)
            ax.errorbar(lam, i, xerr=[[lam - low], [high - lam]], fmt="o", color="#1f77b4", capsize=2)
        for j, target in enumerate(("EGFR", "AKT1", "MET", "MAPK1")):
            group = [x["lambda_star"] for x in output["lambda_star_by_resource"]["GDSC2"]
                     if x["target"] == target and x["lambda_star"] is not None]
            if group:
                yy = i + (j - 1.5) * .08
                ax.plot([min(group), max(group)], [yy, yy], color=f"C{j+1}", alpha=.8, linewidth=2)
                ax.plot(np.median(group), yy, marker="|", color=f"C{j+1}")
    ax.set_yticks(y, genes_plot)
    ax.set_xlabel("Reliability / analytic λ* bound")
    ax.set_title("Chronos screen retest reliability and same-target λ* ranges (GDSC2)")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A3_lambda.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"decision": output["decision"],
                      "n_retest_models": reliability_diag["n_models_with_two_screens"],
                      "lambda_retest": {g: {"lambda_single": v["lambda_single"],
                                            "ci": v["lambda_single_ci95"],
                                            "lambda_cohort": v["lambda_single_atlas_cohort"],
                                            "n_cohort": v["n_atlas_cohort_models"]}
                                          for g, v in reliability.items()},
                      "replacement": {r: {m: output["replacement_D"][r][m]["egfr_pair_median_retention"]
                                          for m in ("c1_D2_only", "c2_Chronos_D2_RNA")}
                                     for r in ("GDSC2", "PRISM")}}, indent=2))


if __name__ == "__main__":
    main()
