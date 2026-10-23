#!/usr/bin/env python3
"""Execute the frozen Phase 1b N3 analysis and required robustness modules."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import yaml
from scipy import stats
from scipy.stats import rankdata, spearmanr
from rdkit import Chem, rdBase
from rdkit.Chem import DataStructs, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
REV = ROOT / "revision"
DATA = ROOT / "data" / "phase1b"
RAW = ROOT / "raw"
REVISION_CODE = ROOT / "discordance" / "revision"
sys.path.insert(0, str(REVISION_CODE))
sys.path.insert(0, str(ROOT / "discordance"))

import rev_common as rc  # noqa: E402
import rev_A2_nonlinear as a2  # noqa: E402
import rev_A3_attenuation as a3  # noqa: E402
from audit_prism_coverage import brd_core  # noqa: E402

SEED = 20260927
N_BOOT = 10_000
N_PERM = 10_000
MIN_N = 200
MIN_PAIR_N = 30
ELIGIBILITY_PATH = DATA / "PHASE1B_ELIGIBILITY.tsv"
GROUPS_PATH = DATA / "PHASE1B_TARGET_GROUPS.tsv"
SCAFFOLDS_PATH = DATA / "PHASE1B_SCAFFOLDS.tsv"
EXCLUSION_PATH = DATA / "PHASE1B_EXCLUSION.tsv"
MANIFEST_PATH = ROOT / "data" / "INPUT_MANIFEST.json"
GDSC_AUC_PATH = ROOT / "raw" / "GDSC2AUCMatrix.csv"
PRISM_PATH = ROOT / "raw" / "prism" / "secondary-screen-dose-response-curve-parameters.csv"
DEMETER2_PATH = RAW / "revision" / "D2_combined_gene_dep_scores.csv"
MODEL_PATH = Path("/path/to/depmap/26Q1/Model.csv")
PHASE1_PLANE_PATH = ROOT / "discordance" / "fig3_atlas_plane.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def median(values) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.median(x)) if len(x) else float("nan")


def ci95(values) -> list[float | None]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(x):
        return [None, None]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def load_gdsc_inputs():
    eligibility = pd.read_csv(ELIGIBILITY_PATH, sep="\t", dtype={"compound_id": str})
    included = eligibility.loc[eligibility["status"].eq("included")].copy()
    scaffolds = pd.read_csv(SCAFFOLDS_PATH, sep="\t", dtype={"compound_id": str})
    if "scaffold_smiles" in included.columns:
        included = included.drop(columns=["scaffold_smiles"])
    included = included.merge(
        scaffolds[["compound_id", "scaffold_smiles"]], on="compound_id", how="left", validate="one_to_one"
    )
    target_groups = pd.read_csv(GROUPS_PATH, sep="\t")
    eligible_mask = target_groups["target_group_eligible"].astype(str).str.lower().eq("true")
    eligible_targets = target_groups.loc[eligible_mask, "primary_target"].astype(str).tolist()
    frozen_sha = {
        "eligibility": sha256(ELIGIBILITY_PATH),
        "target_groups": sha256(GROUPS_PATH),
        "scaffolds": sha256(SCAFFOLDS_PATH),
    }
    with (ROOT / "03_analysis" / RUN.name / "run_config.yaml").open() as handle:
        run_config = yaml.safe_load(handle)
    frozen_expected = run_config["frozen_table_sha256"]
    expected = {
        "eligibility": frozen_expected["data/phase1b/PHASE1B_ELIGIBILITY.tsv"],
        "target_groups": frozen_expected["data/phase1b/PHASE1B_TARGET_GROUPS.tsv"],
        "scaffolds": frozen_expected["data/phase1b/PHASE1B_SCAFFOLDS.tsv"],
    }
    if frozen_sha != expected:
        raise RuntimeError(f"Frozen table SHA mismatch: observed={frozen_sha} expected={expected}")
    if len(included) != 43 or len(eligible_targets) != 6:
        raise RuntimeError(f"Frozen eligibility universe changed: drugs={len(included)}, targets={eligible_targets}")

    chronos = rc.gdsc2.load_chronos()
    models, lineage = rc._models_lineage()
    auc = pd.read_csv(GDSC_AUC_PATH, index_col=0, low_memory=False)
    auc.index = auc.index.astype(str)
    cells = auc.index.intersection(chronos.index).intersection(lineage.index)
    lineage = lineage.reindex(cells)
    drug_rows = {}
    for row in included.itertuples(index=False):
        drug, target = str(row.compound_id), str(row.primary_target)
        if drug not in auc.columns or target not in chronos.columns:
            raise RuntimeError(f"Frozen included drug/target absent from input: {drug} / {target}")
        response = rc.gdsc2.response_vector(auc.loc[cells, drug])
        dependency = -chronos.loc[cells, target].astype(float)
        residual = rc.linear_ols(response, dependency, lineage)
        n = int(residual.notna().sum())
        if n < MIN_N:
            raise RuntimeError(f"Frozen included compound fell below MIN_N: {drug}, n={n}")
        concordance = rc.gdsc2.partial_corr(response, dependency, lineage)
        drug_rows[drug] = {
            "compound_id": drug,
            "compound_name": str(row.compound_name),
            "target": target,
            "brd_id": "" if pd.isna(row.brd_id) else str(row.brd_id),
            "frozen_n_complete": int(row.n_complete),
            "n": n,
            "response": response,
            "chronos": dependency,
            "E": residual,
            "concordance_partial": float(concordance),
            "scaffold_smiles": str(row.scaffold_smiles),
        }
    print(f"Loaded GDSC2 Phase 1b: {len(drug_rows)} drugs, {len(eligible_targets)} target groups, {len(cells)} common models", flush=True)
    return included, target_groups, eligible_targets, chronos.reindex(cells), lineage, cells, drug_rows, run_config


def pair_table(residuals: dict[str, pd.Series], target_map: dict[str, str],
               names: dict[str, str] | None = None, similarity: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    drugs = [d for d in target_map if d in residuals]
    rows = []
    for a_i, a in enumerate(drugs):
        for b in drugs[a_i + 1:]:
            rho, n = rc.gdsc2.spearman_pair(residuals[a], residuals[b])
            row = {
                "drug_a": a, "drug_b": b,
                "name_a": (names or {}).get(a), "name_b": (names or {}).get(b),
                "target_a": target_map[a], "target_b": target_map[b],
                "same_target": target_map[a] == target_map[b],
                "n": int(n), "rho": float(rho) if np.isfinite(rho) else np.nan,
            }
            if similarity is not None:
                row["scaffold_tanimoto"] = similarity.get((a, b), similarity.get((b, a), np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def target_estimands(pairs: pd.DataFrame, target_map: dict[str, str],
                     target_list: list[str], drug_meta: dict[str, dict] | None = None) -> pd.DataFrame:
    rows = []
    for target in target_list:
        in_target = {d for d, t in target_map.items() if t == target}
        same = pairs.loc[pairs["drug_a"].isin(in_target) & pairs["drug_b"].isin(in_target) &
                         np.isfinite(pairs["rho"]) & pairs["n"].ge(MIN_PAIR_N)]
        cross_mask = pairs["drug_a"].isin(in_target) ^ pairs["drug_b"].isin(in_target)
        cross = pairs.loc[cross_mask & np.isfinite(pairs["rho"]) & pairs["n"].ge(MIN_PAIR_N)]
        w, c = median(same["rho"].to_numpy()), median(cross["rho"].to_numpy())
        meta = (drug_meta or {})
        n_scaffolds = len({meta[d].get("scaffold_smiles") for d in in_target if d in meta}) if meta else None
        rows.append({
            "target": target, "n_drugs": len(in_target), "n_scaffolds": n_scaffolds,
            "within_pair_count": int(len(same)), "cross_pair_count": int(len(cross)),
            "W_t": w, "C_t": c, "delta_t": w - c if np.isfinite(w) and np.isfinite(c) else np.nan,
            "compound_ids": "|".join(sorted(in_target)),
        })
    return pd.DataFrame(rows)


def summarize_effect(targets: pd.DataFrame) -> dict:
    valid = targets[np.isfinite(targets["delta_t"])].copy()
    deltas = valid["delta_t"].to_numpy(dtype=float)
    if not len(deltas):
        return {"median_delta": None, "n_targets": 0, "positive_fraction": None, "status": "NOT_ESTIMABLE"}
    rng = np.random.default_rng(SEED)
    draws = np.median(deltas[rng.integers(0, len(deltas), size=(N_BOOT, len(deltas)))], axis=1)
    lo, hi = ci95(draws)
    positive = float(np.mean(deltas > 0))
    passed = lo is not None and lo > 0 and positive >= 0.70
    return {
        "median_delta": float(np.median(deltas)), "ci95_target_bootstrap": [lo, hi],
        "n_targets": int(len(deltas)), "positive_targets": int(np.sum(deltas > 0)),
        "positive_fraction": positive, "bootstrap_n": N_BOOT, "bootstrap_seed": SEED,
        "status": "PASS" if passed else "FAIL",
    }


def pair_rho_anchor_lookup(drugs: list[str], genes: list[str], anchor_residuals: dict[tuple[str, str], pd.Series]):
    """Cache exact pairwise Spearman values under every possible target-anchor assignment."""
    mask_groups = {}
    for drug in drugs:
        grouped = {}
        for g_idx, gene in enumerate(genes):
            values = anchor_residuals[(drug, gene)].to_numpy(dtype=float)
            mask = np.isfinite(values)
            key = np.packbits(mask).tobytes()
            if key not in grouped:
                grouped[key] = [mask, []]
            grouped[key][1].append(g_idx)
        mask_groups[drug] = list(grouped.values())

    pair_i, pair_j = [], []
    rho_cache, n_cache = [], []
    for i, a in enumerate(drugs):
        for j in range(i + 1, len(drugs)):
            b = drugs[j]
            rho_matrix = np.full((len(genes), len(genes)), np.nan, dtype=float)
            n_matrix = np.zeros((len(genes), len(genes)), dtype=np.int16)
            for mask_a, genes_a in mask_groups[a]:
                for mask_b, genes_b in mask_groups[b]:
                    common = mask_a & mask_b
                    n_common = int(common.sum())
                    if n_common < MIN_PAIR_N:
                        continue
                    A = np.vstack([anchor_residuals[(a, genes[g])].to_numpy(dtype=float)[common] for g in genes_a])
                    B = np.vstack([anchor_residuals[(b, genes[g])].to_numpy(dtype=float)[common] for g in genes_b])
                    rank_a = rankdata(A, axis=1, method="average")
                    rank_b = rankdata(B, axis=1, method="average")
                    rank_a -= rank_a.mean(axis=1, keepdims=True)
                    rank_b -= rank_b.mean(axis=1, keepdims=True)
                    norm_a = np.sqrt(np.square(rank_a).sum(axis=1, keepdims=True))
                    norm_b = np.sqrt(np.square(rank_b).sum(axis=1, keepdims=True))
                    valid_a, valid_b = norm_a[:, 0] > 0, norm_b[:, 0] > 0
                    block = np.full((len(genes_a), len(genes_b)), np.nan, dtype=float)
                    if valid_a.any() and valid_b.any():
                        block[np.ix_(valid_a, valid_b)] = (
                            (rank_a[valid_a] / norm_a[valid_a]) @ (rank_b[valid_b] / norm_b[valid_b]).T
                        )
                    rho_matrix[np.ix_(genes_a, genes_b)] = block
                    n_matrix[np.ix_(genes_a, genes_b)] = n_common
            pair_i.append(i); pair_j.append(j)
            rho_cache.append(rho_matrix); n_cache.append(n_matrix)
        if (i + 1) % 10 == 0:
            print(f"  cached anchor-pair matrices for {i + 1}/{len(drugs)} drugs", flush=True)
    return np.asarray(pair_i), np.asarray(pair_j), np.stack(rho_cache), np.stack(n_cache)


def label_permutation_test(drugs: list[str], labels: np.ndarray, genes: list[str], eligible_targets: list[str],
                           anchor_residuals: dict[tuple[str, str], pd.Series], observed_effect: float,
                           label: str) -> tuple[dict, pd.DataFrame]:
    print(f"Building {label} target-label permutation cache", flush=True)
    anchor_idx = {g: i for i, g in enumerate(genes)}
    for idx, drug in enumerate(drugs):
        for g in genes:
            if int(anchor_residuals[(drug, g)].notna().sum()) < MIN_N:
                raise RuntimeError(f"Permuted target anchor violates MIN_N: {drug}, anchor={g}")
        if (idx + 1) % 10 == 0:
            print(f"  residualized {idx + 1}/{len(drugs)} drugs across all target anchors", flush=True)
    pair_i, pair_j, rho_lookup, n_lookup = pair_rho_anchor_lookup(drugs, genes, anchor_residuals)
    actual_gene_idx = np.asarray([anchor_idx[g] for g in labels], dtype=int)
    selected = np.arange(len(pair_i))
    actual_rho = rho_lookup[selected, actual_gene_idx[pair_i], actual_gene_idx[pair_j]]
    # Validate the lookup implementation against the direct frozen pairwise Spearman calculation.
    max_lookup_delta = 0.0
    for k, (i, j) in enumerate(zip(pair_i, pair_j)):
        a, b = drugs[i], drugs[j]
        direct, _ = rc.gdsc2.spearman_pair(anchor_residuals[(a, labels[i])], anchor_residuals[(b, labels[j])])
        cached = actual_rho[k]
        if np.isfinite(direct) and np.isfinite(cached):
            max_lookup_delta = max(max_lookup_delta, abs(float(direct) - float(cached)))
        elif np.isfinite(direct) != np.isfinite(cached):
            raise RuntimeError(f"Target permutation cache/direct rho validity differs: {a} / {b}")
    if max_lookup_delta > 1e-10:
        raise RuntimeError(f"Target permutation cache/direct rho mismatch: {max_lookup_delta}")

    rng = np.random.default_rng(SEED + (1 if label == "GDSC2" else 101))
    observed_base_labels = np.asarray(labels, dtype=object)
    null = np.full(N_PERM, np.nan, dtype=float)
    rows = []
    for rep in range(N_PERM):
        perm_labels = rng.permutation(observed_base_labels)
        gene_ids = np.asarray([anchor_idx[t] for t in perm_labels], dtype=int)
        vals = rho_lookup[selected, gene_ids[pair_i], gene_ids[pair_j]]
        ns = n_lookup[selected, gene_ids[pair_i], gene_ids[pair_j]]
        per_target = []
        for target in eligible_targets:
            a_in = perm_labels[pair_i] == target
            b_in = perm_labels[pair_j] == target
            within = a_in & b_in & np.isfinite(vals) & (ns >= MIN_PAIR_N)
            cross = (a_in ^ b_in) & np.isfinite(vals) & (ns >= MIN_PAIR_N)
            w = median(vals[within]); c = median(vals[cross])
            per_target.append(w - c if np.isfinite(w) and np.isfinite(c) else np.nan)
        effect = median(per_target)
        null[rep] = effect
        rows.append({"replicate": rep + 1, "median_delta_t": effect,
                     "n_target_deltas_valid": int(np.isfinite(per_target).sum())})
        if (rep + 1) % 1000 == 0:
            print(f"  {label} target-label permutations {rep + 1}/{N_PERM}", flush=True)
    valid = np.isfinite(null)
    p = float((1 + np.sum(null[valid] >= observed_effect)) / (1 + int(valid.sum()))) if valid.any() else None
    return ({"n_permutations_requested": N_PERM, "n_permutations_valid": int(valid.sum()),
             "seed": SEED + (1 if label == "GDSC2" else 101), "observed_median_delta": observed_effect,
             "null_median": float(np.median(null[valid])) if valid.any() else None,
             "p_one_sided": p, "max_abs_lookup_vs_direct_rho": max_lookup_delta,
             "anchor_residual_min_n": int(min(s.notna().sum() for s in anchor_residuals.values())),
             "cache_method": "all drug × target-anchor OLS residuals precomputed; permutation selects the exact residuals and recomputes all target-level statistics"},
            pd.DataFrame(rows))


def make_anchor_residuals(drugs: list[str], genes: list[str], drug_data: dict[str, dict],
                          chronos: pd.DataFrame, lineage: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    anchor_residuals = {}
    for idx, drug in enumerate(drugs):
        response = drug_data[drug]["response"]
        for gene in genes:
            dependency = -chronos[gene].reindex(response.index).astype(float)
            anchor_residuals[(drug, gene)] = rc.linear_ols(response, dependency, lineage)
        if (idx + 1) % 10 == 0:
            print(f"  precomputed anchor residuals {idx + 1}/{len(drugs)}", flush=True)
    return anchor_residuals


def compute_scaffolds(included: pd.DataFrame, groups: pd.DataFrame):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, errors = {}, []
    for row in included.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.scaffold_smiles))
        if mol is None:
            errors.append(str(row.compound_id)); continue
        fps[str(row.compound_id)] = fpgen.GetFingerprint(mol)
    if errors or len(fps) != len(included):
        raise RuntimeError(f"Unable to fingerprint frozen scaffolds: {errors}")
    similarity = {}
    for a, b in itertools.combinations(list(fps), 2):
        similarity[(a, b)] = float(DataStructs.TanimotoSimilarity(fps[a], fps[b]))
    frozen_vals = {}
    for row in groups.itertuples(index=False):
        text = "" if pd.isna(row.distinct_pairs) else str(row.distinct_pairs)
        for token in filter(None, text.split("|")):
            left, value = token.split(":")
            a, b = left.split("~")
            frozen_vals[(a, b)] = float(value)
    diffs = []
    for pair, frozen in frozen_vals.items():
        current = similarity.get(pair, similarity.get((pair[1], pair[0])))
        if current is not None:
            diffs.append(abs(current - frozen))
    diag = {
        "runtime_rdkit_version": rdBase.rdkitVersion,
        "frozen_eligibility_rdkit_version": "2025.09.6",
        "version_match": rdBase.rdkitVersion == "2025.09.6",
        "fingerprint": "Morgan radius=2, fpSize=2048 over frozen scaffold_smiles",
        "n_frozen_within_target_pair_comparisons": len(diffs),
        "max_abs_similarity_difference_vs_frozen_3dp_values": float(max(diffs)) if diffs else None,
        "n_frozen_within_target_values_equal_after_3dp_rounding": int(sum(round(similarity.get(p, similarity.get((p[1], p[0]))), 3) == round(v, 3) for p, v in frozen_vals.items() if similarity.get(p, similarity.get((p[1], p[0]))) is not None)),
    }
    return similarity, diag


def scaffold_matched_test(pairs: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = pairs[np.isfinite(pairs["scaffold_tanimoto"]) & np.isfinite(pairs["rho"])].copy()
    data["similarity_bin"] = pd.cut(
        data["scaffold_tanimoto"], bins=[0.0, 0.2, 0.4, 0.7],
        labels=["[0,0.2)", "[0.2,0.4)", "[0.4,0.7)"], right=False,
    ).astype("object")
    data.loc[data["scaffold_tanimoto"].ge(0.7), "similarity_bin"] = "outside_[0,0.7)"
    in_bins = data[data["similarity_bin"].isin(["[0,0.2)", "[0.2,0.4)", "[0.4,0.7)"])].copy()
    bin_rows = []
    for bin_name in ["[0,0.2)", "[0.2,0.4)", "[0.4,0.7)"]:
        block = in_bins[in_bins["similarity_bin"].eq(bin_name)]
        same = block.loc[block["same_target"], "rho"].to_numpy(dtype=float)
        cross = block.loc[~block["same_target"], "rho"].to_numpy(dtype=float)
        bin_rows.append({"similarity_bin": bin_name, "n_same_target_pairs": len(same),
                         "n_cross_target_pairs": len(cross), "median_same_rho": median(same),
                         "median_cross_rho": median(cross),
                         "median_difference": median(same) - median(cross) if len(same) and len(cross) else np.nan})
    usable = [r for r in bin_rows if np.isfinite(r["median_difference"])]
    if not usable:
        return data, pd.DataFrame(bin_rows), {"status": "NOT_ESTIMABLE", "p_one_sided": None}
    observed = float(np.mean([r["median_difference"] for r in usable]))
    rng = np.random.default_rng(seed)
    null = np.full(N_PERM, np.nan)
    group_blocks = {r["similarity_bin"]: in_bins[in_bins["similarity_bin"].eq(r["similarity_bin"])] for r in usable}
    for rep in range(N_PERM):
        diffs = []
        for r in usable:
            block = group_blocks[r["similarity_bin"]]
            labels = block["same_target"].to_numpy(dtype=bool)
            values = block["rho"].to_numpy(dtype=float)
            shuffled = rng.permutation(labels)
            diffs.append(median(values[shuffled]) - median(values[~shuffled]))
        null[rep] = float(np.mean(diffs))
    p = float((1 + np.sum(null >= observed)) / (N_PERM + 1))
    return data, pd.DataFrame(bin_rows), {
        "status": "COMPLETED", "n_pairs_in_frozen_bins": int(len(in_bins)),
        "n_pairs_outside_frozen_bins_ge_0_7": int(data["similarity_bin"].eq("outside_[0,0.7)").sum()),
        "aggregation": "unweighted mean of per-bin (median same-target rho − median cross-target rho); permutation shuffles pair labels within bins, preserving each bin's same-target count",
        "observed_stratified_difference": observed, "p_one_sided": p, "n_permutations": N_PERM,
        "seed": seed, "null_median": float(np.median(null)),
        "dyadic_pairs_are_dependent": True,
    }


def partial_mantel_test(pairs: pd.DataFrame, drugs: list[str], labels: np.ndarray, seed: int):
    idx = {d: i for i, d in enumerate(drugs)}
    usable = pairs[np.isfinite(pairs["rho"]) & pairs["scaffold_tanimoto"].notna()].copy()
    i = usable["drug_a"].map(idx).to_numpy(dtype=int)
    j = usable["drug_b"].map(idx).to_numpy(dtype=int)
    y = usable["rho"].to_numpy(dtype=float)
    f = usable["scaffold_tanimoto"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(f)), f])
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    def statistic(target_labels: np.ndarray) -> float:
        t = (target_labels[i] == target_labels[j]).astype(float)
        rt = t - design @ np.linalg.lstsq(design, t, rcond=None)[0]
        if np.std(ry) == 0 or np.std(rt) == 0:
            return float("nan")
        return float(np.corrcoef(ry, rt)[0, 1])

    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    null = np.full(N_PERM, np.nan)
    for rep in range(N_PERM):
        null[rep] = statistic(rng.permutation(labels))
        if (rep + 1) % 2000 == 0:
            print(f"  partial Mantel permutations {rep + 1}/{N_PERM}", flush=True)
    valid = np.isfinite(null)
    p = float((1 + np.sum(null[valid] >= observed)) / (1 + int(valid.sum()))) if valid.any() else None
    return ({"method": "partial Mantel Pearson correlation of upper-triangle rho(E) and same-target indicator controlling scaffold Morgan Tanimoto",
             "observed_partial_mantel_r": observed, "p_one_sided": p,
             "n_drugs": len(drugs), "n_drug_pairs": len(usable),
             "n_permutations_requested": N_PERM, "n_permutations_valid": int(valid.sum()),
             "seed": seed, "null_median": float(np.median(null[valid])) if valid.any() else None},
            pd.DataFrame({"replicate": np.arange(1, N_PERM + 1), "partial_mantel_r": null}))


def target_class_results(targets: pd.DataFrame) -> pd.DataFrame:
    classes = {"EGFR": "kinase", "MTOR": "kinase", "ATR": "kinase", "CHEK1": "kinase",
               "ESR1": "non-kinase", "TOP1": "non-kinase"}
    out = targets.copy()
    out["target_class"] = out["target"].map(classes)
    rows = []
    for cls in ["kinase", "non-kinase"]:
        block = out[out["target_class"].eq(cls) & np.isfinite(out["delta_t"])]
        rows.append({"target_class": cls, "n_targets": int(len(block)),
                     "median_delta_t": median(block["delta_t"]),
                     "target_deltas": "|".join(f"{r.target}:{r.delta_t:.8g}" for r in block.itertuples()),
                     "status": "BLOCKED" if cls == "non-kinase" and len(block) < 3 else "DESCRIPTIVE"})
    return out, pd.DataFrame(rows)


def all_target_descriptives(pairs: pd.DataFrame, target_map: dict[str, str],
                            eligible_targets: list[str], drug_data: dict[str, dict]) -> pd.DataFrame:
    targets = sorted(set(target_map.values()))
    result = target_estimands(pairs, target_map, targets, drug_data)
    result["eligible_primary_group"] = result["target"].isin(eligible_targets)
    return result


def atlas_plane(n3_targets: pd.DataFrame, n3_drugs: dict[str, dict]):
    phase1 = json.loads(PHASE1_PLANE_PATH.read_text())
    points = []
    for row in n3_targets.itertuples(index=False):
        ds = [d for d, v in n3_drugs.items() if v["target"] == row.target]
        x = float(np.mean([n3_drugs[d]["concordance_partial"] for d in ds]))
        y = float(row.W_t)
        cls = "II" if x >= .15 and y >= .30 else "I" if x >= .15 else "III" if y >= .30 else "IV"
        points.append({"target": row.target, "source": "Phase 1b N3", "x_mean_concordance_partial": x,
                       "y_within_target_W": y, "atlas_class": cls, "n_drugs": len(ds)})
    for row in phase1["multi_chemotype_targets"]:
        if row["target"] in {"EGFR", "AKT1", "MAPK1"}:
            points.append({"target": row["target"], "source": "Phase 1 frozen", "x_mean_concordance_partial": float(row["x"]),
                           "y_within_target_W": float(row["y"]), "atlas_class": row["atlas_class"],
                           "n_drugs": int(row["n_drugs"])})
    point_df = pd.DataFrame(points)
    rng = np.random.default_rng(SEED + 4)
    rho_draws = []
    arr_x = point_df["x_mean_concordance_partial"].to_numpy(dtype=float)
    arr_y = point_df["y_within_target_W"].to_numpy(dtype=float)
    for _ in range(N_BOOT):
        take = rng.integers(0, len(arr_x), size=len(arr_x))
        rho = spearmanr(arr_x[take], arr_y[take]).statistic
        if np.isfinite(rho): rho_draws.append(float(rho))
    rho_ci = ci95(rho_draws)
    class_i = bool(point_df["atlas_class"].eq("I").any())
    class_ii = bool(point_df["atlas_class"].eq("II").any())
    ci_includes_zero = rho_ci[0] is not None and rho_ci[0] <= 0 <= rho_ci[1]
    return point_df, {"n_points": int(len(point_df)), "includes_phase1_egfr_and_phase1b_egfr_separately": True,
                      "class_I_present": class_i, "class_II_present": class_ii,
                      "spearman_x_y": float(spearmanr(arr_x, arr_y).statistic),
                      "spearman_bootstrap_ci95": rho_ci, "bootstrap_n": N_BOOT,
                      "bootstrap_seed": SEED + 4, "ci_includes_zero": bool(ci_includes_zero),
                      "two_axes_independence_retained": bool(class_i and class_ii and ci_includes_zero),
                      "duplicate_target_note": "Phase 1 EGFR and Phase 1b EGFR are separate frozen/expanded points, as specified."}


def plot_outputs(all_targets: pd.DataFrame, atlas_points: pd.DataFrame, outdir: Path):
    figures = outdir / "figures"
    figures.mkdir(exist_ok=True)
    all_targets = all_targets.sort_values(["eligible_primary_group", "delta_t"], ascending=[False, False]).reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(8, len(all_targets) * .28)), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(all_targets))
    wvals = all_targets["W_t"].to_numpy(dtype=float)
    cvals = all_targets["C_t"].to_numpy(dtype=float)
    ax1.scatter(wvals, y - .13, marker="D", s=28, color="#164E78", label="Within target W")
    ax1.scatter(cvals, y + .13, marker="o", s=24, color="#D97721", label="Cross-target C")
    ax1.axvline(0, color="#808080", linewidth=.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r.target}  (n={r.n_drugs})" for r in all_targets.itertuples()], fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlabel("Median Spearman ρ(E)")
    ax1.set_title("All 30 annotated targets; W is NA for singletons")
    ax1.legend(loc="lower right", frameon=False)
    primary = all_targets[all_targets["eligible_primary_group"] & np.isfinite(all_targets["delta_t"])].copy()
    primary = primary.sort_values("delta_t")
    ax2.barh(primary["target"], primary["delta_t"], color=np.where(primary["delta_t"] > 0, "#287D56", "#A33A3A"))
    ax2.axvline(0, color="#333333", linewidth=.9)
    ax2.set_xlabel("Δₜ = Wₜ − Cₜ")
    ax2.set_title("Frozen six multi-scaffold groups")
    fig.suptitle("Phase 1b GDSC2 target expansion", y=.995)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"N3_target_estimands.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    markers = {"Phase 1 frozen": "s", "Phase 1b N3": "o"}
    for source, block in atlas_points.groupby("source", sort=True):
        ax.scatter(block["x_mean_concordance_partial"], block["y_within_target_W"],
                   s=60, marker=markers[source], label=source, alpha=.9)
        for row in block.itertuples(index=False):
            label = f"{row.target} ({'P1' if source.startswith('Phase 1 ') else 'P1b'})"
            ax.annotate(label, (row.x_mean_concordance_partial, row.y_within_target_W), xytext=(5, 4),
                        textcoords="offset points", fontsize=8)
    ax.axvline(.15, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(.30, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean concordance partial correlation xₜ")
    ax.set_ylabel("Within-target residual concordance Wₜ")
    ax.set_title("Atlas plane extension (Phase 1 figure unchanged)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"N3_atlas_plane.{ext}", dpi=240, bbox_inches="tight")
    draft_dir = ROOT / "revision_figs"
    draft_dir.mkdir(exist_ok=True)
    fig.savefig(draft_dir / "N3_atlas_plane.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    eligibility = pd.read_csv(ELIGIBILITY_PATH, sep="\t")
    exclusions = eligibility.loc[~eligibility["status"].eq("included")].copy()
    exclusions["reason_code"] = exclusions["reason_code"].fillna("unspecified")
    counts = (exclusions.groupby(["status", "reason_code"], dropna=False).size()
              .sort_values().reset_index(name="n_compounds"))
    labels = [f"{r.status} · {r.reason_code}" for r in counts.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(10, max(5, .34 * len(counts))))
    colors = ["#B45D4C" if status == "excluded" else "#7A7A7A" for status in counts["status"]]
    ax.barh(labels, counts["n_compounds"], color=colors)
    for y_pos, count in enumerate(counts["n_compounds"]):
        ax.text(count + .5, y_pos, str(int(count)), va="center", fontsize=8)
    ax.set_xlabel("Compounds")
    ax.set_title("Frozen Phase 1b exclusions and unresolved identities")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"N3_eligibility_exclusion_reasons.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_prism_coverage(included: pd.DataFrame, eligible_targets: list[str],
                        chronos: pd.DataFrame, lineage: pd.DataFrame):
    curves = pd.read_csv(PRISM_PATH, usecols=["broad_id", "depmap_id", "auc"], low_memory=False)
    curves["core"] = curves["broad_id"].astype(str).map(brd_core)
    curves["ach"] = curves["depmap_id"].astype(str)
    curves["auc"] = pd.to_numeric(curves["auc"], errors="coerce")
    curves = curves[curves["ach"].str.startswith("ACH-") & curves["auc"].notna()]
    all_brd = set()
    mapping = {}
    for row in included.itertuples(index=False):
        parts = [] if pd.isna(row.brd_id) else [brd_core(p) for p in str(row.brd_id).split("|")]
        parts = list(dict.fromkeys([p for p in parts if p]))
        mapping[str(row.compound_id)] = parts
        all_brd.update(parts)
    curves = curves[curves["core"].isin(all_brd)]
    raw_by_drug = {}
    rows = []
    target_map = dict(zip(included["compound_id"].astype(str), included["primary_target"].astype(str)))
    meta_by_drug = included.set_index("compound_id").to_dict("index")
    for drug in target_map:
        brds = mapping[drug]
        if not brds:
            rows.append({"compound_id": drug, "compound_name": meta_by_drug[drug]["compound_name"],
                         "target": target_map[drug], "brd_ids": "", "n_models_complete": 0,
                         "status": "NO_FROZEN_BRD_MAPPING"})
            continue
        subset = curves[curves["core"].isin(brds)]
        auc = subset.groupby("ach", sort=False)["auc"].mean()
        idx = auc.index.intersection(chronos.index).intersection(lineage.index)
        response = rc.gdsc2.response_vector(auc.reindex(idx))
        dependency = -chronos[target_map[drug]].reindex(idx).astype(float)
        n = int(pd.concat([response.rename("R"), dependency.rename("D"), lineage.reindex(idx)], axis=1).dropna().shape[0])
        status = "LEGAL" if n >= MIN_N else "BELOW_MIN_N"
        if status == "LEGAL":
            raw_by_drug[drug] = {"response": response, "target": target_map[drug], "brd_ids": brds,
                                 "n": n, "name": str(meta_by_drug[drug]["compound_name"]),
                                 "scaffold_smiles": str(meta_by_drug[drug]["scaffold_smiles"])}
        rows.append({"compound_id": drug, "compound_name": meta_by_drug[drug]["compound_name"],
                     "target": target_map[drug], "brd_ids": "|".join(brds), "n_models_complete": n,
                     "n_curve_rows": int(len(subset)), "status": status})
    coverage = pd.DataFrame(rows)
    group_counts = {t: int(((coverage["target"].eq(t)) & coverage["status"].eq("LEGAL")).sum()) for t in eligible_targets}
    covered_groups = {t: n for t, n in group_counts.items() if n >= 2}
    return raw_by_drug, coverage, {"source": "PRISM Secondary 20Q2 AUC", "minimum_n": MIN_N,
                                    "frozen_primary_target_groups": eligible_targets,
                                    "legal_drugs_by_target": group_counts,
                                    "groups_with_at_least_two_legal_compounds": covered_groups,
                                    "n_covered_groups": len(covered_groups),
                                    "coverage_gate_required": 4,
                                    "status": "LEGAL" if len(covered_groups) >= 4 else "BLOCKED_BY_COVERAGE"}


def cross_resource_xs_xc(gdsc_residuals: dict[str, pd.Series], prism_residuals: dict[str, pd.Series],
                         gdsc_target_map: dict[str, str], prism_target_map: dict[str, str],
                         gdsc_cells: pd.Index, prism_cells: pd.Index):
    shared_drugs = [d for d in gdsc_residuals if d in prism_residuals]
    if len(shared_drugs) < 2:
        return pd.DataFrame(), {"status": "NOT_ESTIMABLE", "n_shared_drugs": len(shared_drugs)}
    common = pd.Index(sorted(set(map(str, gdsc_cells)) & set(map(str, prism_cells))), dtype=object)
    rows = []
    for a, b in itertools.combinations(shared_drugs, 2):
        r1, n1 = rc.gdsc2.spearman_pair(gdsc_residuals[a].reindex(common), prism_residuals[b].reindex(common))
        r2, n2 = rc.gdsc2.spearman_pair(gdsc_residuals[b].reindex(common), prism_residuals[a].reindex(common))
        x = float(np.mean([r1, r2])) if np.isfinite(r1) and np.isfinite(r2) else np.nan
        rows.append({"drug_a": a, "drug_b": b, "target_a": gdsc_target_map[a], "target_b": gdsc_target_map[b],
                     "same_target": gdsc_target_map[a] == gdsc_target_map[b], "rho_Ga_Pb": r1,
                     "rho_Gb_Pa": r2, "n_Ga_Pb": n1, "n_Gb_Pa": n2, "x_pair": x})
    pairs = pd.DataFrame(rows)
    same = pairs.loc[pairs["same_target"], "x_pair"].to_numpy(dtype=float)
    cross = pairs.loc[~pairs["same_target"], "x_pair"].to_numpy(dtype=float)
    return pairs, {"status": "DESCRIPTIVE", "n_common_cells": len(common), "n_shared_drugs": len(shared_drugs),
                   "n_same_target_pairs": int(np.isfinite(same).sum()), "n_cross_target_pairs": int(np.isfinite(cross).sum()),
                   "xS_median": median(same), "xC_median": median(cross),
                   "definition": "X1: mean of reciprocal cross-resource drug-pair Spearman correlations on the shared-cell set; medians split by frozen target labels"}


def prism_analysis(raw_by_drug: dict[str, dict], coverage: pd.DataFrame, gate: dict,
                   included: pd.DataFrame, eligible_targets: list[str], chronos: pd.DataFrame,
                   lineage: pd.DataFrame, gdsc_drugs: dict[str, dict], gdsc_cells: pd.Index):
    if gate["status"] != "LEGAL":
        return {"coverage": gate, "primary": {"status": "BLOCKED_BY_COVERAGE"}}, None, None, None
    prism_cell_index = pd.Index([], dtype=object)
    for row in raw_by_drug.values():
        prism_cell_index = prism_cell_index.union(row["response"].index, sort=False)
    data = {}
    for drug, row in raw_by_drug.items():
        response = row["response"].reindex(prism_cell_index)
        dep = -chronos[row["target"]].reindex(prism_cell_index).astype(float)
        e = rc.linear_ols(response, dep, lineage.reindex(prism_cell_index))
        if int(e.notna().sum()) < MIN_N:
            raise RuntimeError(f"PRISM coverage gate/OLS complete cases disagree for {drug}")
        data[drug] = {**row, "response": response, "chronos": dep, "E": e}
    targets = {d: x["target"] for d, x in data.items()}
    names = {d: x["name"] for d, x in data.items()}
    prism_eligible_targets = [t for t, n in gate["legal_drugs_by_target"].items() if n >= 2]
    prism_pairs = pair_table({d: x["E"] for d, x in data.items()}, targets, names)
    prism_target_rows = target_estimands(prism_pairs, targets, prism_eligible_targets, data)
    prism_target_rows.to_csv(RUN / "n3_prism_targets.tsv", sep="\t", index=False)
    prism_pairs.to_csv(RUN / "n3_prism_pairs.tsv", sep="\t", index=False)
    prism_main = summarize_effect(prism_target_rows)
    # Recompute E under each label permutation, selecting precomputed exact OLS residuals per target gene.
    prism_genes = sorted(set(targets.values()))
    prism_anchor = make_anchor_residuals(list(data), prism_genes, data, chronos, lineage)
    prism_perm, prism_null = label_permutation_test(list(data), np.asarray([targets[d] for d in data]), prism_genes,
                                                     prism_eligible_targets,
                                                     prism_anchor, prism_main["median_delta"], "PRISM")
    prism_null.to_csv(RUN / "n3_prism_null_distribution.tsv", sep="\t", index=False)
    prism_valid_cells = prism_cell_index
    gdsc_prism_common = pd.Index(sorted(set(map(str, gdsc_cells)) & set(map(str, prism_valid_cells))), dtype=object)
    gdsc_e = {d: v["E"] for d, v in gdsc_drugs.items() if d in data}
    prism_e = {d: v["E"] for d, v in data.items()}
    gdsc_targets = {d: gdsc_drugs[d]["target"] for d in gdsc_e}
    xs_pairs, xs = cross_resource_xs_xc(gdsc_e, prism_e, gdsc_targets, targets, gdsc_prism_common, prism_valid_cells)
    if xs_pairs is not None:
        xs_pairs.to_csv(RUN / "n3_cross_resource_xs_xc.tsv", sep="\t", index=False)
    return {"coverage": gate, "primary": prism_main, "target_level": prism_target_rows.to_dict("records"),
            "target_label_permutation": prism_perm, "cross_resource_xS_xC": xs}, data, prism_pairs, prism_target_rows


def main():
    gate_path = REVISION_CODE / "rev_common_regression.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS" or gate.get("checks_passed") != 254 or gate.get("checks_total") != 254:
        raise RuntimeError("Regression gate is not PASS 254/254; stop before Phase 1b analysis")
    (RUN / "figures").mkdir(exist_ok=True)
    included, groups, eligible_targets, chronos, lineage, cells, drug_data, run_config = load_gdsc_inputs()
    drug_names = {d: row["compound_name"] for d, row in drug_data.items()}
    target_map = {d: row["target"] for d, row in drug_data.items()}
    scaffold_similarity, scaffold_diag = compute_scaffolds(included, groups)
    print(f"GDSC2 observed residuals computed for {len(drug_data)} compounds", flush=True)

    drug_rows = []
    for d, row in drug_data.items():
        drug_rows.append({k: v for k, v in row.items() if k not in {"response", "chronos", "E"}})
    pd.DataFrame(drug_rows).to_csv(RUN / "n3_drug_results.tsv", sep="\t", index=False)

    base_pairs = pair_table({d: x["E"] for d, x in drug_data.items()}, target_map, drug_names, scaffold_similarity)
    base_pairs.to_csv(RUN / "n3_pairs_gdsc2.tsv", sep="\t", index=False)
    target_rows = target_estimands(base_pairs, target_map, eligible_targets, drug_data)
    all_target_rows = all_target_descriptives(base_pairs, target_map, eligible_targets, drug_data)
    target_rows.to_csv(RUN / "n3_target_results.tsv", sep="\t", index=False)
    all_target_rows.to_csv(RUN / "n3_all_target_descriptives.tsv", sep="\t", index=False)
    main_result = summarize_effect(target_rows)
    labels = np.asarray([target_map[d] for d in drug_data], dtype=object)
    genes = sorted(set(labels.tolist()))
    anchor_residuals = make_anchor_residuals(list(drug_data), genes, drug_data, chronos, lineage)
    permutation, perm_dist = label_permutation_test(list(drug_data), labels, genes, eligible_targets,
                                                     anchor_residuals, main_result["median_delta"], "GDSC2")
    perm_dist.to_csv(RUN / "n3_gdsc2_null_distribution.tsv", sep="\t", index=False)

    # Mandatory delete-one-compound robustness; a two-drug target becomes non-estimable when one is removed.
    loo_drug = []
    for drug in drug_data:
        keep = [d for d in drug_data if d != drug]
        keep_target_map = {d: target_map[d] for d in keep}
        pp = base_pairs[base_pairs["drug_a"].isin(keep) & base_pairs["drug_b"].isin(keep)]
        target_set = [t for t in eligible_targets if sum(keep_target_map[d] == t for d in keep) >= 2]
        rr = target_estimands(pp, keep_target_map, target_set, drug_data)
        smry = summarize_effect(rr)
        loo_drug.append({"removed_drug": drug, "removed_name": drug_names[drug], "removed_target": target_map[drug],
                         "n_targets_estimable": smry["n_targets"], "median_delta": smry["median_delta"],
                         "ci95_target_bootstrap": smry.get("ci95_target_bootstrap"),
                         "positive_fraction": smry.get("positive_fraction"), "status": smry["status"]})
    loo_drug_df = pd.DataFrame(loo_drug)
    loo_drug_df.to_csv(RUN / "n3_leave_one_drug_out.tsv", sep="\t", index=False)

    loo_target = []
    for dropped in eligible_targets:
        keep = [d for d in drug_data if target_map[d] != dropped]
        keep_target_map = {d: target_map[d] for d in keep}
        remaining_targets = [t for t in eligible_targets if t != dropped]
        pp = base_pairs[base_pairs["drug_a"].isin(keep) & base_pairs["drug_b"].isin(keep)]
        rr = target_estimands(pp, keep_target_map, remaining_targets, drug_data)
        smry = summarize_effect(rr)
        loo_target.append({"removed_target": dropped, "n_drugs_remaining": len(keep),
                           "n_targets_estimable": smry["n_targets"], "median_delta": smry["median_delta"],
                           "ci95_target_bootstrap": smry.get("ci95_target_bootstrap"),
                           "positive_fraction": smry.get("positive_fraction"), "status": smry["status"]})
    loo_target_df = pd.DataFrame(loo_target)
    loo_target_df.to_csv(RUN / "n3_leave_one_target_out.tsv", sep="\t", index=False)

    # Scaffold similarity control. Frozen Murcko scaffolds are fingerprinted with the runtime version;
    # frozen in-target similarities are independently checked against the table values.
    scaffold_pairs, scaffold_bins, scaffold_result = scaffold_matched_test(base_pairs, SEED + 2)
    scaffold_pairs.to_csv(RUN / "n3_scaffold_matched_pairs.tsv", sep="\t", index=False)
    scaffold_bins.to_csv(RUN / "n3_scaffold_matched_bins.tsv", sep="\t", index=False)
    write_json(RUN / "n3_scaffold_fingerprint_provenance.json", scaffold_diag)

    class_targets, class_summary = target_class_results(target_rows)
    class_targets.to_csv(RUN / "n3_target_class_stratification.tsv", sep="\t", index=False)
    class_summary.to_csv(RUN / "n3_target_class_summary.tsv", sep="\t", index=False)
    annotation = {"status": "BLOCKED", "reason": "No second fully documented, frozen target-annotation source is present.",
                  "source_used": "ChEMBL 36 drug_mechanism frozen eligibility table", "rescue_or_reassignment_performed": False}
    write_json(RUN / "n3_annotation_sensitivity.json", annotation)

    mantel, mantel_dist = partial_mantel_test(base_pairs, list(drug_data), labels, SEED + 3)
    mantel_dist.to_csv(RUN / "n3_partial_mantel_null_distribution.tsv", sep="\t", index=False)

    # Section 6: combine Phase 1b coordinates with three frozen Phase 1 anchors.
    atlas_points, atlas_result = atlas_plane(target_rows, drug_data)
    atlas_points.to_csv(RUN / "n3_atlas_plane_points.tsv", sep="\t", index=False)
    plot_outputs(all_target_rows, atlas_points, RUN)

    # Section 8: exact A2 spline residualizer, on the N3 compounds and frozen cells.
    spline_diag = {}
    spline_fn = a2.make_residualizer("spline", spline_diag)
    spline_E = {d: spline_fn(v["response"], v["chronos"], lineage) for d, v in drug_data.items()}
    spline_pairs = pair_table(spline_E, target_map, drug_names)
    spline_targets = target_estimands(spline_pairs, target_map, eligible_targets, drug_data)
    spline_summary = summarize_effect(spline_targets)
    spline_targets.to_csv(RUN / "n3_spline_target_results.tsv", sep="\t", index=False)
    spline_pairs.to_csv(RUN / "n3_spline_pairs.tsv", sep="\t", index=False)
    pd.DataFrame([{"compound_id": k, **v} for k, v in spline_diag.items()]).to_csv(RUN / "n3_spline_fit_diagnostics.tsv", sep="\t", index=False)
    spline_retention = (spline_summary["median_delta"] / main_result["median_delta"]
                        if spline_summary.get("median_delta") is not None and main_result.get("median_delta") not in (None, 0) else None)

    # Section 8: DEMETER2 replacement on the exact per-drug intersection of response, Chronos, D2, and lineage.
    d2_first_col = pd.read_csv(DEMETER2_PATH, usecols=[0], dtype=str).iloc[:, 0].astype(str)
    d2_genes_present = []
    d2_genes_missing = []
    for gene in sorted(set(target_map.values())):
        try:
            a3.gene_column(d2_first_col, gene)
            d2_genes_present.append(gene)
        except KeyError:
            d2_genes_missing.append(gene)
    d2_map, d2_diag = a3.map_external_matrix(DEMETER2_PATH, MODEL_PATH, d2_genes_present) if d2_genes_present else ({}, {})
    d2_E, chronos_subset_E, d2_n = {}, {}, {}
    for d, row in drug_data.items():
        target = row["target"]
        if target not in d2_map:
            d2_E[d] = pd.Series(np.nan, index=row["response"].index)
            chronos_subset_E[d] = pd.Series(np.nan, index=row["response"].index)
            d2_n[d] = 0
            continue
        pred = d2_map[target].reindex(row["response"].index)
        frame = pd.concat([row["response"].rename("R"), row["chronos"].rename("Chronos"),
                           pred.rename("DEMETER2"), lineage], axis=1).dropna()
        d2_n[d] = int(len(frame))
        if len(frame) < MIN_N:
            d2_E[d] = pd.Series(np.nan, index=row["response"].index)
            chronos_subset_E[d] = pd.Series(np.nan, index=row["response"].index)
            continue
        idx = frame.index
        e_d2, _ = a3.ols_residual_multi(row["response"].reindex(idx), pd.DataFrame({"D2": pred.reindex(idx)}),
                                       lineage.reindex(idx), min_n=MIN_N)
        d2_E[d] = e_d2.reindex(row["response"].index)
        chronos_subset_E[d] = rc.linear_ols(row["response"].reindex(idx), row["chronos"].reindex(idx), lineage.reindex(idx)).reindex(row["response"].index)
    d2_pairs = pair_table(d2_E, target_map, drug_names)
    d2_base_pairs = pair_table(chronos_subset_E, target_map, drug_names)
    d2_targets = target_estimands(d2_pairs, target_map, eligible_targets, drug_data)
    d2_baseline_targets = target_estimands(d2_base_pairs, target_map, eligible_targets, drug_data)
    d2_summary = summarize_effect(d2_targets)
    d2_baseline_summary = summarize_effect(d2_baseline_targets)
    d2_retention = (d2_summary["median_delta"] / d2_baseline_summary["median_delta"]
                    if d2_summary.get("median_delta") is not None and d2_baseline_summary.get("median_delta") not in (None, 0) else None)
    d2_targets.to_csv(RUN / "n3_demeter2_target_results.tsv", sep="\t", index=False)
    d2_pairs.to_csv(RUN / "n3_demeter2_pairs.tsv", sep="\t", index=False)
    pd.DataFrame([{"compound_id": d, "target": target_map[d], "n_common_R_Chronos_D2_lineage": d2_n[d],
                   "status": "NA_missing_target_gene" if target_map[d] in d2_genes_missing else "ANALYZED" if d2_n[d] >= MIN_N else "NA_n_below_200"}
                  for d in drug_data]).to_csv(RUN / "n3_demeter2_drug_coverage.tsv", sep="\t", index=False)

    # Section 7 PRISM: first count frozen BRD/AUC coverage; compute residuals only if >=4 groups remain.
    prism_raw, prism_cov, prism_gate = load_prism_coverage(included, eligible_targets, chronos, lineage)
    prism_cov.to_csv(RUN / "n3_prism_compound_coverage.tsv", sep="\t", index=False)
    prism_result, prism_data, prism_pairs, prism_targets = prism_analysis(
        prism_raw, prism_cov, prism_gate, included, eligible_targets, chronos, lineage, drug_data, cells
    )
    if prism_gate["status"] != "LEGAL":
        prism_result["cross_resource_xS_xC"] = {"status": "BLOCKED_WITH_PRISM_COVERAGE", "run": False}

    # Consolidated candidate wording status; claim text and manuscript figures remain untouched.
    mantel_pass = mantel.get("p_one_sided") is not None and mantel["p_one_sided"] < 0.01
    nonkinase_positive = bool(target_rows.loc[target_rows["target"].isin(["ESR1", "TOP1"]), "delta_t"].gt(0).any())
    if main_result["status"] == "PASS" and mantel_pass and nonkinase_positive:
        n3_candidate = "ENABLE_WITHIN_TESTED_UNIVERSE: target-associated beyond scaffold similarity across kinase and non-kinase targets"
    elif main_result["status"] == "PASS" and mantel_pass:
        n3_candidate = "ENABLE_LIMITED: beyond chemical similarity in the tested target universe; do not claim non-kinase generality"
    else:
        n3_candidate = "DO_NOT_ENABLE_N3_TARGET_EXPANSION_SENTENCE"
    summary = {
        "analysis": "Phase 1b N3 target expansion; frozen cross-target validation",
        "spec_tag": "phase1r-n3-spec", "spec_commit": run_config["spec_commit"],
        "git_head_at_run_config": run_config["git_head"], "run_date_local": run_config["run_date_local"],
        "preliminary_attempts": run_config.get("preliminary_attempts", []),
        "regression_gate": {"status": gate["status"], "checks_passed": gate["checks_passed"], "checks_total": gate["checks_total"]},
        "frozen_table_sha256_match": True,
        "n_included_drugs": len(drug_data), "n_primary_targets": len(eligible_targets),
        "gdsc2_main": main_result, "gdsc2_target_results": target_rows.to_dict("records"),
        "target_label_permutation": permutation,
        "robustness": {
            "leave_one_drug_out": {"n_runs": len(loo_drug_df), "median_delta_range": [float(loo_drug_df["median_delta"].min()), float(loo_drug_df["median_delta"].max())],
                                   "n_status_pass": int(loo_drug_df["status"].eq("PASS").sum()), "n_status_fail": int(loo_drug_df["status"].eq("FAIL").sum())},
            "leave_one_target_out": {"n_runs": len(loo_target_df), "median_delta_range": [float(loo_target_df["median_delta"].min()), float(loo_target_df["median_delta"].max())]},
            "scaffold_matched": scaffold_result, "scaffold_fingerprint_provenance": scaffold_diag,
            "target_class": class_summary.to_dict("records"), "annotation_sensitivity": annotation,
        },
        "partial_mantel": mantel,
        "atlas_plane": atlas_result,
        "prism": prism_result,
        "spline_sensitivity": {"median_delta": spline_summary.get("median_delta"),
                                "ci95_target_bootstrap": spline_summary.get("ci95_target_bootstrap"),
                                "positive_fraction": spline_summary.get("positive_fraction"),
                                "retention_vs_linear_primary": spline_retention},
        "demeter2_sensitivity": {"missing_target_genes": d2_genes_missing,
                                  "n_drugs_with_at_least_200_common_models": int(sum(n >= MIN_N for n in d2_n.values())),
                                  "primary_median_delta_demeter2": d2_summary.get("median_delta"),
                                  "primary_median_delta_chronos_same_subset": d2_baseline_summary.get("median_delta"),
                                  "retention_vs_chronos_same_subset": d2_retention,
                                  "target_results": d2_targets.to_dict("records"),
                                  "mapping_diagnostics": d2_diag},
        "n3_claim_candidate_status": n3_candidate,
        "stop_conditions": {"regression_gate_failed": False, "run_config_missing_before_analysis": False,
                            "eligibility_sha_mismatch": False, "required_universe_or_threshold_change": False},
        "limitations": ["Only six target groups contribute to GDSC2 target-balanced inference.",
                        "PRISM validation follows the frozen four-group coverage gate.",
                        "RDKit runtime is 2025.9.2 while eligibility provenance records 2025.09.6; within-group similarity consistency is reported explicitly."],
    }
    write_json(RUN / "n3_results.json", summary)
    write_readme(summary, permutation, loo_drug_df, loo_target_df, scaffold_bins, mantel, atlas_points,
                 prism_cov, spline_targets, spline_retention, d2_targets, d2_retention, drug_data)
    write_stage_report(summary, target_rows, loo_drug_df, loo_target_df, scaffold_bins,
                       class_summary, atlas_points, prism_cov, spline_targets, d2_targets)
    write_summary_bundle(summary, run_config)
    write_provenance(run_config)
    print(json.dumps({"GDSC2_main": main_result, "Mantel": mantel, "scaffold_matched": scaffold_result,
                      "Atlas_plane": atlas_result, "PRISM_coverage": prism_gate,
                      "N3_candidate": n3_candidate, "report": str(REV / "STAGE_3_N3_REPORT.md")}, indent=2), flush=True)


def write_readme(summary, permutation, loo_drug, loo_target, scaffold_bins, mantel, atlas_points,
                 prism_cov, spline_targets, spline_retention, d2_targets, d2_retention, drug_data):
    run_md = RUN / "README.md"
    main = summary["gdsc2_main"]
    lines = [
        "# Phase 1b N3 target expansion",
        "",
        f"Run date: {summary['run_date_local']}  ",
        f"Spec: `{summary['spec_tag']}` at `{summary['spec_commit']}`; analysis HEAD `{summary['git_head_at_run_config']}`  ",
        f"Regression gate: **{summary['regression_gate']['status']} {summary['regression_gate']['checks_passed']}/{summary['regression_gate']['checks_total']}**  ",
        "Run config was written before residual calculations. Frozen eligibility/scaffold inputs matched their preregistered SHA-256 values.",
        "",
        "## GDSC2 primary analysis",
        "",
        f"43 included compounds; six frozen multi-scaffold target groups. Median Δₜ = **{main['median_delta']:.4f}**, target-bootstrap 95% CI [{main['ci95_target_bootstrap'][0]:.4f}, {main['ci95_target_bootstrap'][1]:.4f}], positive targets {main['positive_targets']}/{main['n_targets']}; decision **{main['status']}**.",
        f"Target-label permutation p = {permutation['p_one_sided']:.5g} ({permutation['n_permutations_valid']}/{permutation['n_permutations_requested']} valid). Labels were permuted over all included compounds preserving group sizes; residuals were selected from exact precomputed OLS fits for each drug × assigned target gene.",
        "",
        "See `n3_target_results.tsv`, `n3_drug_results.tsv`, `n3_pairs_gdsc2.tsv`, and `n3_gdsc2_null_distribution.tsv`.",
        "The all-target table and figure retain all 30 annotated target labels; singletons show Wₜ as NA and remain cross-target comparators.",
        "Frozen Phase 1 overlap drugs retained: erlotinib (DPC-002573), gefitinib (DPC-002973), and osimertinib (DPC-000819).",
        "",
        "## Required robustness and chemical-similarity control",
        "",
        f"Leave-one-drug-out runs: {len(loo_drug)}; leave-one-target-out runs: {len(loo_target)}. See their TSV files.",
        f"Scaffold-matched stratified pair-label permutation: {summary['robustness']['scaffold_matched'].get('status')}, observed equal-bin-weight mean Δmedian = {summary['robustness']['scaffold_matched'].get('observed_stratified_difference')}, one-sided p = {summary['robustness']['scaffold_matched'].get('p_one_sided')}. Per-bin results: `n3_scaffold_matched_bins.tsv`.",
        f"Runtime RDKit {rdBase.rdkitVersion}; frozen eligibility records RDKit 2025.09.6. Morgan fingerprints used radius 2 / 2048 bits on frozen scaffold SMILES. Recomputed in-target Tanimoto values matched the frozen three-decimal values for {summary['robustness']['scaffold_fingerprint_provenance']['n_frozen_within_target_values_equal_after_3dp_rounding']}/{summary['robustness']['scaffold_fingerprint_provenance']['n_frozen_within_target_pair_comparisons']} comparisons; this package-version discrepancy is retained as a deviation.",
        f"Partial Mantel: r = {mantel['observed_partial_mantel_r']:.4f}, one-sided drug-label permutation p = {mantel['p_one_sided']:.5g}; controls the upper-triangle target-residual Spearman matrix for scaffold Morgan Tanimoto.",
        "Target-class and annotation sensitivity results are in the corresponding TSV/JSON files. Non-kinase class inference is BLOCKED because only ESR1 and TOP1 (two target groups) are in the frozen eligible set; no second full annotation source was available.",
        "",
        "## Atlas plane",
        "",
        f"The merged plane contains {len(atlas_points)} points, counting Phase 1 EGFR and Phase 1b EGFR separately. Class I present={summary['atlas_plane']['class_I_present']}; class II present={summary['atlas_plane']['class_II_present']}; Spearman 95% bootstrap CI={summary['atlas_plane']['spearman_bootstrap_ci95']}; preserve the independent-axes criterion={summary['atlas_plane']['two_axes_independence_retained']}.",
        "The new atlas sketch is `revision_figs/N3_atlas_plane.png` and run-specific PDF/SVG/PNG files; the frozen manuscript figure files were not touched.",
        "",
        "## PRISM secondary AUC",
        "",
        f"Coverage: {summary['prism']['coverage']['status']}, {summary['prism']['coverage']['n_covered_groups']}/{summary['prism']['coverage']['coverage_gate_required']} target groups meet ≥2 compounds at MIN_N=200.",
        (f"PRISM target-level {summary['prism']['primary']['status']}: median Δ={summary['prism']['primary']['median_delta']:.4f}, "
         f"95% CI={summary['prism']['primary']['ci95_target_bootstrap']}, positive {summary['prism']['primary']['positive_targets']}/{summary['prism']['primary']['n_targets']}; "
         f"permutation p={summary['prism']['target_label_permutation']['p_one_sided']:.5g}. "
         f"Cross-resource xS={summary['prism']['cross_resource_xS_xC']['xS_median']:.4f}, "
         f"xC={summary['prism']['cross_resource_xS_xC']['xC_median']:.4f}, common cells={summary['prism']['cross_resource_xS_xC']['n_common_cells']}."
         if summary['prism']['coverage']['status'] == 'LEGAL' else
         "No PRISM residual analysis ran because the frozen coverage gate failed; no LFC substitute or MIN_N relaxation was used."),
        "",
        "## Sensitivity analyses",
        "",
        f"Spline residualizer: target-balanced median Δ={summary['spline_sensitivity']['median_delta']}, retention vs linear={spline_retention}.",
        f"DEMETER2: same per-drug R/Chronos/D2/lineage complete-case subset; median Δ={summary['demeter2_sensitivity']['primary_median_delta_demeter2']}, retention vs Chronos on the same subset={d2_retention}; missing target genes={summary['demeter2_sensitivity']['missing_target_genes']}.",
        "",
        "## Provenance and files",
        "",
        "Frozen eligibility, exclusion, target-group and scaffold tables are copied unchanged into this directory. `run_config.yaml` records input checksums, release paths, versions, seed, and HEAD. `provenance.json` records the analysis code hash and generated-file hashes.",
        "",
        "No claim text or manuscript figure was edited. No Git commit or tag was created.",
        "",
    ]
    run_md.write_text("\n".join(lines), encoding="utf-8")


def write_summary_bundle(summary, run_config):
    compact = {
        "module": "Phase 1b N3 target expansion",
        "spec_tag": summary["spec_tag"], "spec_commit": summary["spec_commit"],
        "run_config": "03_analysis/2026-09-28_phase1b_target_expansion/run_config.yaml",
        "regression_gate": summary["regression_gate"],
        "n_included_drugs": summary["n_included_drugs"], "n_primary_targets": summary["n_primary_targets"],
        "GDSC2": {"status": summary["gdsc2_main"]["status"],
                  "median_delta": summary["gdsc2_main"]["median_delta"],
                  "ci95": summary["gdsc2_main"]["ci95_target_bootstrap"],
                  "positive_fraction": summary["gdsc2_main"]["positive_fraction"],
                  "permutation_p_one_sided": summary["target_label_permutation"]["p_one_sided"],
                  "target_results": summary["gdsc2_target_results"]},
        "partial_mantel": summary["partial_mantel"],
        "scaffold_matched": summary["robustness"]["scaffold_matched"],
        "atlas_plane": summary["atlas_plane"], "PRISM": summary["prism"],
        "spline_sensitivity": summary["spline_sensitivity"],
        "DEMETER2_sensitivity": summary["demeter2_sensitivity"],
        "N3_claim_candidate_status": summary["n3_claim_candidate_status"],
        "limitations": summary["limitations"],
        "summary_path": "03_analysis/2026-09-28_phase1b_target_expansion/n3_results.json",
    }
    write_json(ROOT / "discordance" / "phase1b_target_expansion_summary.json", compact)


def markdown_table(headers, rows):
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def fmt(v, digits=4):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def write_stage_report(summary, target_rows, loo_drug, loo_target, scaffold_bins,
                       class_summary, atlas_points, prism_cov, spline_targets, d2_targets):
    main = summary["gdsc2_main"]
    lo, hi = main["ci95_target_bootstrap"]
    primary_table = markdown_table(
        ["靶点", "药物数", "scaffold 数", "Wₜ", "Cₜ", "Δₜ"],
        [[r.target, r.n_drugs, r.n_scaffolds, fmt(r.W_t), fmt(r.C_t), fmt(r.delta_t)]
         for r in target_rows.itertuples(index=False)],
    )
    loo_drug_table = markdown_table(
        ["移除药物", "靶点", "可估靶点数", "median Δ", "判定"],
        [[r.removed_name, r.removed_target, r.n_targets_estimable, fmt(r.median_delta), r.status]
         for r in loo_drug.itertuples(index=False)],
    )
    loo_target_table = markdown_table(
        ["移除靶点", "剩余药物数", "可估靶点数", "median Δ", "判定"],
        [[r.removed_target, r.n_drugs_remaining, r.n_targets_estimable, fmt(r.median_delta), r.status]
         for r in loo_target.itertuples(index=False)],
    )
    scaffold_table = markdown_table(
        ["Morgan Tanimoto 分箱", "同靶点对 n", "跨靶点对 n", "同靶点 median ρ", "跨靶点 median ρ", "差"],
        [[r.similarity_bin, r.n_same_target_pairs, r.n_cross_target_pairs, fmt(r.median_same_rho),
          fmt(r.median_cross_rho), fmt(r.median_difference)] for r in scaffold_bins.itertuples(index=False)],
    )
    class_table = markdown_table(
        ["类别", "靶点数", "median Δₜ", "各靶点 Δₜ", "状态"],
        [[r.target_class, r.n_targets, fmt(r.median_delta_t), r.target_deltas, r.status]
         for r in class_summary.itertuples(index=False)],
    )
    prism_gate = summary["prism"]["coverage"]
    prism_rows = [[r.target, int((prism_cov.target.eq(r.target) & prism_cov.status.eq("LEGAL")).sum())]
                  for r in target_rows.itertuples(index=False)]
    prism_table = markdown_table(["冻结多 scaffold 靶点", "PRISM 合法化合物数"], prism_rows)
    prism_primary = summary["prism"].get("primary", {})
    if prism_gate["status"] == "LEGAL":
        prism_target_df = pd.DataFrame(summary["prism"].get("target_level", []))
        prism_target_table = markdown_table(
            ["靶点", "合法药物数", "scaffold 数", "Wₜ", "Cₜ", "Δₜ"],
            [[r.target, r.n_drugs, r.n_scaffolds, fmt(r.W_t), fmt(r.C_t), fmt(r.delta_t)]
             for r in prism_target_df.itertuples(index=False)],
        )
        prism_cross = summary["prism"]["cross_resource_xS_xC"]
        prism_result_text = (f"PRISM status **{prism_primary['status']}**；median Δₜ={fmt(prism_primary['median_delta'])}，"
                             f"95% CI [{fmt(prism_primary['ci95_target_bootstrap'][0])}, {fmt(prism_primary['ci95_target_bootstrap'][1])}]，"
                             f"正向 {prism_primary['positive_targets']}/{prism_primary['n_targets']}；"
                             f"标签置换 p={fmt(summary['prism']['target_label_permutation']['p_one_sided'], 5)}。"
                             f"共同细胞上的跨资源 xS={fmt(prism_cross['xS_median'])}、xC={fmt(prism_cross['xC_median'])} "
                             f"（{prism_cross['n_common_cells']} cells，{prism_cross['n_shared_drugs']} shared drugs）。")
    else:
        prism_target_table = "PRISM residual analysis not run because the frozen coverage threshold was not met."
        prism_result_text = ""
    atlas_table = markdown_table(
        ["靶点", "版本", "x=concordance", "y=Wₜ", "类别", "药物数"],
        [[r.target, r.source, fmt(r.x_mean_concordance_partial), fmt(r.y_within_target_W), r.atlas_class, r.n_drugs]
         for r in atlas_points.itertuples(index=False)],
    )
    spline_table = markdown_table(
        ["靶点", "Wₜ", "Cₜ", "Δₜ"],
        [[r.target, fmt(r.W_t), fmt(r.C_t), fmt(r.delta_t)] for r in spline_targets.itertuples(index=False)],
    )
    d2_table = markdown_table(
        ["靶点", "Wₜ", "Cₜ", "Δₜ"],
        [[r.target, fmt(r.W_t), fmt(r.C_t), fmt(r.delta_t)] for r in d2_targets.itertuples(index=False)],
    )
    robust = summary["robustness"]
    mantel = summary["partial_mantel"]
    atlas = summary["atlas_plane"]
    prism = summary["prism"]
    report = f'''# Stage 3 N3 执行报告

日期：2026-09-28  
规范：`revision/STAGE3_N3_SPEC.md`，tag `{summary['spec_tag']}`（`{summary['spec_commit'][:7]}`）  
执行目录：`03_analysis/2026-09-28_phase1b_target_expansion/`

## 执行状态

    - 运行前确认 `git log --oneline -2`：`{summary['spec_commit'][:7]} Freeze N3 spec; sign Phase 1b decision 4 before any Phase 1b residuals`；当前 HEAD/tag 均为 `{summary['git_head_at_run_config'][:7]}`。
- `run_config.yaml` 在任何 Phase 1b 残差计算前写入，记录了资格表、靶点组、scaffold、manifest 与分析输入 SHA-256、软件版本、seed 20260927 和 HEAD。
- 冻结资格表 SHA 匹配；回归门槛 **{summary['regression_gate']['status']} {summary['regression_gate']['checks_passed']}/{summary['regression_gate']['checks_total']}**。
- 六个冻结多 scaffold 组和 43 个 included 药物均按原表纳入；没有改药物、靶点或阈值。主分析 FAIL 不属于停止条件，已继续完成全部模块。
- 没有执行 Git 写操作，没有改 claim 文本或既有论文图；N3 草图另存。

## GDSC2 主分析

对象为 43 个冻结 included 化合物。每个药物以 `R ~ D_primary_target + lineage` 计算 OLS 残差；`MIN_N=200`，药物对 Spearman 至少 30 个共同模型。结果按六个多 scaffold 靶点组等权汇总。

**median Δₜ = {fmt(main['median_delta'])}，靶点 bootstrap 95% CI [{fmt(lo)}, {fmt(hi)}]；{main['positive_targets']}/{main['n_targets']} 个 Δₜ>0；Discovery 判定：{main['status']}。** 置换检验 p={fmt(summary['target_label_permutation']['p_one_sided'],5)}（10,000 次，保留 43 个药物的原靶点组大小；对每个药物×可分配靶点的 OLS 残差预计算后，在每次分配中重新取对应残差并重算统计量）。

{primary_table}

完整逐药、全部 30 个靶点描述值和 903 个 pair 结果见 `n3_drug_results.tsv`、`n3_all_target_descriptives.tsv`、`n3_pairs_gdsc2.tsv`。单药靶点保留为 cross-target 对照，其 Wₜ 标为 NA，没有从表或图中隐藏。

与 Phase 1 重叠的药物照常纳入：erlotinib（DPC-002573）、gefitinib（DPC-002973）、osimertinib（DPC-000819）。

## 强制稳健性

Leave-one-drug-out 共 {len(loo_drug)} 次，median Δ 范围 [{fmt(loo_drug['median_delta'].min())}, {fmt(loo_drug['median_delta'].max())}]；每次结果如下。

{loo_drug_table}

Leave-one-target-out 共 {len(loo_target)} 次：

{loo_target_table}

### Scaffold 相似度匹配

每个冻结 scaffold 用 Morgan radius 2 / 2048-bit 指纹，按 [0,0.2)、[0.2,0.4)、[0.4,0.7) 分箱。分层统计采用每个可估分箱的 median(同靶点 ρ)−median(跨靶点 ρ) 等权平均；置换在各箱内打乱 pair 的同靶点标签、保持箱内标签数不变，单侧 p={fmt(robust['scaffold_matched'].get('p_one_sided'),5)}。规范未定义分箱差的合并权重，采用等箱权重并在此披露。Pair 是共享药物的 dyad，不能视作独立观察；同时报告药物标签层面的偏 Mantel结果。

{scaffold_table}

**版本偏离：**资格表记录 RDKit 2025.09.6，执行运行时是 {robust['scaffold_fingerprint_provenance']['runtime_rdkit_version']}。输入仍使用冻结 scaffold 字符串；冻结组内的三位小数 Tanimoto 有 {robust['scaffold_fingerprint_provenance']['n_frozen_within_target_values_equal_after_3dp_rounding']}/{robust['scaffold_fingerprint_provenance']['n_frozen_within_target_pair_comparisons']} 项复算一致。跨靶点指纹相似度在当前版本计算，因此此版本差异是需要保留的技术限制。

### 靶点类别与注释敏感性

{class_table}

注释敏感性为 **BLOCKED**：没有第二套完整且有文档的冻结 primary-target 注释；未另行挑选或改注释。

## 化学相似度调整的偏 Mantel

对 43 个药物的残差 Spearman 矩阵、同靶点矩阵与 scaffold Tanimoto 矩阵做偏 Pearson/Mantel 相关，控制化学相似度；药物标签置换时同步改变同靶点矩阵的行列结构。结果 r={fmt(mantel['observed_partial_mantel_r'])}，单侧 p={fmt(mantel['p_one_sided'],5)}（{mantel['n_permutations_valid']} 次有效置换）。完整零分布在 `n3_partial_mantel_null_distribution.tsv`。

## Atlas 平面扩展

{atlas_table}

Class I 存在：{atlas['class_I_present']}；Class II 存在：{atlas['class_II_present']}；x/y Spearman={fmt(atlas['spearman_x_y'])}，靶点 bootstrap 95% CI [{fmt(atlas['spearman_bootstrap_ci95'][0])}, {fmt(atlas['spearman_bootstrap_ci95'][1])}]。两个轴独立判据 **{'保留' if atlas['two_axes_independence_retained'] else '不满足'}**。Phase 1 与 Phase 1b 的 EGFR 分别作为两个点。新增草图：`revision_figs/N3_atlas_plane.png`；原论文图未替换。

## PRISM Secondary 20Q2 AUC

覆盖门槛：**{prism_gate['status']}**；六个冻结多 scaffold 组中 {prism_gate['n_covered_groups']} 组达到每组至少 2 个合法化合物，规范要求至少 4 组。

{prism_table}

{prism_result_text}

{prism_target_table}

{'完整 PRISM pair 级结果见 `n3_prism_pairs.tsv`、`n3_prism_null_distribution.tsv` 和 `n3_cross_resource_xs_xc.tsv`。' if prism_gate['status']=='LEGAL' else '因覆盖不足，未计算 PRISM 残差或 PRISM target-level 统计，也未计算跨资源 xS/xC；未换用 LFC、未降低 MIN_N。'}

## 描述性残差化敏感性

### A2 spline

{spline_table}

Spline median Δ={fmt(summary['spline_sensitivity']['median_delta'])}，相对线性主效应保留率={fmt(summary['spline_sensitivity']['retention_vs_linear_primary'])}。这是描述性敏感性，不改变主判定。

### A3(c) DEMETER2

{d2_table}

DEMETER2 同一 drug 的 R、Chronos、D2、lineage 完整病例子集上比较 D2 残差与同子集 Chronos 残差。目标层 median Δ：D2={fmt(summary['demeter2_sensitivity']['primary_median_delta_demeter2'])}；Chronos 同子集={fmt(summary['demeter2_sensitivity']['primary_median_delta_chronos_same_subset'])}；保留率={fmt(summary['demeter2_sensitivity']['retention_vs_chronos_same_subset'])}。缺失基因：{', '.join(summary['demeter2_sensitivity']['missing_target_genes']) if summary['demeter2_sensitivity']['missing_target_genes'] else '无'}。按冻结规则缺少基因的靶点按 NA 处理。

## Claim 候选状态

| 候选句 / 决策 | 状态 | 依据 |
|---|---|---|
| 扩展到预先定义的靶点平衡宇宙 | **{main['status']}** | median Δ={fmt(main['median_delta'])}，CI [{fmt(lo)}, {fmt(hi)}]，正向 {main['positive_targets']}/{main['n_targets']} |
| “target-associated, beyond scaffold similarity, across kinase and non-kinase targets” | **{'可作为候选交审' if summary['n3_claim_candidate_status'].startswith('ENABLE_WITHIN') else '不启用'}** | 要求主分析 PASS、Mantel p<0.01，且 ESR1/TOP1 至少一组 Δ>0；本次状态：{summary['n3_claim_candidate_status']} |
| “beyond chemical similarity in the tested target universe” | **{'有限候选' if summary['n3_claim_candidate_status'].startswith('ENABLE_LIMITED') else '不启用'}** | 仅主分析 PASS 且 Mantel p<0.01、但无非激酶正向靶点时可用 |
| Fig. 3 两轴独立 | **{'判据保留，交审' if atlas['two_axes_independence_retained'] else '判据未保留；交 Claude 审查措辞'}** | 合并平面分类与跨靶点 x/y bootstrap Spearman CI |
| PRISM 外部验证 | **{prism_gate['status']}** | 按四组、每组两个合法化合物的覆盖门槛 |

N3 候选状态机器标签：`{summary['n3_claim_candidate_status']}`。按规范，若 Discovery FAIL，稿件解释应限于 Phase 1 panel；不追加靶点、不改注释。此报告没有改写论文 claim。

## 偏离、停止规则与新文件

- 偏离：scaffold Tanimoto 复算使用 RDKit {robust['scaffold_fingerprint_provenance']['runtime_rdkit_version']}，与冻结资格表版本 2025.09.6 不同；被冻结组内相似度三位小数复核结果及影响已上报。分层匹配检验采用等权平均各可估箱内中位数差，因为规范未冻结箱间合并权重。
- 偏离记录：`run_config.yaml` 保留三次未完成尝试（资格/scaffold 合并字段读取、macOS GUI 后端、PRISM 响应索引对齐）；每次均在预设统计停止条件之外修复，后续完整运行使用确定性 seed 重算。最终配置记录最终 entrypoint 与 gate SHA。
- PRISM 若未达到覆盖门槛则按 BLOCKED_BY_COVERAGE 跳过，不作为全项目停止条件。
- 停止规则核对：regression FAIL=False；run_config missing=False；资格表 SHA mismatch=False；需更改药物/靶点/阈值=False。
- 新产物位于 `03_analysis/2026-09-28_phase1b_target_expansion/`，包含 README、run config、冻结表副本、主分析 pair/target/drug 表、10,000 次零分布、稳健性表、Mantel/Atlas/敏感性结果、图、结果 JSON 与 provenance SHA 清单。
- `N3_eligibility_exclusion_reasons.*` 展示冻结 `excluded`/`unresolved` 的所有 reason_code 汇总；逐化合物原因见随包 eligibility/exclusion TSV。
- 额外稳定摘要：`discordance/phase1b_target_expansion_summary.json`。建议后续版本控制提交说明：`Add Phase 1b N3 target expansion analysis and report`。未执行 commit/tag。
'''
    (REV / "STAGE_3_N3_REPORT.md").write_text(report, encoding="utf-8")


def write_provenance(run_config):
    generated = sorted(p for p in RUN.rglob("*") if p.is_file() and p.name != "provenance.json")
    source_files = [Path(__file__), REVISION_CODE / "rev_common.py", REVISION_CODE / "rev_A2_nonlinear.py",
                    REVISION_CODE / "rev_A3_attenuation.py", ROOT / "discordance" / "build_phase1b_chembl36_eligibility.py"]
    record = {
        "run_config_sha256": sha256(RUN / "run_config.yaml"),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "software_versions": run_config["software_versions"],
        "generated_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in generated},
    }
    write_json(RUN / "provenance.json", record)


if __name__ == "__main__":
    main()
