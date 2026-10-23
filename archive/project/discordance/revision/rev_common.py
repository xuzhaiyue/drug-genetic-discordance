#!/usr/bin/env python3
"""Shared Stage 2A data loading and frozen Atlas calculations.

The Atlas calculations below intentionally delegate to the original scripts.
No frozen result file is written by this module.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEP_MAP = Path("/path/to/depmap")
REVISION = ROOT / "discordance" / "revision"
sys.path.insert(0, str(ROOT / "discordance"))

import run_discordance_atlas_gdsc2 as gdsc2  # noqa: E402
import run_discordance_atlas_prism as prism  # noqa: E402

MIN_N = gdsc2.MIN_N
PANEL = gdsc2.PANEL
MULTI_CHEMO = gdsc2.MULTI_CHEMO


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "patsy"):
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            result[pkg] = "missing"
    return result


def sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_metadata(inputs: list[str | Path], seed: int | None = None,
                   n_perm: int | None = None) -> dict:
    head = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    cached_hashes: dict[str, str] = {}
    regression_path = REVISION / "rev_common_regression.json"
    if regression_path.exists():
        try:
            cached_hashes = json.loads(regression_path.read_text()).get("input_sha256", {})
        except (OSError, json.JSONDecodeError):
            cached_hashes = {}
    hashes = {}
    for item in inputs:
        key = str(Path(item))
        hashes[key] = cached_hashes.get(key) or sha256(item)
    return {
        "git_commit": head,
        "prereg_tag": "phase1r-prereg",
        "spec_tag": "phase1r-impl-spec",
        "input_sha256": hashes,
        "seed": seed,
        "n_perm": n_perm,
        "software_versions": package_versions(),
        "runtime_utc": datetime.now(timezone.utc).isoformat(),
    }


def _models_lineage() -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(gdsc2.MODEL_PATH, usecols=["ModelID", "OncotreeLineage"])
    models["ModelID"] = models["ModelID"].astype(str)
    series = models.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"]
    # Preserve the frozen get_dummies construction, including its full lineage basis.
    lin = pd.get_dummies(series, prefix="lin", dummy_na=False)
    return models, lin


def load_resource(resource: str) -> dict:
    """Load drug response, Chronos D, lineage dummies and panel for one resource."""
    name = resource.upper()
    if name not in {"GDSC2", "PRISM"}:
        raise ValueError("resource must be 'GDSC2' or 'PRISM'")
    chronos = gdsc2.load_chronos()
    _, lin = _models_lineage()
    panel = [(d, t, c) for d, t, c in PANEL]
    raw_response: dict[str, pd.Series] = {}
    atlas_valid_cells = pd.Index([], dtype=object)
    input_paths: list[Path] = [gdsc2.CRISPR_PATH, gdsc2.MODEL_PATH]

    if name == "GDSC2":
        auc_path = gdsc2.RAW / "GDSC2AUCMatrix.csv"
        auc = pd.read_csv(auc_path, index_col=0)
        auc.index = auc.index.astype(str)
        eligible = auc.index.intersection(chronos.index).intersection(lin.index)
        atlas_valid_cells = eligible
        for drug, target, cid in panel:
            if cid not in auc.columns or target not in chronos.columns:
                continue
            cells = eligible
            raw_response[drug] = gdsc2.response_vector(auc.loc[cells, cid])
        input_paths.append(auc_path)
    else:
        coverage_path = prism.OUT / "prism_coverage.json"
        coverage = json.loads(coverage_path.read_text())
        cov = coverage["secondary_20Q2_auc"]
        if cov.get("verdict") != "LEGAL":
            raise RuntimeError("PRISM secondary is not LEGAL in frozen coverage audit")
        legal = {r["drug"] for r in cov["drugs"] if r["legal"]}
        auc_map = prism.load_secondary_auc(legal)
        for drug, target, _ in panel:
            if drug not in legal or drug not in auc_map or target not in chronos.columns:
                continue
            cells = auc_map[drug].index.intersection(chronos.index).intersection(lin.index)
            raw_response[drug] = gdsc2.response_vector(auc_map[drug].reindex(cells))
        if raw_response:
            for series in raw_response.values():
                atlas_valid_cells = atlas_valid_cells.union(series.index, sort=False)
        input_paths.extend([
            prism.COMPOUNDS,
            prism.PRISM_DIR / prism.URLS["secondary_curves"][0],
            coverage_path,
        ])

    # Keep the original frozen-script cell order separately for every drug. A shared
    # rectangular DataFrame is useful downstream, but must not dictate OLS row order.
    index = chronos.index.intersection(lin.index)
    R = pd.DataFrame(index=index)
    D = pd.DataFrame(index=index)
    R_source: dict[str, pd.Series] = {}
    D_source: dict[str, pd.Series] = {}
    cell_orders: dict[str, pd.Index] = {}
    drug_panel = []
    for drug, target, cid in panel:
        if drug not in raw_response:
            continue
        source_order = raw_response[drug].index
        r = raw_response[drug].reindex(index)
        d = -chronos[target].reindex(index).astype(float)
        valid = r.notna() & d.notna() & lin.reindex(index).notna().all(axis=1)
        if int(valid.sum()) < MIN_N:
            continue
        R[drug] = r
        D[drug] = d
        R_source[drug] = raw_response[drug]
        D_source[drug] = -chronos.loc[source_order, target].astype(float)
        cell_orders[drug] = source_order
        drug_panel.append({"drug": drug, "target": target, "compound_id": cid})
    return {
        "resource": name,
        "R": R,
        "D": D,
        "R_source": R_source,
        "D_source": D_source,
        "cell_orders": cell_orders,
        "chronos": chronos.reindex(index),
        "lineage": lin.reindex(index),
        "panel": drug_panel,
        "min_n": MIN_N,
        "input_paths": input_paths,
        "cells": index,
        "atlas_valid_cells": atlas_valid_cells,
    }


def linear_ols(R: pd.Series, D: pd.Series | pd.DataFrame,
               L: pd.DataFrame) -> pd.Series:
    if isinstance(D, pd.DataFrame):
        if D.shape[1] != 1:
            raise ValueError("linear_ols expects one D series per drug")
        D = D.iloc[:, 0]
    return gdsc2.ols_residual(R, D, L)


def compute_E(resource: dict, residualizer: Callable | None = None,
              D_source: str = "chronos", cells: pd.Index | None = None) -> pd.DataFrame:
    """Compute per-drug residuals using a common caller-supplied residualizer."""
    residualizer = residualizer or linear_ols
    data = resource
    output: dict[str, pd.Series] = {}
    for meta in data["panel"]:
        drug = meta["drug"]
        source_order = data.get("cell_orders", {}).get(drug, data["R"][drug].index)
        if cells is None:
            idx = source_order
        else:
            allowed = set(cells)
            idx = source_order[source_order.isin(allowed)]
        response_source = data.get("R_source", data["R"])
        R = response_source[drug].reindex(idx)
        if D_source == "chronos":
            dependency_source = data.get("D_source", data["D"])
            D = dependency_source[drug].reindex(idx)
        else:
            source = data[D_source]
            target = meta["target"]
            D = source[target].reindex(idx) if isinstance(source, pd.DataFrame) else source.reindex(idx)
        L = data["lineage"].reindex(idx)
        output[drug] = residualizer(R, D, L)
    # Each resource/drug may cover a different cell set (notably PRISM). Let pandas
    # form the union of Series indexes instead of truncating every drug to the last
    # loop iteration's cell index.
    return pd.DataFrame(output)


def pair_table(E: pd.DataFrame, panel: list[dict]) -> pd.DataFrame:
    targets = {p["drug"]: p["target"] for p in panel}
    drugs = [d for d in E.columns if d in targets]
    rows = []
    from itertools import combinations
    for a, b in combinations(drugs, 2):
        rho, n = gdsc2.spearman_pair(E[a], E[b])
        if np.isfinite(rho):
            rows.append({"drug_a": a, "drug_b": b,
                         "target_a": targets[a], "target_b": targets[b],
                         "same_target": targets[a] == targets[b],
                         "n": int(n), "rho": float(rho)})
    return pd.DataFrame(rows, columns=["drug_a", "drug_b", "target_a", "target_b",
                                       "same_target", "n", "rho"])


def summary(pairs: pd.DataFrame) -> dict:
    same = pairs.loc[pairs["same_target"], "rho"].to_numpy(dtype=float)
    cross = pairs.loc[~pairs["same_target"], "rho"].to_numpy(dtype=float)
    return {
        "same_median": float(np.median(same)) if len(same) else None,
        "cross_median": float(np.median(cross)) if len(cross) else None,
        "n_same": int(len(same)),
        "n_cross": int(len(cross)),
        "mw_p": float(stats.mannwhitneyu(same, cross, alternative="greater").pvalue)
        if len(same) and len(cross) else None,
        "per_target_median": {
            str(target): float(np.median(group["rho"].to_numpy(dtype=float)))
            for target, group in pairs.loc[pairs["same_target"]].groupby("target_a", sort=True)
        },
    }


def drug_label_perm(E: pd.DataFrame, panel: list[dict], n_perm: int = 10000,
                    seed: int = 20260927) -> dict:
    """Permute target labels among all resource-available panel drugs."""
    pairs = pair_table(E, panel)
    drugs = list(E.columns)
    meta = {p["drug"]: p["target"] for p in panel if p["drug"] in drugs}
    drugs = [d for d in drugs if d in meta]
    # Include all target classes and rail drugs by permuting the full label vector.
    labels = np.array([meta[d] for d in drugs], dtype=object)
    pos = {d: i for i, d in enumerate(drugs)}
    rows = pairs[pairs["drug_a"].isin(drugs) & pairs["drug_b"].isin(drugs)]
    i = rows["drug_a"].map(pos).to_numpy(dtype=int)
    j = rows["drug_b"].map(pos).to_numpy(dtype=int)
    rho = rows["rho"].to_numpy(dtype=float)
    obs_mask = labels[i] == labels[j]
    observed = float(np.median(rho[obs_mask])) if obs_mask.any() else float("nan")
    rng = np.random.default_rng(seed)

    def run_perm(indices: np.ndarray, target_drop: str | None = None) -> dict:
        keep_pair = np.ones(len(rho), dtype=bool)
        if target_drop is not None:
            keep_drug = np.array([x != target_drop for x in labels])
            keep_pair = keep_drug[i] & keep_drug[j]
            labels_sub = labels[keep_drug]
            remap = np.full(len(labels), -1, dtype=int)
            remap[np.where(keep_drug)[0]] = np.arange(int(keep_drug.sum()))
            ii, jj = remap[i], remap[j]
        else:
            labels_sub = labels
            ii, jj = i, j
        valid_rho = rho[keep_pair]
        ii, jj = ii[keep_pair], jj[keep_pair]
        obs = labels_sub[ii] == labels_sub[jj]
        obs_med = float(np.median(valid_rho[obs])) if obs.any() else float("nan")
        perm_vals = np.empty(n_perm, dtype=float)
        for k in range(n_perm):
            shuffled = rng.permutation(labels_sub)
            same = shuffled[ii] == shuffled[jj]
            perm_vals[k] = np.median(valid_rho[same]) if same.any() else np.nan
        perm_vals = perm_vals[np.isfinite(perm_vals)]
        p = float((1 + np.count_nonzero(perm_vals >= obs_med)) / (1 + len(perm_vals)))
        return {"observed_median_same": obs_med, "p_drugperm": p,
                "n_perm_valid": int(len(perm_vals)),
                "perm_median_null_median": float(np.median(perm_vals)) if len(perm_vals) else None}

    all_result = run_perm(np.arange(len(labels)))
    loo = {}
    for target in ("EGFR", "AKT1", "MET", "MAPK1"):
        keep = np.array([lab != target for lab in labels])
        sublabels = labels[keep]
        remap = np.full(len(labels), -1, dtype=int)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        pairkeep = keep_pair = keep[i] & keep[j]
        ii, jj, rr = remap[i[pairkeep]], remap[j[pairkeep]], rho[pairkeep]
        same = sublabels[ii] == sublabels[jj]
        cross = ~same
        perm_result = run_perm(np.arange(len(labels)), target_drop=target)
        loo[target] = {
            "same_median": float(np.median(rr[same])) if same.any() else None,
            "cross_median": float(np.median(rr[cross])) if cross.any() else None,
            "n_same": int(same.sum()), "n_cross": int(cross.sum()),
            **perm_result,
        }
    return {"observed_median_same": observed, **all_result,
            "n_drugs": len(drugs), "target_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            "leave_one_target_out": loo}
