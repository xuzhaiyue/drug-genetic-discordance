#!/usr/bin/env python3
"""Target-anchor decomposition and matched non-target negative control (see SPEC.md)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
N3 = ROOT / "03_analysis" / "2026-09-28_phase1b_target_expansion"
sys.path.insert(0, str(ROOT / "discordance" / "revision"))
sys.path.insert(0, str(ROOT / "discordance"))
import rev_common as rc  # noqa: E402
from audit_prism_coverage import brd_core  # noqa: E402

MIN_N = 200
MIN_PAIR_N = 30
N_NULL = 100
N_AGG = 10_000
SEED = 20261029
G_COVERAGE = 0.80
FOCAL = ["EGFR", "MTOR", "ESR1", "TOP1", "ATR", "CHEK1"]
MERGED_GROUPS = {"FKBP1A": "MTOR"}
ELIGIBILITY_PATH = ROOT / "data" / "phase1b" / "PHASE1B_ELIGIBILITY.tsv"
GDSC_AUC_PATH = ROOT / "raw" / "GDSC2AUCMatrix.csv"
PRISM_PATH = ROOT / "raw" / "prism" / "secondary-screen-dose-response-curve-parameters.csv"
PRISM_FOCAL = ["EGFR", "MTOR", "ESR1", "TOP1"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ols_resid(y: pd.Series, covars: list[pd.Series], lineage: pd.DataFrame | None, cells: pd.Index) -> pd.Series:
    parts = [np.ones(len(cells))] + [c.reindex(cells).to_numpy(float) for c in covars]
    X = np.column_stack(parts + ([lineage.reindex(cells).to_numpy(float)] if lineage is not None else []))
    yv = y.reindex(cells).to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(yv).all():
        raise RuntimeError("non-finite design on declared cohort")
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return pd.Series(yv - X @ beta, index=cells)


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    return rc.gdsc2.spearman_pair(a, b)


def median(x) -> float:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return float(np.median(x)) if len(x) else float("nan")


def target_stats(rho: dict[tuple[str, str], tuple[float, int]], tmap: dict[str, str], target: str) -> dict:
    within, cross = [], []
    for (a, b), (r, n) in rho.items():
        if not np.isfinite(r) or n < MIN_PAIR_N:
            continue
        ina, inb = tmap[a] == target, tmap[b] == target
        if ina and inb:
            within.append(r)
        elif ina ^ inb:
            cross.append(r)
    w, c = median(within), median(cross)
    return {"W_t": w, "C_t": c, "delta_t": w - c, "n_within": len(within), "n_cross": len(cross)}


def group_of(target: str) -> str:
    return MERGED_GROUPS.get(target, target)


def load():
    elig = pd.read_csv(ELIGIBILITY_PATH, sep="\t", dtype={"compound_id": str})
    inc = elig.loc[elig["status"].eq("included")]
    if len(inc) != 43:
        raise RuntimeError("frozen universe changed")
    chronos = rc.gdsc2.load_chronos()
    _, lineage = rc._models_lineage()
    auc = pd.read_csv(GDSC_AUC_PATH, index_col=0, low_memory=False)
    auc.index = auc.index.astype(str)
    cells = auc.index.intersection(chronos.index).intersection(lineage.index)
    lineage = lineage.reindex(cells)
    chronos = chronos.reindex(cells)
    drugs = {}
    for row in inc.itertuples(index=False):
        d, t = str(row.compound_id), str(row.primary_target)
        R = rc.gdsc2.response_vector(auc.loc[cells, d])
        D = -chronos[t].astype(float)
        cohort = cells[(R.notna() & D.notna()).to_numpy()]
        if len(cohort) < MIN_N:
            raise RuntimeError(f"{d} below MIN_N")
        drugs[d] = {"name": str(row.compound_name), "target": t, "R": R, "D": D, "cohort": cohort}
    return drugs, chronos, lineage


def stage_residuals(drugs, lineage) -> dict[str, dict[str, pd.Series]]:
    out = {s: {} for s in ["RAW", "D", "L", "DL"]}
    for d, m in drugs.items():
        c = m["cohort"]
        out["RAW"][d] = m["R"].reindex(c)
        out["D"][d] = ols_resid(m["R"], [m["D"]], None, c)
        out["L"][d] = ols_resid(m["R"], [], lineage, c)
        out["DL"][d] = ols_resid(m["R"], [m["D"]], lineage, c)
    return out


def all_pairs(drugs) -> list[tuple[str, str]]:
    ids = list(drugs)
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]


def g_score(z: pd.DataFrame, exclude_groups: set[str], gmap: dict[str, str]) -> pd.Series:
    bg = [d for d in z.columns if gmap[d] not in exclude_groups]
    sub = z[bg]
    ok = sub.notna().sum(axis=1) >= np.ceil(G_COVERAGE * len(bg))
    return sub.median(axis=1, skipna=True).where(ok)


def g_stage_rhos(drugs, lineage, dl: dict[str, pd.Series], pairs):
    tmap = {d: m["target"] for d, m in drugs.items()}
    gmap = {d: group_of(t) for d, t in tmap.items()}
    z = pd.DataFrame({d: e / e.std(ddof=1) for d, e in dl.items()})
    g_cache: dict[frozenset, pd.Series] = {}
    res_cache: dict[tuple[str, frozenset, str], pd.Series] = {}
    g_n = {}

    def resid(d, key, with_d):
        k = (d, key, "DLG" if with_d else "LG")
        if k not in res_cache:
            G = g_cache[key]
            m = drugs[d]
            c = m["cohort"][G.reindex(m["cohort"]).notna().to_numpy()]
            if len(c) < MIN_N:
                res_cache[k] = pd.Series(dtype=float)
            else:
                cov = ([m["D"]] if with_d else []) + [G]
                res_cache[k] = ols_resid(m["R"], cov, lineage, c)
        return res_cache[k]

    rho = {"LG": {}, "DLG": {}}
    for a, b in pairs:
        key = frozenset({gmap[a], gmap[b]})
        if key not in g_cache:
            g_cache[key] = g_score(z, set(key), gmap)
            g_n[",".join(sorted(key))] = int(g_cache[key].notna().sum())
        for stage, with_d in (("LG", False), ("DLG", True)):
            ra, rb = resid(a, key, with_d), resid(b, key, with_d)
            rho[stage][(a, b)] = spearman(ra, rb) if len(ra) and len(rb) else (np.nan, 0)
    return rho, g_n


def partial_r2(R, cov, lineage, cells) -> float:
    sse_full = float(np.sum(ols_resid(R, [cov], lineage, cells) ** 2))
    sse_l = float(np.sum(ols_resid(R, [], lineage, cells) ** 2))
    return 1.0 - sse_full / sse_l


def load_prism(gdsc_chronos: pd.DataFrame, gdsc_lineage: pd.DataFrame):
    """Mirror run_n3.load_prism_coverage + prism_analysis, including the GDSC2-universe cell restriction."""
    cov = pd.read_csv(N3 / "n3_prism_compound_coverage.tsv", sep="\t", dtype={"compound_id": str})
    legal = cov.loc[cov["status"].eq("LEGAL")]
    curves = pd.read_csv(PRISM_PATH, usecols=["broad_id", "depmap_id", "auc"], low_memory=False)
    curves["core"] = curves["broad_id"].astype(str).map(brd_core)
    curves["ach"] = curves["depmap_id"].astype(str)
    curves["auc"] = pd.to_numeric(curves["auc"], errors="coerce")
    curves = curves[curves["ach"].str.startswith("ACH-") & curves["auc"].notna()]
    responses = {}
    for row in legal.itertuples(index=False):
        brds = list(dict.fromkeys(brd_core(p) for p in str(row.brd_ids).split("|") if brd_core(p)))
        auc = curves[curves["core"].isin(brds)].groupby("ach", sort=False)["auc"].mean()
        idx = auc.index.intersection(gdsc_chronos.index).intersection(gdsc_lineage.index)
        responses[str(row.compound_id)] = (rc.gdsc2.response_vector(auc.reindex(idx)), str(row.target),
                                           str(row.compound_name))
    cells = pd.Index([], dtype=object)
    for R, _, _ in responses.values():
        cells = cells.union(R.index, sort=False)
    chronos = gdsc_chronos.reindex(cells)
    lineage = gdsc_lineage.reindex(cells)
    drugs = {}
    for d, (R, t, name) in responses.items():
        R = R.reindex(cells)
        D = -chronos[t].astype(float)
        cohort = cells[(R.notna() & D.notna()).to_numpy()]
        if len(cohort) < MIN_N:
            raise RuntimeError(f"PRISM {d} below MIN_N")
        drugs[d] = {"name": name, "target": t, "R": R, "D": D, "cohort": cohort}
    if len(drugs) != 30:
        raise RuntimeError(f"PRISM universe changed: {len(drugs)}")
    return drugs, chronos, lineage


def main() -> None:
    resource = sys.argv[1] if len(sys.argv) > 1 else "GDSC2"
    gdsc_drugs, chronos, lineage = load()
    gdsc_labels = {m["target"] for m in gdsc_drugs.values()}
    if resource == "GDSC2":
        analyze(gdsc_drugs, chronos, lineage, FOCAL, N3 / "n3_target_results.tsv", RUN, gdsc_labels)
    elif resource == "PRISM":
        drugs, p_chronos, p_lineage = load_prism(chronos, lineage)
        out = RUN / "prism"
        out.mkdir(exist_ok=True)
        analyze(drugs, p_chronos, p_lineage, PRISM_FOCAL, N3 / "n3_prism_targets.tsv", out, gdsc_labels)
    else:
        raise SystemExit("usage: run_decomposition.py [GDSC2|PRISM]")


def analyze(drugs, chronos, lineage, FOCAL, frozen_path: Path, OUT: Path, excluded_genes: set[str]) -> None:
    tmap = {d: m["target"] for d, m in drugs.items()}
    pairs = all_pairs(drugs)
    stages = stage_residuals(drugs, lineage)

    rho = {s: {(a, b): spearman(res[a], res[b]) for a, b in pairs} for s, res in stages.items()}

    frozen = pd.read_csv(frozen_path, sep="\t").set_index("target")
    gate = {}
    for t in FOCAL:
        s = target_stats(rho["DL"], tmap, t)
        gate[t] = max(abs(s["W_t"] - frozen.loc[t, "W_t"]), abs(s["C_t"] - frozen.loc[t, "C_t"]))
    # Near-tied residuals are tied in one BLAS/NumPy build and split in another; see SPEC deviation 1.
    if max(gate.values()) > 1e-4:
        raise RuntimeError(f"DL stage does not reproduce frozen N3: {gate}")
    print(f"Reproduction gate PASS (max abs diff {max(gate.values()):.2e})", flush=True)

    g_rho, g_n = g_stage_rhos(drugs, lineage, stages["DL"], pairs)
    rho.update(g_rho)

    stage_order = ["RAW", "D", "L", "DL", "LG", "DLG"]
    rows = []
    for s in stage_order:
        for t in FOCAL:
            rows.append({"stage": s, "target": t, **target_stats(rho[s], tmap, t)})
    stage_tab = pd.DataFrame(rows)
    stage_tab.to_csv(OUT / "A_stage_target_results.tsv", sep="\t", index=False)
    stage_summary = stage_tab.groupby("stage", sort=False).agg(
        median_W=("W_t", "median"), median_C=("C_t", "median"), median_delta=("delta_t", "median")
    ).reindex(stage_order)
    stage_summary.to_csv(OUT / "A_stage_summary.tsv", sep="\t")

    piv = stage_tab.pivot(index="target", columns="stage", values="W_t").reindex(FOCAL)
    attn = pd.DataFrame({
        "W_L": piv["L"], "W_DL": piv["DL"], "A_t": piv["L"] - piv["DL"],
        "f_t": (piv["L"] - piv["DL"]) / piv["L"],
        "W_LG": piv["LG"], "W_DLG": piv["DLG"], "A_t_G": piv["LG"] - piv["DLG"],
    })

    drug_rows = []
    for d, m in drugs.items():
        drug_rows.append({"compound_id": d, "name": m["name"], "target": m["target"], "n": len(m["cohort"]),
                          "partial_r2_target_given_lineage": partial_r2(m["R"], m["D"], lineage, m["cohort"])})
    drug_tab = pd.DataFrame(drug_rows)

    # Part B
    null_rows, null_gene_rows, drug_null_rows = [], [], []
    for t in FOCAL:
        members = [d for d in drugs if tmap[d] == t]
        union = drugs[members[0]]["cohort"]
        for d in members[1:]:
            union = union.union(drugs[d]["cohort"])
        Dall = -chronos.reindex(union)
        full = Dall.columns[Dall.notna().all(axis=0).to_numpy()]
        cand = [g for g in full if g not in excluded_genes]
        feats = pd.DataFrame({"mean": Dall[cand].mean(), "sd": Dall[cand].std(ddof=1)})
        mu, sd = feats.mean(), feats.std(ddof=1)
        tf = pd.Series({"mean": Dall[t].mean(), "sd": Dall[t].std(ddof=1)})
        dist = np.sqrt((((feats - tf) / sd) ** 2).sum(axis=1))
        null_genes = dist.nsmallest(N_NULL).index.tolist()
        for rank, g in enumerate(null_genes, 1):
            null_gene_rows.append({"target": t, "rank": rank, "gene": g, "distance": float(dist[g]),
                                   "mean_D": float(feats.loc[g, "mean"]), "sd_D": float(feats.loc[g, "sd"]),
                                   "target_mean_D": float(tf["mean"]), "target_sd_D": float(tf["sd"])})
        for g in null_genes:
            alt = dict(stages["DL"])
            for d in members:
                c = drugs[d]["cohort"]
                alt[d] = ols_resid(drugs[d]["R"], [Dall[g]], lineage, c)
                drug_null_rows.append({"compound_id": d, "target": t, "gene": g,
                                       "partial_r2": partial_r2(drugs[d]["R"], Dall[g], lineage, c)})
            sub = {(a, b): spearman(alt[a], alt[b]) for a, b in pairs if tmap[a] == t or tmap[b] == t}
            s = target_stats(sub, tmap, t)
            null_rows.append({"target": t, "gene": g, **s, "A_null": attn.loc[t, "W_L"] - s["W_t"]})
        print(f"  null done for {t}", flush=True)

    null_tab = pd.DataFrame(null_rows)
    null_tab.to_csv(OUT / "B_null_target_results.tsv", sep="\t", index=False)
    pd.DataFrame(null_gene_rows).to_csv(OUT / "B_null_gene_sets.tsv", sep="\t", index=False)
    drug_null = pd.DataFrame(drug_null_rows)
    drug_null.to_csv(OUT / "B_null_drug_partial_r2.tsv", sep="\t", index=False)

    per_target = []
    for t in FOCAL:
        nt = null_tab[null_tab["target"].eq(t)]
        a_true = attn.loc[t, "A_t"]
        d_true = stage_tab.query("stage=='DL' and target==@t")["delta_t"].iloc[0]
        per_target.append({
            "target": t, "A_true": a_true, "A_null_median": float(nt["A_null"].median()),
            "A_null_q95": float(nt["A_null"].quantile(0.95)),
            "P_A": float((1 + (nt["A_null"] >= a_true).sum()) / (1 + len(nt))),
            "delta_true": d_true, "delta_null_median": float(nt["delta_t"].median()),
            "delta_true_percentile_in_null": float((nt["delta_t"] < d_true).mean()),
        })
    per_target = pd.DataFrame(per_target)
    attn = attn.join(per_target.set_index("target"))
    attn.to_csv(OUT / "A_B_target_attenuation.tsv", sep="\t")

    rng = np.random.default_rng(SEED)
    A_by_t = {t: null_tab.loc[null_tab["target"].eq(t), "A_null"].to_numpy() for t in FOCAL}
    draws = np.median(np.column_stack([A_by_t[t][rng.integers(0, N_NULL, N_AGG)] for t in FOCAL]), axis=1)
    agg_true = float(np.median(attn["A_t"]))
    agg = {"median_A_true": agg_true, "null_median": float(np.median(draws)),
           "null_q95": float(np.quantile(draws, 0.95)),
           "P": float((1 + np.sum(draws >= agg_true)) / (1 + N_AGG))}

    dn = drug_null.groupby("compound_id")["partial_r2"]
    focal_drugs = drug_tab[drug_tab["target"].isin(FOCAL)].set_index("compound_id")
    focal_drugs["null_partial_r2_median"] = dn.median()
    focal_drugs["P_partial_r2"] = [
        (1 + (drug_null.loc[drug_null["compound_id"].eq(d), "partial_r2"] >= focal_drugs.loc[d, "partial_r2_target_given_lineage"]).sum()) / (1 + N_NULL)
        for d in focal_drugs.index]
    focal_drugs.to_csv(OUT / "A_B_focal_drug_partial_r2.tsv", sep="\t")
    drug_tab.to_csv(OUT / "A_all_drug_partial_r2.tsv", sep="\t", index=False)

    summary = {
        "spec": "SPEC.md", "spec_sha256": sha256(RUN / "SPEC.md"), "script_sha256": sha256(Path(__file__)),
        "reproduction_gate_max_abs_diff": max(gate.values()),
        "stage_summary": stage_summary.reset_index().to_dict(orient="records"),
        "target_attenuation": attn.reset_index().to_dict(orient="records"),
        "aggregate_attenuation": agg,
        "g_valid_cells_by_exclusion": g_n,
        "seed": SEED, "n_null_genes": N_NULL, "n_aggregate_draws": N_AGG,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(stage_summary.round(4).to_string())
    print(attn.round(4).to_string())
    print(json.dumps(agg, indent=2))
    print(focal_drugs.round(4).to_string())


if __name__ == "__main__":
    main()
