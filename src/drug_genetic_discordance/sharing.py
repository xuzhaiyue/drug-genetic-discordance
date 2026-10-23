"""Original N3 pair and target estimands, extracted without modification."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .residuals import spearman_pair
rc = SimpleNamespace(gdsc2=SimpleNamespace(spearman_pair=spearman_pair))
MIN_PAIR_N = 30

def median(values) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.median(x)) if len(x) else float("nan")

def ci95(values) -> list[float | None]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(x):
        return [None, None]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

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
