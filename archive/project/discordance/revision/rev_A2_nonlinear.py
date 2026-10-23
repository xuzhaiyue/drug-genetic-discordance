#!/usr/bin/env python3
"""Stage 2A A2 nonlinear residualizer sensitivity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from rev_common import (REVISION, compute_E, drug_label_perm, linear_ols,
                        load_resource, pair_table, stage_metadata, summary)

N_PERM = 10000
SEED = 20260927
METHODS = ("linear", "spline", "rank_spline", "decile")


def make_residualizer(method: str, diagnostics: dict[str, dict]) :
    def residualizer(R: pd.Series, D: pd.Series | pd.DataFrame,
                     L: pd.DataFrame) -> pd.Series:
        if isinstance(D, pd.DataFrame):
            D = D.iloc[:, 0]
        df = pd.concat([R.rename("R"), D.rename("D"), L], axis=1).dropna()
        out = pd.Series(np.nan, index=R.index, dtype=float)
        if len(df) < 200:
            return out
        y = df["R"].astype(float)
        x = df["D"].astype(float)
        lineage = df[L.columns].astype(float)
        # Keep the frozen linear residual as the exact baseline implementation.
        lin_e = linear_ols(R, D, L).reindex(df.index)
        lin_x = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                           x.rename("D"), lineage], axis=1)
        lin_fit = sm.OLS(y, lin_x).fit()
        if method == "linear":
            out.loc[df.index] = lin_e.to_numpy()
            fitted = lin_fit
            fit_design = lin_x
        elif method in {"spline", "rank_spline"}:
            if method == "spline":
                basis_x = x
            else:
                ranks = x.rank(method="average")
                lo, hi = float(ranks.min()), float(ranks.max())
                basis_x = (ranks - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=x.index)
            basis = patsy.dmatrix("cr(z, df=4)", {"z": basis_x.to_numpy()},
                                  return_type="dataframe")
            basis.index = df.index
            fit_design = pd.concat([basis, lineage], axis=1)
            fitted = sm.OLS(y, fit_design).fit()
            out.loc[df.index] = fitted.resid.to_numpy()
        elif method == "decile":
            q = pd.qcut(x, 10, duplicates="drop")
            bins = pd.get_dummies(q, drop_first=True, dtype=float)
            bins.index = df.index
            fit_design = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                                    bins, lineage], axis=1)
            fitted = sm.OLS(y, fit_design).fit()
            out.loc[df.index] = fitted.resid.to_numpy()
        else:
            raise ValueError(method)
        f_p = None
        if method in {"spline", "rank_spline"}:
            try:
                _f, f_p, _df_diff = fitted.compare_f_test(lin_fit)
                f_p = float(f_p)
            except (ValueError, np.linalg.LinAlgError):
                f_p = None
        diagnostics[R.name or "drug"] = {
            "n": int(len(df)),
            "r2_linear": float(lin_fit.rsquared),
            "r2_method": float(fitted.rsquared),
            "delta_r2_vs_linear": float(fitted.rsquared - lin_fit.rsquared),
            "spline_f_test_p": f_p,
        }
        return out
    return residualizer


def main() -> None:
    output = {"analysis": "A2 nonlinear residualizer sensitivity",
              "n_perm": N_PERM, "seed": SEED, "decision": {}, "resources": {}}
    pair_blocks = []
    inputs = [REVISION / "rev_common.py", Path(__file__).resolve(),
              REVISION / "rev_common_regression.json", REVISION / "rev_A1a_drugperm.json",
              REVISION / "cross_target_pairs.tsv"]
    for name in ("GDSC2", "PRISM"):
        data = load_resource(name)
        inputs.extend(data["input_paths"])
        fits = {}
        baseline_summary = None
        for method in METHODS:
            diagnostics: dict[str, dict] = {}
            residualizer = make_residualizer(method, diagnostics)
            E = compute_E(data, residualizer=residualizer)
            pairs = pair_table(E, data["panel"])
            pairs.insert(0, "residualizer", method)
            pairs.insert(0, "resource", name)
            pair_blocks.append(pairs)
            summ = summary(pairs)
            perm = drug_label_perm(E, data["panel"], n_perm=N_PERM, seed=SEED)
            if method == "linear":
                baseline_summary = summ
            retention = (float(summ["same_median"] / baseline_summary["same_median"])
                         if baseline_summary and baseline_summary["same_median"] else None)
            fits[method] = {
                **summ,
                "retention_vs_same_subset_linear": retention,
                "p_drugperm": perm["p_drugperm"],
                "fit_diagnostics_by_drug": diagnostics,
            }
        output["resources"][name] = fits
    all_retention = [output["resources"][r][m]["retention_vs_same_subset_linear"]
                     for r in ("GDSC2", "PRISM") for m in METHODS[1:]]
    all_perm_pass = all(output["resources"][r][m]["p_drugperm"] < 0.01
                        for r in ("GDSC2", "PRISM") for m in METHODS[1:])
    if any(x < 0.50 for x in all_retention):
        branch = "CORE_CLAIM_REWRITE_STOP_STAGE3"
    elif all(x >= 0.80 for x in all_retention) and all_perm_pass:
        branch = "PASS"
    else:
        branch = "PARTIAL"
    linear_a1_flags = {r: output["resources"][r]["linear"]["p_drugperm"] >= 0.01
                       for r in ("GDSC2", "PRISM")}
    output["decision"] = {
        "a2_branch": branch,
        "all_three_residualizers_both_resources_retention_ge_0_80": bool(all(x >= 0.80 for x in all_retention)),
        "all_three_residualizers_both_resources_p_drugperm_lt_0_01": bool(all_perm_pass),
        "any_retention_below_0_50": bool(any(x < 0.50 for x in all_retention)),
        "linear_p_drugperm_ge_0_01_a1_issue": linear_a1_flags,
    }
    output.update(stage_metadata(list(dict.fromkeys(inputs)), seed=SEED, n_perm=N_PERM))
    (REVISION / "rev_A2_nonlinear.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.concat(pair_blocks, ignore_index=True).to_csv(REVISION / "rev_A2_pairs.tsv", sep="\t", index=False)
    print(json.dumps({"decision": output["decision"],
                      "summary": {r: {m: {"same_median": output["resources"][r][m]["same_median"],
                                            "cross_median": output["resources"][r][m]["cross_median"],
                                            "retention": output["resources"][r][m]["retention_vs_same_subset_linear"],
                                            "p_drugperm": output["resources"][r][m]["p_drugperm"]}
                                       for m in METHODS} for r in ("GDSC2", "PRISM")}}, indent=2))


if __name__ == "__main__":
    main()
