#!/usr/bin/env python3
"""PRISM coverage audit for frozen Atlas PANEL.

Counts only. Does not compute concordance, E, or residual reproducibility.
Sources audited separately; never mixed.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path("/path/to/depmap")
RAW = ROOT / "pharmacology" / "raw"
OUT = ROOT / "pharmacology" / "discordance"
PRISM_DIR = RAW / "prism"
CRISPR_PATH = ROOT / "26Q1" / "CRISPRGeneEffect.csv"
MODEL_PATH = ROOT / "26Q1" / "Model.csv"
COMPOUNDS = RAW / "PortalCompounds.csv"

MIN_N = 200
MIN_CROSS_PAIRS = 5

# Frozen Atlas PANEL — (atlas_name, target, DPC)
PANEL = [
    ("ERLOTINIB", "EGFR", "DPC-002573"),
    ("GEFITINIB", "EGFR", "DPC-002973"),
    ("AFATINIB", "EGFR", "DPC-000358"),
    ("LAPATINIB", "EGFR", "DPC-003736"),
    ("OSIMERTINIB", "EGFR", "DPC-000819"),
    ("AZD8931", "ERBB2", "DPC-000861"),
    ("BMS-754807", "IGF1R", "DPC-001165"),
    ("DABRAFENIB", "BRAF", "DPC-002044"),
    ("AZD5991", "MCL1", "DPC-000850"),
    ("MK 2206", "AKT1", "DPC-004237"),
    ("GSK-2141795", "AKT1", "DPC-003086"),
    ("IPATASERTIB", "AKT1", "DPC-003457"),
    ("FK866", "NAMPT", "DPC-000635"),
    ("SAVOLITINIB", "MET", "DPC-005781"),
    ("CRIZOTINIB", "MET", "DPC-001953"),
    ("SCH772984", "MAPK1", "DPC-003397"),
    ("VX-11E", "MAPK1", "DPC-006904"),
    ("NILOTINIB", "PDGFRB", "DPC-004577"),
]
MULTI_CHEMO = {
    "EGFR": ["ERLOTINIB", "GEFITINIB", "AFATINIB", "LAPATINIB", "OSIMERTINIB"],
    "AKT1": ["MK 2206", "GSK-2141795", "IPATASERTIB"],
    "MAPK1": ["SCH772984", "VX-11E"],
    "MET": ["SAVOLITINIB", "CRIZOTINIB"],
}

URLS = {
    "primary_compounds": (
        "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv",
        "https://ndownloader.figshare.com/files/46630981",
    ),
    "primary_matrix": (
        "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        "https://ndownloader.figshare.com/files/46630984",
    ),
    "primary_cells": (
        "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv",
        "https://ndownloader.figshare.com/files/46630978",
    ),
    "secondary_curves": (
        "secondary-screen-dose-response-curve-parameters.csv",
        "https://ndownloader.figshare.com/files/36794595",
    ),
}


def brd_core(s: str) -> str:
    s = str(s).upper().strip()
    if "BRD-" not in s:
        return ""
    token = s.split("BRD-")[-1]
    token = token.split(";")[0].split(",")[0].strip()
    if token.startswith("K") or token.startswith("A"):
        return "BRD-" + token.split("-")[0]  # BRD-K70401845
    return ""


def dpc_brds(sample_ids: str) -> list[str]:
    out = []
    for part in str(sample_ids).split(";"):
        part = part.strip()
        if part.startswith("BRD:"):
            core = brd_core(part.replace("BRD:", ""))
            if core and core not in out:
                out.append(core)
    return out


def fetch(name: str, url: str) -> Path:
    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRISM_DIR / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"download {name}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_chronos_models() -> tuple[pd.Index, pd.Index]:
    crispr = pd.read_csv(CRISPR_PATH, index_col=0)
    crispr.index = crispr.index.astype(str)
    models = pd.read_csv(MODEL_PATH, usecols=["ModelID", "OncotreeLineage", "StrippedCellLineName", "CCLEName"])
    models["ModelID"] = models["ModelID"].astype(str)
    has_lin = models.dropna(subset=["OncotreeLineage"]).set_index("ModelID")
    chronos_ids = crispr.index.intersection(has_lin.index)
    return chronos_ids, has_lin.index


def map_prism_cells_to_ach(index: pd.Index, models: pd.DataFrame) -> pd.Series:
    """Map PRISM labels to ModelID (ACH-* pass through)."""
    stripped = models.dropna(subset=["StrippedCellLineName"]).copy()
    stripped["k"] = stripped["StrippedCellLineName"].astype(str).str.upper().str.replace("-", "", regex=False)
    lut = stripped.drop_duplicates("k").set_index("k")["ModelID"]
    ach_set = set(models["ModelID"].astype(str))
    mapped = []
    for lab in index.astype(str):
        u = lab.upper()
        if u.startswith("ACH-") and u in ach_set:
            mapped.append(u)
            continue
        k = u.replace("-", "")
        mapped.append(lut.loc[k] if k in lut.index else None)
    return pd.Series(mapped, index=index)


def legal_targets(drug_ok: dict[str, bool]) -> dict[str, list[str]]:
    out = {}
    for t, drugs in MULTI_CHEMO.items():
        ok = [d for d in drugs if drug_ok.get(d)]
        if len(ok) >= 2:
            out[t] = ok
    return out


def gate_a(legal: dict[str, list[str]], drug_target: dict[str, str]) -> tuple[bool, int]:
    if len(legal) < 2:
        return False, 0
    drugs = [d for ds in legal.values() for d in ds]
    n_cross = 0
    for i, a in enumerate(drugs):
        for b in drugs[i + 1 :]:
            if drug_target[a] != drug_target[b]:
                n_cross += 1
    return n_cross >= MIN_CROSS_PAIRS, n_cross


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compounds = pd.read_csv(COMPOUNDS)
    compounds = compounds.set_index("CompoundID")
    dpc_to_brd = {dpc: dpc_brds(compounds.loc[dpc, "SampleIDs"]) for _, _, dpc in PANEL if dpc in compounds.index}
    models = pd.read_csv(MODEL_PATH, usecols=["ModelID", "OncotreeLineage", "StrippedCellLineName", "CCLEName"])
    models["ModelID"] = models["ModelID"].astype(str)
    chronos_ids, _ = load_chronos_models()
    drug_target = {name: t for name, t, _ in PANEL}

    for key, (fname, url) in URLS.items():
        fetch(fname, url)

    # --- Primary 24Q2 matrix: rows = BRD IDs, columns = ACH-* ---
    mat_path = PRISM_DIR / URLS["primary_matrix"][0]
    prim = pd.read_csv(mat_path, index_col=0)
    ach_cols = [c for c in prim.columns if str(c).startswith("ACH-")]
    prim_ach = [c for c in ach_cols if c in set(chronos_ids)]
    row_core = pd.Index(prim.index.astype(str).map(brd_core))

    primary_rows = []
    primary_ok: dict[str, bool] = {}
    for name, target, dpc in PANEL:
        brds = dpc_to_brd.get(dpc, [])
        hit = row_core.isin(brds)
        if not hit.any():
            primary_rows.append(
                {"source": "primary_24Q2", "drug": name, "target": target, "dpc": dpc,
                 "brds": brds, "column": None, "n_prism": 0, "n_intersect": 0, "legal": False}
            )
            primary_ok[name] = False
            continue
        block = prim.loc[hit, ach_cols].apply(pd.to_numeric, errors="coerce")
        any_val = block.notna().any(axis=0)
        n_prism = int(any_val.sum())
        n_int = int(block[prim_ach].notna().any(axis=0).sum())
        row_ids = [str(i) for i in prim.index[hit][:4]]
        legal = n_int >= MIN_N
        primary_ok[name] = legal
        primary_rows.append(
            {"source": "primary_24Q2", "drug": name, "target": target, "dpc": dpc,
             "brds": brds, "column": row_ids, "n_prism": n_prism, "n_intersect": n_int,
             "n_rows_matched": int(hit.sum()), "legal": legal}
        )

    # --- Secondary 20Q2 AUC ---
    sec_path = PRISM_DIR / URLS["secondary_curves"][0]
    sec = pd.read_csv(sec_path)
    sec_cols = {c.lower(): c for c in sec.columns}

    def col(*cands: str) -> str | None:
        for c in cands:
            if c in sec.columns:
                return c
            if c.lower() in sec_cols:
                return sec_cols[c.lower()]
        return None

    broad_col = col("broad_id", "broad.id", "broad_cpd_id", "pert_id")
    auc_col = col("auc", "auc_raw", "area_under_curve")
    cell_col = col("depmap_id", "depmap.id")
    name_col = col("name", "pert_iname", "compound_name")

    secondary_rows = []
    secondary_ok: dict[str, bool] = {}
    for name, target, dpc in PANEL:
        brds = dpc_to_brd.get(dpc, [])
        sub = pd.DataFrame()
        if broad_col is not None:
            cores = sec[broad_col].astype(str).map(brd_core)
            sub = sec[cores.isin(brds)]
        if sub.empty and name_col is not None:
            key = name.replace(" ", "").replace("-", "").upper()
            nm = sec[name_col].astype(str).str.replace(" ", "", regex=False).str.replace("-", "", regex=False).str.upper()
            sub = sec[nm == key]
        if sub.empty or auc_col is None or cell_col is None:
            secondary_rows.append(
                {"source": "secondary_20Q2_auc", "drug": name, "target": target, "dpc": dpc,
                 "brds": brds, "n_prism": 0, "n_intersect": 0, "legal": False,
                 "note": "unmapped or missing auc/cell column"}
            )
            secondary_ok[name] = False
            continue
        cells = pd.Index(sub[cell_col].astype(str))
        auc = pd.to_numeric(sub[auc_col], errors="coerce")
        tmp = pd.DataFrame({"ach": cells.to_numpy(), "auc": auc.to_numpy()}).dropna()
        tmp = tmp[tmp["ach"].str.startswith("ACH-")]
        tmp = tmp.drop_duplicates("ach").set_index("ach")["auc"]
        n_prism = int(tmp.notna().sum())
        n_int = int(tmp.reindex(chronos_ids).dropna().shape[0])
        legal = n_int >= MIN_N
        secondary_ok[name] = legal
        secondary_rows.append(
            {"source": "secondary_20Q2_auc", "drug": name, "target": target, "dpc": dpc,
             "brds": brds, "n_prism": n_prism, "n_intersect": n_int, "legal": legal,
             "n_raw_rows": int(len(sub))}
        )

    def summarize(ok: dict[str, bool], rows: list[dict], source: str) -> dict:
        legal = legal_targets(ok)
        a, n_cross = gate_a(legal, drug_target)
        b = all(t in legal for t in ("EGFR", "AKT1", "MAPK1", "MET"))
        if a:
            verdict = "LEGAL"
        else:
            verdict = "BLOCKED_BY_COVERAGE"
        return {
            "source": source,
            "min_n": MIN_N,
            "legal_multi_chemotype_targets": {t: ds for t, ds in legal.items()},
            "n_legal_multi_targets": len(legal),
            "n_cross_target_pairs_among_legal": n_cross,
            "gate_A_same_vs_cross_testable": a,
            "gate_B_four_poles": b,
            "verdict": verdict,
            "drugs": rows,
        }

    report = {
        "module": "PRISM_REPLICATION_COVERAGE",
        "prereg": "PRISM_REPLICATION_PREREG.md",
        "computed": False,
        "note": "Coverage counts only. Concordance and residual repro were not calculated.",
        "dpc_to_brd": dpc_to_brd,
        "primary_24Q2_extended": summarize(primary_ok, primary_rows, "primary_24Q2_extended"),
        "secondary_20Q2_auc": summarize(secondary_ok, secondary_rows, "secondary_20Q2_auc"),
        "preferred_if_both_legal": "secondary_20Q2_auc",
        "matrix_shape_primary": [int(prim.shape[0]), int(prim.shape[1])],
        "secondary_columns": list(sec.columns[:40]),
    }
    # overall: legal if either source Gate A
    any_legal = (
        report["primary_24Q2_extended"]["verdict"] == "LEGAL"
        or report["secondary_20Q2_auc"]["verdict"] == "LEGAL"
    )
    report["overall_verdict"] = "LEGAL" if any_legal else "BLOCKED_BY_COVERAGE"
    if report["secondary_20Q2_auc"]["verdict"] == "LEGAL":
        report["run_source_if_legal"] = "secondary_20Q2_auc"
    elif report["primary_24Q2_extended"]["verdict"] == "LEGAL":
        report["run_source_if_legal"] = "primary_24Q2_extended"
    else:
        report["run_source_if_legal"] = None

    (OUT / "prism_coverage.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("overall_verdict", "run_source_if_legal")}, indent=2))
    for src in ("primary_24Q2_extended", "secondary_20Q2_auc"):
        s = report[src]
        print(f"\n=== {src}  {s['verdict']}  gateA={s['gate_A_same_vs_cross_testable']} gateB={s['gate_B_four_poles']} ===")
        for r in s["drugs"]:
            print(f"  {r['drug']:16} {r['target']:6} n∩={r['n_intersect']:4} legal={r['legal']} col={r.get('column')}")
        print("  legal targets:", s["legal_multi_chemotype_targets"])


if __name__ == "__main__":
    main()
