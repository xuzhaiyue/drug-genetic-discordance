#!/usr/bin/env python3
"""Build the frozen Phase 1b ChEMBL 36 eligibility table.

Does NOT compute concordance, residual E, or Atlas classes.
Requires a SHA-256-verified chembl_36 SQLite dump.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ROOT = Path("/path/to/depmap")
RAW = ROOT / "pharmacology" / "raw"
CHEMBL_DIR = RAW / "chembl36"
OUT = ROOT / "pharmacology" / "data" / "phase1b"
DISC = ROOT / "pharmacology" / "discordance"
CHUNK = 8 * 1024 * 1024
MIN_N = 200
TANIMOTO_DISTINCT = 0.70
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
TAX_HUMAN = 9606
STEREO_PREFIX = re.compile(r"^\((?:\+|-|R|S|RS)\)-?", re.I)
EXPECTED_SHA = {
    "chembl_36_sqlite.tar.gz": "b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9",
    "chembl_36_chemreps.txt.gz": "f9c2fa730ac5000da9b0c0d4033c99bfd8fe2b99950f19df1d842c6e73dd7cc4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_dump() -> dict[str, str]:
    official = {}
    for line in (CHEMBL_DIR / "checksums.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[a-fA-F0-9]{64}", parts[0]):
            official[Path(parts[-1]).name] = parts[0].lower()
    got = {}
    for name, expected in EXPECTED_SHA.items():
        p = CHEMBL_DIR / name
        if not p.exists():
            raise FileNotFoundError(p)
        digest = sha256(p)
        listed = official.get(name, expected)
        if digest != expected or digest != listed:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} vs {expected}/{listed}")
        got[name] = digest
    return got


def extract_sqlite() -> Path:
    tar_path = CHEMBL_DIR / "chembl_36_sqlite.tar.gz"
    db_candidates = list(CHEMBL_DIR.glob("**/*.db")) + list(CHEMBL_DIR.glob("**/*.sqlite"))
    if db_candidates:
        return max(db_candidates, key=lambda p: p.stat().st_size)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith((".db", ".sqlite", ".sqlite3"))]
        if not members:
            names = tar.getnames()[:30]
            raise FileNotFoundError(f"No sqlite db in archive. Example members: {names}")
        member = max(members, key=lambda m: m.size)
        tar.extract(member, path=CHEMBL_DIR)
        return CHEMBL_DIR / member.name


def norm_key(text: str, strip_stereo: bool = False) -> str:
    s = str(text).strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    if strip_stereo:
        s = STEREO_PREFIX.sub("", s).strip()
    return re.sub(r"[-\s]", "", s)


def name_keys(text: str) -> set[str]:
    out = set()
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return out
    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return out
    out.add(norm_key(raw, False))
    out.add(norm_key(raw, True))
    return {k for k in out if k}


def load_gdsc2_inventory() -> pd.DataFrame:
    auc = pd.read_csv(RAW / "GDSC2AUCMatrix.csv", index_col=0)
    auc.index = auc.index.astype(str)
    crispr = pd.read_csv(ROOT / "26Q1" / "CRISPRGeneEffect.csv", index_col=0)
    crispr.index = crispr.index.astype(str)
    genes = {c.split(" ")[0] for c in crispr.columns.astype(str)}
    models = pd.read_csv(ROOT / "26Q1" / "Model.csv", usecols=["ModelID", "OncotreeLineage"])
    models["ModelID"] = models["ModelID"].astype(str)
    lin = models.dropna(subset=["OncotreeLineage"]).set_index("ModelID")
    common = auc.index.intersection(crispr.index).intersection(lin.index)
    n = auc.loc[common].notna().sum(axis=0)
    portal = pd.read_csv(RAW / "PortalCompounds.csv")
    portal["CompoundID"] = portal["CompoundID"].astype(str)
    rows = []
    for dpc in auc.columns.astype(str):
        hit = portal.loc[portal["CompoundID"] == dpc]
        rec = hit.iloc[0].to_dict() if len(hit) else {}
        rows.append({
            "compound_id": dpc,
            "compound_name": rec.get("CompoundName"),
            "synonyms": rec.get("Synonyms"),
            "portal_chembl_id": rec.get("ChEMBLID") if pd.notna(rec.get("ChEMBLID")) else None,
            "portal_inchikey": rec.get("InChIKey") if pd.notna(rec.get("InChIKey")) else None,
            "portal_smiles": rec.get("SMILES") if pd.notna(rec.get("SMILES")) else None,
            "portal_pubchem_cid": rec.get("PubChemCID") if pd.notna(rec.get("PubChemCID")) else None,
            "brd_sample_ids": rec.get("SampleIDs"),
            "n_complete": int(n.get(dpc, 0)),
        })
    inv = pd.DataFrame(rows)
    return inv, genes


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def murcko_scaffold(smiles: str) -> tuple[str | None, object | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None, None
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None, None
    smi = Chem.MolToSmiles(scaf, canonical=True)
    fp = FPGEN.GetFingerprint(scaf)
    return smi, fp


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def build_name_index(con: sqlite3.Connection) -> dict[str, set[int]]:
    idx: dict[str, set[int]] = defaultdict(set)
    for molregno, pref in con.execute(
        "SELECT molregno, pref_name FROM molecule_dictionary WHERE pref_name IS NOT NULL"
    ):
        for k in name_keys(pref):
            idx[k].add(int(molregno))
    for molregno, syn in con.execute(
        "SELECT molregno, synonyms FROM molecule_synonyms WHERE synonyms IS NOT NULL"
    ):
        for k in name_keys(syn):
            idx[k].add(int(molregno))
    return idx


def parent_of(con: sqlite3.Connection, molregno: int) -> int:
    row = con.execute(
        "SELECT parent_molregno FROM molecule_hierarchy WHERE molregno = ?",
        (molregno,),
    ).fetchone()
    if row and row["parent_molregno"] is not None:
        return int(row["parent_molregno"])
    return molregno


def molecule_row(con: sqlite3.Connection, molregno: int) -> dict | None:
    row = con.execute(
        """
        SELECT d.molregno, d.chembl_id, d.pref_name, d.molecule_type, d.structure_type,
               s.canonical_smiles, s.standard_inchi_key
        FROM molecule_dictionary d
        LEFT JOIN compound_structures s ON s.molregno = d.molregno
        WHERE d.molregno = ?
        """,
        (molregno,),
    ).fetchone()
    return dict(row) if row else None


def resolve_identity(
    rec: dict,
    con: sqlite3.Connection,
    name_idx: dict[str, set[int]],
) -> dict:
    evidence = []
    candidates: set[int] | None = None

    def unique(cands: set[int], method: str) -> dict | None:
        if len(cands) == 1:
            molregno = next(iter(cands))
            parent = parent_of(con, molregno)
            hit = molecule_row(con, molregno)
            par = molecule_row(con, parent) if parent != molregno else hit
            smiles_src = (par or {}).get("canonical_smiles") or (hit or {}).get("canonical_smiles")
            inchikey_src = (par or {}).get("standard_inchi_key") or (hit or {}).get("standard_inchi_key")
            return {
                "identity_status": "resolved",
                "identity_method": method,
                "molregno": molregno,
                "parent_molregno": parent,
                "chembl_id": (hit or {}).get("chembl_id"),
                "chembl_pref_name": (hit or {}).get("pref_name"),
                "canonical_smiles": smiles_src,
                "inchikey": inchikey_src,
                "identity_n_candidates": 1,
                "identity_evidence": ";".join(evidence + [method]),
            }
        if len(cands) > 1:
            parents = {parent_of(con, m) for m in cands}
            if len(parents) == 1:
                return unique(parents, method + "+parent_collapse")
        return None

    cid = rec.get("portal_chembl_id")
    if cid and str(cid).upper().startswith("CHEMBL"):
        row = con.execute(
            "SELECT molregno FROM molecule_dictionary WHERE chembl_id = ?",
            (str(cid).upper(),),
        ).fetchone()
        evidence.append(f"portal_chembl:{cid}")
        if row:
            got = unique({int(row["molregno"])}, "portal_chembl_id")
            if got:
                return got
        evidence.append("portal_chembl_id_not_in_chembl36")

    ik = rec.get("portal_inchikey")
    if ik and str(ik).strip():
        row = con.execute(
            "SELECT molregno FROM compound_structures WHERE standard_inchi_key = ?",
            (str(ik).strip(),),
        ).fetchone()
        evidence.append(f"portal_inchikey:{ik}")
        if row:
            got = unique({int(row["molregno"])}, "portal_inchikey")
            if got:
                return got

    tokens = []
    tokens.extend(name_keys(rec.get("compound_name")))
    syn = rec.get("synonyms")
    if syn and pd.notna(syn):
        for part in str(syn).split(";"):
            tokens.extend(name_keys(part))
    tokens = [t for t in dict.fromkeys(tokens) if t]
    name_hits: set[int] = set()
    for t in tokens:
        name_hits |= name_idx.get(t, set())
    evidence.append(f"name_keys={len(tokens)};hits={len(name_hits)}")
    if name_hits:
        got = unique(name_hits, "pref_name_or_synonym")
        if got:
            return got
        candidates = name_hits

    smi = rec.get("portal_smiles")
    if smi and pd.notna(smi):
        computed = smiles_to_inchikey(str(smi))
        evidence.append(f"portal_smiles_inchikey:{computed}")
        if computed:
            row = con.execute(
                "SELECT molregno FROM compound_structures WHERE standard_inchi_key = ?",
                (computed,),
            ).fetchone()
            if row:
                got = unique({int(row["molregno"])}, "portal_smiles_inchikey")
                if got:
                    return got

    n_cand = len(candidates) if candidates else 0
    return {
        "identity_status": "unresolved",
        "identity_method": "identity_ambiguous" if n_cand > 1 else "identity_none",
        "molregno": None,
        "parent_molregno": None,
        "chembl_id": None,
        "chembl_pref_name": None,
        "canonical_smiles": None,
        "inchikey": None,
        "identity_n_candidates": n_cand,
        "identity_evidence": ";".join(evidence),
        "identity_candidate_molregnos": ",".join(str(x) for x in sorted(candidates)) if candidates else "",
    }


def gene_symbols_for_tid(con: sqlite3.Connection, tid: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT csyn.component_synonym AS gene
        FROM target_components tc
        JOIN component_sequences seq ON seq.component_id = tc.component_id
        JOIN component_synonyms csyn ON csyn.component_id = seq.component_id
        WHERE tc.tid = ?
          AND seq.tax_id = ?
          AND UPPER(csyn.syn_type) IN ('GENE_SYMBOL', 'GENE SYMBOL')
        """,
        (tid, TAX_HUMAN),
    ).fetchall()
    genes = []
    for r in rows:
        g = str(r["gene"]).strip().upper()
        if g and g not in genes:
            genes.append(g)
    return genes


def family_molregnos(con: sqlite3.Connection, molregno: int, parent_molregno: int) -> list[int]:
    """Parent plus all salts/children that share that parent.

    ChEMBL often stores drug_mechanism on a salt molregno (e.g. erlotinib HCl)
    while identity resolved to the parent. Looking only at the hit/parent misses
    those rows.
    """
    parent = parent_molregno if parent_molregno is not None else molregno
    rows = con.execute(
        """
        SELECT molregno FROM molecule_hierarchy
        WHERE parent_molregno = ? OR molregno = ?
        """,
        (parent, parent),
    ).fetchall()
    out = {int(molregno), int(parent)}
    for r in rows:
        out.add(int(r["molregno"]))
    return sorted(out)


def audit_targets(con: sqlite3.Connection, molregno: int, parent_molregno: int) -> tuple[dict, list[dict]]:
    mols = family_molregnos(con, molregno, parent_molregno)
    q = f"""
        SELECT m.mec_id, m.molregno, m.mechanism_of_action, m.action_type,
               m.direct_interaction, m.molecular_mechanism, m.disease_efficacy,
               m.selectivity_comment, m.tid,
               t.chembl_id AS target_chembl_id, t.pref_name AS target_pref_name,
               t.target_type, t.tax_id, t.organism
        FROM drug_mechanism m
        LEFT JOIN target_dictionary t ON t.tid = m.tid
        WHERE m.molregno IN ({",".join("?" * len(mols))})
          AND m.molecular_mechanism = 1
    """
    mechs = [dict(r) for r in con.execute(q, mols).fetchall()]
    evidence_rows = []
    human_genes: list[str] = []
    for mech in mechs:
        genes = gene_symbols_for_tid(con, mech["tid"]) if mech.get("tid") is not None else []
        is_hsp = (
            mech.get("target_type") == "SINGLE PROTEIN"
            and mech.get("tax_id") == TAX_HUMAN
        )
        if is_hsp:
            for g in genes:
                if g not in human_genes:
                    human_genes.append(g)
        evidence_rows.append({
            **{k: mech.get(k) for k in (
                "mec_id", "molregno", "mechanism_of_action", "action_type",
                "direct_interaction", "molecular_mechanism", "disease_efficacy",
                "selectivity_comment", "tid", "target_chembl_id", "target_pref_name",
                "target_type", "tax_id", "organism",
            )},
            "human_gene_symbols": "|".join(genes),
            "used_for_primary": bool(is_hsp and genes),
        })
    if not mechs:
        return {
            "target_status": "unresolved",
            "target_reason": "no_mechanism",
            "primary_target": None,
            "n_human_single_protein_genes": 0,
            "human_single_protein_genes": "",
            "n_mechanism_rows": 0,
        }, evidence_rows
    if len(human_genes) == 0:
        return {
            "target_status": "excluded_candidate",
            "target_reason": "not_human_single_protein",
            "primary_target": None,
            "n_human_single_protein_genes": 0,
            "human_single_protein_genes": "",
            "n_mechanism_rows": len(mechs),
        }, evidence_rows
    if len(human_genes) > 1:
        return {
            "target_status": "excluded_candidate",
            "target_reason": "multi_target",
            "primary_target": None,
            "n_human_single_protein_genes": len(human_genes),
            "human_single_protein_genes": "|".join(human_genes),
            "n_mechanism_rows": len(mechs),
        }, evidence_rows
    return {
        "target_status": "resolved",
        "target_reason": "chembl36_drug_mechanism",
        "primary_target": human_genes[0],
        "n_human_single_protein_genes": 1,
        "human_single_protein_genes": human_genes[0],
        "n_mechanism_rows": len(mechs),
        "target_confidence_tier": "chembl36_drug_mechanism",
    }, evidence_rows


def brd_ids(sample_ids: str | None) -> str:
    if sample_ids is None or (isinstance(sample_ids, float) and pd.isna(sample_ids)):
        return ""
    out = []
    for part in str(sample_ids).split(";"):
        part = part.strip()
        if part.startswith("BRD:"):
            token = part.replace("BRD:", "").split("BRD-")[-1]
            token = token.split("-")[0]
            if token.startswith("K") or token.startswith("A"):
                bid = "BRD-" + token
                if bid not in out:
                    out.append(bid)
    return "|".join(out)


def assign_status(row: dict, chronos_genes: set[str]) -> dict:
    reasons = []
    if row["identity_status"] != "resolved":
        return {
            "status": "unresolved",
            "exclude_from_primary": True,
            "reason_code": row.get("identity_method") or "identity_none",
            "reason_detail": row.get("identity_evidence"),
        }
    if not row.get("canonical_smiles"):
        reasons.append("no_structure")
    tgt = row.get("target_status")
    if tgt == "unresolved":
        return {
            "status": "unresolved",
            "exclude_from_primary": True,
            "reason_code": row.get("target_reason") or "no_mechanism",
            "reason_detail": row.get("human_single_protein_genes"),
        }
    if tgt != "resolved":
        reasons.append(row.get("target_reason") or "target_failed")
    elif row.get("primary_target") not in chronos_genes:
        reasons.append("gene_absent_from_chronos")
    if not row.get("scaffold_smiles"):
        reasons.append("no_scaffold")
    if int(row.get("n_complete") or 0) < MIN_N:
        reasons.append("n_lt_200")
    if reasons:
        return {
            "status": "excluded",
            "exclude_from_primary": True,
            "reason_code": "|".join(reasons),
            "reason_detail": row.get("human_single_protein_genes"),
        }
    return {
        "status": "included",
        "exclude_from_primary": False,
        "reason_code": "included",
        "reason_detail": "",
    }


def target_groups(elig: pd.DataFrame, fps: dict[str, object]) -> pd.DataFrame:
    inc = elig[elig["status"] == "included"].copy()
    rows = []
    for target, g in inc.groupby("primary_target"):
        ids = list(g["compound_id"])
        distinct_pairs = []
        n_pairs = 0
        min_t = None
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                n_pairs += 1
                if a in fps and b in fps:
                    t = DataStructs.TanimotoSimilarity(fps[a], fps[b])
                    min_t = t if min_t is None else min(min_t, t)
                    if t < TANIMOTO_DISTINCT:
                        distinct_pairs.append(f"{a}~{b}:{t:.3f}")
        n_scaf = g["scaffold_smiles"].nunique(dropna=True)
        eligible = len(ids) >= 2 and len(distinct_pairs) >= 1
        rows.append({
            "primary_target": target,
            "n_included_drugs": len(ids),
            "n_unique_scaffolds": int(n_scaf),
            "n_pairs": n_pairs,
            "n_distinct_scaffold_pairs": len(distinct_pairs),
            "min_scaffold_tanimoto": min_t,
            "target_group_eligible": eligible,
            "compound_ids": "|".join(ids),
            "distinct_pairs": "|".join(distinct_pairs),
        })
    return pd.DataFrame(rows).sort_values(
        ["target_group_eligible", "n_included_drugs"],
        ascending=[False, False],
    ) if rows else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dumps = verify_dump()
    db = extract_sqlite()
    db_sha = sha256(db)
    inv, chronos_genes = load_gdsc2_inventory()
    con = connect(db)
    name_idx = build_name_index(con)

    identity_rows = []
    evidence_rows = []
    fps: dict[str, object] = {}
    rdkit_ver = rdBase.rdkitVersion

    for rec in inv.to_dict(orient="records"):
        ident = resolve_identity(rec, con, name_idx)
        row = {**rec, **ident}
        if ident["identity_status"] == "resolved" and ident.get("molregno") is not None:
            tgt, ev = audit_targets(con, ident["molregno"], ident["parent_molregno"])
            row.update(tgt)
            for e in ev:
                evidence_rows.append({"compound_id": rec["compound_id"], **e})
        else:
            row.update({
                "target_status": "unresolved",
                "target_reason": "identity_unresolved",
                "primary_target": None,
                "n_human_single_protein_genes": 0,
                "human_single_protein_genes": "",
                "n_mechanism_rows": 0,
            })
        scaf, fp = (None, None)
        if row.get("canonical_smiles"):
            scaf, fp = murcko_scaffold(row["canonical_smiles"])
        row["scaffold_smiles"] = scaf
        row["brd_id"] = brd_ids(rec.get("brd_sample_ids"))
        row["annotation_source"] = "ChEMBL_36"
        row["annotation_date"] = datetime.now(timezone.utc).date().isoformat()
        row["rdkit_version"] = rdkit_ver
        if fp is not None:
            fps[rec["compound_id"]] = fp
        status = assign_status(row, chronos_genes)
        row.update(status)
        identity_rows.append(row)

    elig = pd.DataFrame(identity_rows)
    groups = target_groups(elig, fps)
    evid = pd.DataFrame(evidence_rows)
    excl = elig[elig["status"] != "included"].copy()

    keep_cols = [
        "compound_id", "compound_name", "status", "reason_code", "reason_detail",
        "exclude_from_primary", "n_complete", "chembl_id", "chembl_pref_name",
        "identity_method", "identity_evidence", "primary_target",
        "target_confidence_tier", "target_reason", "human_single_protein_genes",
        "canonical_smiles", "inchikey", "scaffold_smiles", "brd_id",
        "annotation_source", "annotation_date", "rdkit_version", "molregno",
        "parent_molregno",
    ]
    for c in keep_cols:
        if c not in elig.columns:
            elig[c] = None

    elig[keep_cols].to_csv(OUT / "PHASE1B_ELIGIBILITY.tsv", sep="\t", index=False)
    excl[keep_cols].to_csv(OUT / "PHASE1B_EXCLUSION.tsv", sep="\t", index=False)
    elig.to_csv(OUT / "PHASE1B_IDENTITY.tsv", sep="\t", index=False)
    evid.to_csv(OUT / "PHASE1B_TARGET_EVIDENCE.tsv", sep="\t", index=False)
    elig.loc[elig["status"] == "included", [
        "compound_id", "compound_name", "primary_target", "canonical_smiles",
        "inchikey", "scaffold_smiles", "chembl_id",
    ]].to_csv(OUT / "PHASE1B_SCAFFOLDS.tsv", sep="\t", index=False)
    groups.to_csv(OUT / "PHASE1B_TARGET_GROUPS.tsv", sep="\t", index=False)

    table_files = [
        "PHASE1B_ELIGIBILITY.tsv",
        "PHASE1B_EXCLUSION.tsv",
        "PHASE1B_IDENTITY.tsv",
        "PHASE1B_TARGET_EVIDENCE.tsv",
        "PHASE1B_SCAFFOLDS.tsv",
        "PHASE1B_TARGET_GROUPS.tsv",
    ]
    output_sha = {}
    for name in table_files:
        p = OUT / name
        output_sha[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (OUT / "PHASE1B_OUTPUT.sha256").write_text(
        "\n".join(f"{v['sha256']}  {k}" for k, v in output_sha.items()) + "\n"
    )

    counts = elig["status"].value_counts().to_dict()
    reason_counts = elig["reason_code"].value_counts().to_dict()
    summary = {
        "module": "PHASE1B_CHEMBL36_ELIGIBILITY",
        "computed": datetime.now(timezone.utc).isoformat(),
        "no_atlas_or_residual": True,
        "source": {
            "release": "chembl_36",
            "doi": "10.6019/CHEMBL.database.36",
            "ftp": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/",
            "sqlite_db": str(db),
            "sqlite_db_sha256": db_sha,
            "archive_sha256": dumps,
            "rdkit_version": rdkit_ver,
            "morgan": {"radius": MORGAN_RADIUS, "nBits": MORGAN_BITS},
            "tanimoto_distinct": TANIMOTO_DISTINCT,
            "min_n": MIN_N,
        },
        "n_gdsc2_drugs": int(len(elig)),
        "status_counts": {k: int(v) for k, v in counts.items()},
        "reason_counts": {k: int(v) for k, v in reason_counts.items()},
        "n_included": int((elig["status"] == "included").sum()),
        "n_excluded": int((elig["status"] == "excluded").sum()),
        "n_unresolved": int((elig["status"] == "unresolved").sum()),
        "n_eligible_target_groups": int(groups["target_group_eligible"].sum()) if len(groups) else 0,
        "n_target_groups_total": int(len(groups)),
        "all_286_have_status": bool(len(elig) == 286 and elig["status"].isin(["included", "excluded", "unresolved"]).all()),
        "protocol": str(DISC / "PHASE_1B_CHEMBL36_ANNOTATION_PROTOCOL.md"),
        "output_sha256": output_sha,
    }
    (OUT / "PHASE1B_ANNOTATION.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "n_gdsc2_drugs", "status_counts", "n_included", "n_excluded",
        "n_unresolved", "n_eligible_target_groups", "all_286_have_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
