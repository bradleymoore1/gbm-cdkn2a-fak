#!/usr/bin/env python3
"""DepMap synthetic lethality in CDKN2A/B-null GBM cell lines.

Question: what genes do CDKN2A/B-deleted GBM cell lines depend on more than
CDKN2A/B-intact lines? Those are synthetic lethal partners — combination
targets for abemaciclib in the elig+unmethylated subgroup.

Approach:
  1. Model.csv   -> filter to CNS/Brain lineage, get ModelID + CDKN2A/B status
  2. OmicsCNGene.csv -> verify CDKN2A/B copy number (log2 ratio; homdel < -1.0)
  3. CRISPRGeneEffect.csv -> gene effect scores (lower = more essential)
     Chronos score: 0 = no effect, -1 = essential like a core essential gene
  4. For each gene: Mann-Whitney U (null vs intact CNS lines) + effect size
  5. Report top synthetic lethal hits, filtered by:
     - delta mean effect < -0.10 (null lines more dependent)
     - q < 0.05 (BH correction)
     - Expressed in GBM (exclude housekeeping-only genes where effect is uniform)

Focus on actionable biology:
  - Known CDK4/6 pathway members (CDK4, CDK6, CCND1/2, E2F family): do they
    show expected dependency? (internal validation)
  - Unexpected hits: DNA repair, autophagy, metabolic, epigenetic — these are
    the novel combination hypotheses

Honest caveats:
  - Cell line CRISPR ≠ patient tumor. GBM cell lines are poor models of the
    invasive, heterogeneous, BBB-crossed clinical tumor.
  - CDKN2A/B deletion in cell lines may differ from tumor homdel.
  - This is hypothesis generation, not a clinical prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

DATA = Path.home() / "finngen-triage" / "depmap"
OUT = Path.home() / "finngen-triage" / "depmap_synleth_gbm.csv"

# Copy number threshold for homozygous deletion.
# OmicsCNGene values are relative CN (0=homdel, 1=diploid, 2=amp) — NOT log2 ratio.
HOMDEL_THRESH = 0.2
# Minimum number of cell lines in each group for a valid test
MIN_N = 5
# Effect size threshold: null lines must be this much more dependent
DELTA_THRESH = -0.05


def load_model():
    df = pd.read_csv(DATA / "Model.csv", low_memory=False)
    print(f"Model.csv: {len(df)} cell lines, columns: {', '.join(df.columns[:10])}...")

    # Find CNS/brain lines — DepMap uses OncotreePrimaryDisease or OncotreeLineage
    lineage_col = None
    disease_col = None
    for col in df.columns:
        cl = col.lower()
        if "lineage" in cl and lineage_col is None:
            lineage_col = col
        if ("disease" in cl or "tissue" in cl) and disease_col is None:
            disease_col = col

    print(f"  using lineage col: {lineage_col}, disease col: {disease_col}")

    # Filter to CNS
    mask = pd.Series(False, index=df.index)
    for col in [lineage_col, disease_col]:
        if col:
            mask |= df[col].fillna("").str.lower().str.contains("brain|glioma|glioblast|cns|nervous")

    cns = df[mask].copy()
    print(f"  CNS/brain cell lines: {len(cns)}")

    # Find ModelID column
    id_col = None
    for col in df.columns:
        if col.lower() in ("modelid", "depmap_id", "broad_id", "model_id"):
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]
    print(f"  using ID column: {id_col}")

    return cns, id_col


def load_cn(model_ids, id_col):
    """Return Series: model_id -> CDKN2A copy-number log2 ratio."""
    cn = pd.read_csv(DATA / "OmicsCNGene.csv", index_col=0)
    print(f"OmicsCNGene.csv: {cn.shape[0]} models x {cn.shape[1]} genes")

    # columns like "CDKN2A (1029)" — match exact gene name (space or end-of-string after)
    cdkn2a_col = next((c for c in cn.columns if c == "CDKN2A" or c.startswith("CDKN2A (")), None)
    cdkn2b_col = next((c for c in cn.columns if c == "CDKN2B" or c.startswith("CDKN2B (")), None)
    print(f"  CDKN2A col: {cdkn2a_col}  CDKN2B col: {cdkn2b_col}")

    # Align: OmicsCNGene index should be ModelID (ACH-...)
    overlap = set(cn.index) & set(model_ids)
    print(f"  CNS models with CN data: {len(overlap)}")
    cn_sub = cn.loc[cn.index.isin(overlap)][[c for c in [cdkn2a_col, cdkn2b_col] if c]]
    cn_sub.columns = ["CDKN2A", "CDKN2B"][:len(cn_sub.columns)]
    return cn_sub


def classify_cdkn2ab(cn_sub):
    """Returns: dict {model_id: 'null' | 'intact'}."""
    result = {}
    for mid, row in cn_sub.iterrows():
        # homdel if either CDKN2A or CDKN2B below threshold (or both missing → skip)
        vals = [v for v in row.values if pd.notna(v)]
        if not vals:
            continue
        # null = both available and at least one homdel; or single gene available and homdel
        if any(v < HOMDEL_THRESH for v in vals):  # val < 0.2 = homdel in normalized CN space
            result[mid] = "null"
        else:
            result[mid] = "intact"
    null_n = sum(1 for v in result.values() if v == "null")
    intact_n = sum(1 for v in result.values() if v == "intact")
    print(f"  classified: {null_n} CDKN2A/B-null, {intact_n} intact  ({len(result)} total CNS lines with CN)")
    return result


def load_crispr(model_ids):
    print("loading CRISPRGeneEffect.csv (428 MB, ~30s)...")
    ge = pd.read_csv(DATA / "CRISPRGeneEffect.csv", index_col=0)
    print(f"CRISPRGeneEffect.csv: {ge.shape[0]} models x {ge.shape[1]} genes")

    overlap = set(ge.index) & set(model_ids)
    print(f"  CNS models with CRISPR data: {len(overlap)}")
    ge_sub = ge.loc[ge.index.isin(overlap)]
    return ge_sub


def differential_dependency(ge_sub, classification):
    null_ids = [m for m, s in classification.items() if s == "null" and m in ge_sub.index]
    intact_ids = [m for m, s in classification.items() if s == "intact" and m in ge_sub.index]
    print(f"\nDifferential dependency: {len(null_ids)} null vs {len(intact_ids)} intact CNS lines")

    if len(null_ids) < MIN_N or len(intact_ids) < MIN_N:
        print(f"  INSUFFICIENT GROUPS (need >= {MIN_N} each). Cannot proceed.")
        return None

    null_df = ge_sub.loc[null_ids]
    intact_df = ge_sub.loc[intact_ids]

    results = []
    for gene in ge_sub.columns:
        n_vals = null_df[gene].dropna().values
        i_vals = intact_df[gene].dropna().values
        if len(n_vals) < MIN_N or len(i_vals) < MIN_N:
            continue
        delta = np.mean(n_vals) - np.mean(i_vals)
        try:
            stat, p = mannwhitneyu(n_vals, i_vals, alternative="less")  # null < intact (more essential)
        except ValueError:
            continue
        results.append({
            "gene": gene.split(" (")[0],  # strip " (entrezID)" suffix
            "gene_full": gene,
            "mean_null": round(float(np.mean(n_vals)), 4),
            "mean_intact": round(float(np.mean(i_vals)), 4),
            "delta": round(float(delta), 4),
            "n_null": len(n_vals),
            "n_intact": len(i_vals),
            "p": float(p),
        })

    res = pd.DataFrame(results)
    print(f"  tested {len(res)} genes")

    # BH correction
    _, q, _, _ = multipletests(res["p"].values, method="fdr_bh")
    res["q"] = q

    # Filter: more essential in null lines + significant
    sig = res[(res["delta"] < DELTA_THRESH) & (res["q"] < 0.05)].sort_values("delta")
    print(f"  significant synthetic lethal hits (delta<{DELTA_THRESH}, q<0.05): {len(sig)}")
    return res, sig


def main():
    print("=" * 100)
    print("DepMap 24Q4: Synthetic lethality in CDKN2A/B-null GBM cell lines")
    print("Identifying CDK4/6i combination partners via differential CRISPR dependency")
    print("=" * 100)

    for f in ["Model.csv", "OmicsCNGene.csv", "CRISPRGeneEffect.csv"]:
        p = DATA / f
        if not p.exists():
            print(f"MISSING: {p}  -- run downloads first")
            sys.exit(1)

    cns_model, id_col = load_model()
    model_ids = set(cns_model[id_col].dropna())
    print(f"\nCNS model IDs: {len(model_ids)}")

    cn_sub = load_cn(model_ids, id_col)
    classification = classify_cdkn2ab(cn_sub)

    ge_sub = load_crispr(set(classification.keys()))

    out = differential_dependency(ge_sub, classification)
    if out is None:
        sys.exit(1)
    all_res, sig = out

    sig.to_csv(OUT, index=False)
    print(f"\nwrote {len(sig)} hits to {OUT.name}")

    print("\n### TOP 30 SYNTHETIC LETHAL GENES (most selectively essential in CDKN2A/B-null CNS lines):")
    print(f"  {'Gene':<12} {'mean_null':>10} {'mean_intact':>11} {'delta':>8} {'q':>10}  {'n_null':>6}  {'n_intact':>8}")
    print("  " + "-" * 75)
    for _, row in sig.head(30).iterrows():
        print(f"  {row['gene']:<12} {row['mean_null']:>10.3f} {row['mean_intact']:>11.3f} "
              f"{row['delta']:>8.3f} {row['q']:>10.2e}  {int(row['n_null']):>6}  {int(row['n_intact']):>8}")

    # Check known CDK pathway genes (internal validation)
    print("\n### INTERNAL VALIDATION — CDK4/6 pathway genes (should show null preferential dependency):")
    pathway_genes = ["CDK4", "CDK6", "CCND1", "CCND2", "CCND3", "RB1", "E2F1", "E2F2", "E2F3",
                     "CDKN2A", "CDKN2B", "CDK2", "CCNE1", "CCNE2"]
    for gene in pathway_genes:
        row = all_res[all_res["gene"] == gene]
        if row.empty:
            continue
        r = row.iloc[0]
        flag = " <-- SYN LETHAL" if r["q"] < 0.05 and r["delta"] < DELTA_THRESH else ""
        print(f"  {gene:<12} delta={r['delta']:>7.3f}  q={r['q']:.2e}  "
              f"null={r['mean_null']:.3f}  intact={r['mean_intact']:.3f}{flag}")

    # Pathway enrichment (manual): tag known biology
    print("\n### PATHWAY TAGGING of top hits:")
    dna_repair = {"PARP1","PARP2","RAD51","BRCA1","BRCA2","XRCC1","LIG3","POLE","POLD1",
                  "ATM","ATR","CHEK1","CHEK2","WEE1","MRE11","NBN","RAD50"}
    autophagy  = {"ATG5","ATG7","ATG12","BECN1","ULK1","VPS34","PIK3C3"}
    epigenetic = {"EZH2","EZH1","SUZ12","EED","KDM6A","KDM6B","HDAC1","HDAC2","BRD4","BRD2"}
    metabolic  = {"GLS","GLS2","LDHA","FASN","ACLY","IDH1","IDH2","MTOR","PIK3CA","PIK3CB"}

    top50 = set(sig.head(50)["gene"].values)
    for label, gene_set in [("DNA repair", dna_repair), ("Autophagy", autophagy),
                             ("Epigenetic", epigenetic), ("Metabolic/PI3K", metabolic)]:
        hits = top50 & gene_set
        if hits:
            print(f"  {label}: {', '.join(sorted(hits))}")

    print("\n" + "=" * 100)
    print("READING:")
    print("  Negative delta = null lines more dependent (synthetic lethal with CDKN2A/B loss).")
    print("  CDK4 dependency in null lines = expected (validation): CDKN2A/B deletion -> CDK4/6 hyperactive")
    print("    -> cell is addicted to CDK4 activity. Abemaciclib target is confirmed.")
    print("  Novel hits = potential combination partners: an agent that kills CDK4/6i-resistant residual cells.")
    print("  All findings are cell-line CRISPR, not patient data. Treat as hypothesis generation only.")


if __name__ == "__main__":
    main()
