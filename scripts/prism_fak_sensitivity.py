#!/usr/bin/env python3
"""PRISM drug sensitivity: does GSK2256098 (FAK inhibitor) selectively kill CDKN2A/B-null CNS lines?

PRISM Repurposing — LFC_COLLAPSED.csv (LONG format; one row per cell-line x compound x dose x screen).
  Columns: row_id (ACH-id::profile::plate::screen), broad_id (compound), dose, compound_plate, screen, culture, LFC
  LFC = log2 viability fold-change vs DMSO; MORE NEGATIVE = MORE KILLING.

GSK2256098 = broad_id BRD-K00003379-001-01-9 (a potent, selective FAK/PTK2 inhibitor) at 2.5 uM.

Third independent dataset to test the FAK synthetic-lethality hypothesis in CDKN2A/B-null GBM:
  1. DepMap CRISPR: FAK/integrin/focal-adhesion genes more essential in CDKN2A/B-null CNS lines
  2. TCGA RNA-seq:  ITGAV upregulated in CDKN2A/B-null GBM tumors
  3. PRISM drug screen: does FAK-i selectively kill null lines?  (this script)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

DATA  = Path.home() / "finngen-triage" / "depmap"
PRISM = DATA / "prism"
OUT   = Path.home() / "finngen-triage" / "prism_fak_sensitivity.csv"

HOMDEL_THRESH = 0.2   # OmicsCNGene relative CN < 0.2 = homozygous deletion (same as CRISPR analysis)
MIN_N = 5
GSK_BROAD = "BRD-K00003379-001-01-9"   # GSK2256098 (FAK inhibitor)


def load_cns_classification() -> dict[str, str]:
    """Return {ModelID -> 'null'|'intact'} for CNS/brain lines, from OmicsCNGene CDKN2A/B CN."""
    model = pd.read_csv(DATA / "Model.csv", low_memory=False)
    cns_mask = (model["OncotreeLineage"].fillna("").str.lower()
                .str.contains("brain|glioma|glioblast|cns|nervous"))
    cns_ids = set(model.loc[cns_mask, "ModelID"].dropna())
    print(f"CNS model IDs: {len(cns_ids)}")

    # OmicsCNGene: first column is the (unnamed) ModelID index; gene cols are 'CDKN2A (1029)' etc.
    id_col = pd.read_csv(DATA / "OmicsCNGene.csv", nrows=0).columns[0]
    a_col, b_col = "CDKN2A (1029)", "CDKN2B (1030)"
    cn = pd.read_csv(DATA / "OmicsCNGene.csv", usecols=[id_col, a_col, b_col],
                     index_col=id_col, low_memory=False)
    cn_sub = cn.loc[cn.index.isin(cns_ids)]

    classification: dict[str, str] = {}
    for mid, row in cn_sub.iterrows():
        vals = [v for v in (row[a_col], row[b_col]) if pd.notna(v)]
        if not vals:
            continue
        classification[mid] = "null" if any(v < HOMDEL_THRESH for v in vals) else "intact"

    null_n = sum(v == "null" for v in classification.values())
    print(f"  CDKN2A/B-null: {null_n}  intact: {len(classification) - null_n}  "
          f"(of {len(classification)} CNS lines with CN data)")
    return classification


def load_lfc_cns(classification: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load long LFC_COLLAPSED, restrict to CNS lines, collapse to wide [model x compound] (mean LFC)."""
    print("\nLoading PRISM LFC_COLLAPSED.csv (long format, ~150 MB)...")
    lfc = pd.read_csv(PRISM / "LFC_COLLAPSED.csv",
                      usecols=["row_id", "broad_id", "LFC"])
    print(f"  raw rows: {len(lfc):,}")

    # Drop QC failures, derive ModelID from row_id prefix
    lfc = lfc[~lfc["broad_id"].str.contains("QC Failure", na=False)].copy()
    lfc["model_id"] = lfc["row_id"].str.split("::").str[0]

    # Restrict to CNS lines we classified
    lfc = lfc[lfc["model_id"].isin(classification.keys())]
    print(f"  rows on classified CNS lines: {len(lfc):,}  "
          f"(models: {lfc['model_id'].nunique()}, compounds: {lfc['broad_id'].nunique()})")

    # Collapse replicates/doses: mean LFC per (model, compound) -> wide matrix
    wide = (lfc.groupby(["model_id", "broad_id"])["LFC"].mean()
               .unstack("broad_id"))
    print(f"  wide CNS matrix: {wide.shape[0]} lines x {wide.shape[1]} compounds")

    # broad_id -> drug name map
    tm = pd.read_csv(PRISM / "Treatment_Meta.csv", usecols=["broad_id", "name"])
    name_map = (tm.dropna(subset=["broad_id"])
                  .drop_duplicates("broad_id")
                  .set_index("broad_id")["name"].to_dict())
    return wide, name_map


def test_drug(wide, classification, broad_id, alternative="two-sided"):
    null_ids   = [m for m, s in classification.items() if s == "null"   and m in wide.index]
    intact_ids = [m for m, s in classification.items() if s == "intact" and m in wide.index]
    n_vals = wide.loc[null_ids,   broad_id].dropna().values
    i_vals = wide.loc[intact_ids, broad_id].dropna().values
    if len(n_vals) < MIN_N or len(i_vals) < MIN_N:
        return None
    delta = float(np.mean(n_vals) - np.mean(i_vals))
    try:
        _, p = mannwhitneyu(n_vals, i_vals, alternative=alternative)
    except ValueError:
        return None
    return {"mean_null": float(np.mean(n_vals)), "mean_intact": float(np.mean(i_vals)),
            "delta": delta, "p": float(p), "n_null": len(n_vals), "n_intact": len(i_vals)}


def differential_sensitivity(wide, classification, name_map):
    """Per drug Mann-Whitney (null vs intact). delta<0 => null lines more killed (synthetic lethal)."""
    results = []
    for broad_id in wide.columns:
        r = test_drug(wide, classification, broad_id, alternative="less")
        if r is None:
            continue
        r["broad_id"] = broad_id
        r["drug"] = name_map.get(broad_id, broad_id)
        results.append(r)
    res = pd.DataFrame(results)
    print(f"  tested {len(res)} compounds (>= {MIN_N} lines per arm)")
    res["q"] = multipletests(res["p"].values, method="fdr_bh")[1]
    return res.sort_values("delta")


def main():
    print("=" * 100)
    print("PRISM drug sensitivity: FAK inhibitor (GSK2256098) in CDKN2A/B-null CNS lines")
    print("=" * 100)

    classification = load_cns_classification()
    wide, name_map = load_lfc_cns(classification)

    # --- Targeted test: GSK2256098 (FAK-i) ---
    print("\n" + "=" * 100)
    print("### GSK2256098 (FAK inhibitor) — targeted test")
    if GSK_BROAD in wide.columns:
        r = test_drug(wide, classification, GSK_BROAD, alternative="two-sided")
        if r:
            print(f"  CDKN2A/B-null   (n={r['n_null']}):  mean LFC = {r['mean_null']:+.3f}")
            print(f"  CDKN2A/B-intact (n={r['n_intact']}):  mean LFC = {r['mean_intact']:+.3f}")
            print(f"  delta (null - intact) = {r['delta']:+.3f}   Mann-Whitney p = {r['p']:.4f}")
            if r["delta"] < -0.05 and r["p"] < 0.05:
                print("  *** SIGNIFICANT: FAK-i selectively kills CDKN2A/B-null CNS lines ***")
            elif r["delta"] < 0:
                print("  Trend toward selective killing of null lines (delta<0), not significant")
            else:
                print("  No selective sensitivity (delta>=0: null lines NOT preferentially killed)")
        else:
            print(f"  Too few lines per arm for a test (MIN_N={MIN_N}).")
    else:
        print(f"  {GSK_BROAD} absent from CNS wide matrix.")

    # --- Unbiased screen across all compounds ---
    print("\nRunning unbiased differential-sensitivity screen (all compounds)...")
    res = differential_sensitivity(wide, classification, name_map)

    sig = res[(res["delta"] < -0.05) & (res["q"] < 0.20)].head(30)
    print(f"\n### TOP compounds selectively killing CDKN2A/B-null CNS lines (delta<-0.05, q<0.20):")
    print(f"  {'drug':<32}{'null':>8}{'intact':>8}{'delta':>8}{'q':>9}")
    print("  " + "-" * 63)
    for _, r in sig.iterrows():
        flag = " ***" if r["q"] < 0.05 else ""
        print(f"  {str(r['drug'])[:32]:<32}{r['mean_null']:>8.3f}{r['mean_intact']:>8.3f}"
              f"{r['delta']:>8.3f}{r['q']:>9.3f}{flag}")

    # --- Pathway-relevant drugs of interest ---
    print("\n### Drugs of interest (FAK, CDK4/6, PI3K/mTOR, Hippo):")
    terms = ["GSK2256098", "defactinib", "PF-562", "FAK", "VS-6063",
             "palbociclib", "ribociclib", "abemaciclib", "CDK",
             "alpelisib", "buparlisib", "everolimus", "rapamycin", "PI3K", "mTOR"]
    seen = set()
    for term in terms:
        hits = res[res["drug"].astype(str).str.contains(term, case=False, na=False)]
        for _, r in hits.head(2).iterrows():
            if r["broad_id"] in seen:
                continue
            seen.add(r["broad_id"])
            flag = " q<0.05" if r["q"] < 0.05 else ""
            print(f"  {str(r['drug'])[:34]:<34} delta={r['delta']:+.3f}  p={r['p']:.3f}  q={r['q']:.3f}{flag}")

    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.name}  ({len(res)} compounds)")

    print("\n" + "=" * 100)
    print("READING: delta < 0 => CDKN2A/B-null lines MORE killed by the drug (candidate synthetic lethal).")
    print("  If GSK2256098 (FAK-i) delta<0 & significant, the PRISM drug screen corroborates")
    print("  CRISPR (FAK/integrin essential in null) + RNA (ITGAV up in null tumors) = 3 datasets.")


if __name__ == "__main__":
    main()
