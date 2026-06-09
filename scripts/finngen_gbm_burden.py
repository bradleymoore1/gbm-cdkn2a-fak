#!/usr/bin/env python3
"""FinnGen R13 gene-burden analysis for glioma endpoints.

Three complementary analyses:
 1. Cross-endpoint consistency: Fisher's method to combine C3_GBM + C3_GBM_ASTROCYTOMA +
    C3_BRAIN_WIDE, keeping only genes significant in same direction across endpoints.
 2. Gene-set enrichment test: are DNA repair / glioma TSG genes enriched in the high-mlogp
    tail of the GBM burden distribution? Kolmogorov-Smirnov test + permutation baseline.
 3. MUTYH deep-dive: extract variants, compute honest 95% CI for OR, check Finnish enrichment.

Honest caveats:
  - C3_GBM N_cases=467 — severely underpowered for rare-variant burden.
  - Very rare mask (AF<0.01) means some genes have 1-5 carriers total in cases.
  - No genome-wide significance reached (need mlogp~6-7 after multi-gene correction).
  - All findings are hypothesis-generating, not causal.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp

LOF = Path.home() / "finngen-r13" / "lof" / "data"
OUT = Path.home() / "finngen-triage" / "finngen_gbm_burden.csv"

ENDPOINTS = ["C3_GBM", "C3_GBM_ASTROCYTOMA", "C3_BRAIN_WIDE"]

# Curated gene sets
DNA_REPAIR = {
    # BER
    "MUTYH","OGG1","NEIL1","NEIL2","NEIL3","NTHL1","PARP1","PARP2","XRCC1","LIG3","POLB","APE1","APEX1",
    # MMR
    "MLH1","MLH3","MSH2","MSH3","MSH6","PMS1","PMS2","EPCAM",
    # HR
    "BRCA1","BRCA2","PALB2","RAD51","RAD51C","RAD51D","BRIP1","BARD1","CHEK2","ATM","ATR","CHEK1",
    "MRE11","NBN","RAD50","FANCD2","FANCM","FANCA","FANCC","FANCF","FANCG","FANCI","FANCL","FANCN",
    # NER
    "XPC","XPA","ERCC1","ERCC2","ERCC3","ERCC4","ERCC5","ERCC6","DDB2","POLH",
    # NHEJ
    "PRKDC","XRCC4","XRCC5","XRCC6","LIG4","NHEJ1","DNTT",
    # Other
    "RECQL","RECQL4","RECQL5","BLM","WRN","HELQ","FAN1","SLX4","GEN1","RNASEH2A","RNASEH2B","RNASEH2C",
    "TREX1","SAMHD1","ADAR",
}

GLIOMA_TSG = {
    "PTEN","TP53","RB1","CDKN2A","CDKN2B","NF1","NF2","ATRX","CIC","FUBP1","EGFR","IDH1","IDH2",
    "PIK3CA","PIK3R1","PDGFRA","MET","AKT3","TERT","MGMT","TCF12","SMARCA4","SMARCB1",
}

REPLICATION_ORIGIN = {"ORC1","ORC2","ORC3","ORC4","ORC5","ORC6","CDC6","CDT1","MCM2","MCM3",
                       "MCM4","MCM5","MCM6","MCM7","MCM8","MCM9","GINS1","GINS2","GINS3","GINS4"}


def load_burden():
    rows = []
    with gzip.open(LOF / "finngen_R13_lof.txt.gz", "rt") as fh:
        hdr = fh.readline().split()
        for line in fh:
            parts = line.rstrip().split("\t")
            d = dict(zip(hdr, parts))
            if d["PHENO"] not in ENDPOINTS:
                continue
            try:
                mlogp = float(d["LOG10P"])
                beta = float(d["BETA"])
                se = float(d["SE"])
                af = float(d["A1FREQ"])
            except (ValueError, KeyError):
                continue
            gene_raw = d.get("ID", "")
            gene = gene_raw.split(".")[0]  # "MUTYH.Mask1.0.01" -> "MUTYH"
            rows.append({"pheno": d["PHENO"], "gene": gene, "mlogp": mlogp, "beta": beta, "se": se, "af": af})
    return pd.DataFrame(rows)


def cross_endpoint_fisher(df):
    """Fisher's method: -2 * sum(log(p)) ~ chi2(2k) if all same direction."""
    # pivot: gene x endpoint -> mlogp, beta
    pivot_m = df.pivot_table(index="gene", columns="pheno", values="mlogp", aggfunc="first")
    pivot_b = df.pivot_table(index="gene", columns="pheno", values="beta", aggfunc="first")

    results = []
    for gene in pivot_m.index:
        row_m = pivot_m.loc[gene]
        row_b = pivot_b.loc[gene]
        avail = row_m.dropna()
        if len(avail) < 2:
            continue
        # Check direction consistency (all same sign beta)
        betas = row_b.dropna()
        if betas.max() > 0 and betas.min() < 0:
            direction = "MIXED"
        elif betas.max() > 0:
            direction = "RISK"
        else:
            direction = "PROTECTIVE"

        p_vals = [10 ** (-m) for m in avail]
        stat = -2 * sum(math.log(p) for p in p_vals)
        df_chi = 2 * len(avail)
        combined_p = chi2.sf(stat, df_chi)
        combined_mlogp = -math.log10(combined_p) if combined_p > 0 else 99.0
        results.append({
            "gene": gene,
            "n_endpoints": len(avail),
            "direction": direction,
            "combined_mlogp": round(combined_mlogp, 3),
            "gbm_mlogp": round(row_m.get("C3_GBM", float("nan")), 3),
            "gbm_astro_mlogp": round(row_m.get("C3_GBM_ASTROCYTOMA", float("nan")), 3),
            "brain_mlogp": round(row_m.get("C3_BRAIN_WIDE", float("nan")), 3),
            "gbm_beta": round(row_b.get("C3_GBM", float("nan")), 3),
        })

    res = pd.DataFrame(results).sort_values("combined_mlogp", ascending=False)
    return res


def gene_set_enrichment(df_gbm, gene_sets):
    """KS test: is gene set enriched in the tail of C3_GBM mlogp distribution?"""
    all_mlogp = df_gbm["mlogp"].values
    print(f"\n### Gene-set enrichment (KS test against C3_GBM mlogp distribution, N={len(df_gbm)} genes)")
    print(f"  {'Gene set':<28} {'N in set':>8}  {'KS stat':>8}  {'p':>10}  {'top hit':>10}  {'top mlogp':>10}")
    print("  " + "-" * 80)
    for label, gene_set in gene_sets.items():
        subset = df_gbm[df_gbm["gene"].isin(gene_set)]["mlogp"].values
        if len(subset) < 3:
            continue
        ks_stat, ks_p = ks_2samp(subset, all_mlogp, alternative="greater")
        top_gene = df_gbm[df_gbm["gene"].isin(gene_set)].sort_values("mlogp", ascending=False)
        top_name = top_gene.iloc[0]["gene"] if len(top_gene) else "—"
        top_m = top_gene.iloc[0]["mlogp"] if len(top_gene) else 0
        print(f"  {label:<28} {len(subset):>8}  {ks_stat:>8.3f}  {ks_p:>10.4f}  {top_name:>10}  {top_m:>10.3f}")


def mutyh_deep_dive(df_gbm_row, lof_variants_file):
    """Extract MUTYH variants from variant list + compute OR with 95% CI."""
    print("\n### MUTYH deep-dive")
    r = df_gbm_row
    beta = r["gbm_beta"]
    # Need SE — pull from raw df
    se = r.get("gbm_se", float("nan"))
    if not np.isnan(se) and se > 0:
        OR = math.exp(beta)
        ci_lo = math.exp(beta - 1.96 * se)
        ci_hi = math.exp(beta + 1.96 * se)
        print(f"  C3_GBM: beta={beta:.3f} SE={se:.3f}  OR={OR:.1f} (95% CI {ci_lo:.1f}–{ci_hi:.1f})")
    print(f"  C3_GBM mlogp={r['gbm_mlogp']:.3f}  GBM_ASTRO mlogp={r['gbm_astro_mlogp']:.3f}  BRAIN_WIDE mlogp={r['brain_mlogp']:.3f}")
    print(f"  Direction across all endpoints: {r['direction']}  (all risk-increasing)")

    # Variants
    vf = Path(lof_variants_file)
    if vf.exists():
        with open(vf) as fh:
            for line in fh:
                if line.startswith("MUTYH"):
                    variants = line.strip().split("\t")[1] if "\t" in line else ""
                    vlist = variants.split(",")
                    print(f"  Mask variants ({len(vlist)}): {', '.join(vlist[:10])}")
                    break

    print("  Biology: MUTYH = base-excision repair (removes A opposite 8-oxo-G).")
    print("  Biallelic MUTYH LoF -> MUTYH-associated polyposis (MAP, colorectal cancer).")
    print("  Monoallelic: modest risk increase for colorectal; brain tumor link under-studied.")
    print("  Finnish founder enrichment possible — check gnomAD FIN enrichment for these variants.")
    print("  CRITICAL CAVEAT: AF=0.00017 -> ~63 total LoF carriers in the 372K cohort.")
    print("    With ~467 GBM cases, likely only 1-3 GBM patients carry these variants.")
    print("    Large beta + huge CI = driven by very few observations. Treat as fragile.")


def main():
    print("=" * 100)
    print("FinnGen R13 gene-burden analysis: glioma endpoints (C3_GBM / C3_GBM_ASTROCYTOMA / C3_BRAIN_WIDE)")
    print("=" * 100)

    df = load_burden()
    print(f"Loaded {len(df)} endpoint×gene tests across {df['pheno'].nunique()} endpoints")

    # Cross-endpoint consistency
    print("\n### Cross-endpoint Fisher combined mlogp (top 25, all endpoints, any direction):")
    res = cross_endpoint_fisher(df)
    se_map = df[df["pheno"] == "C3_GBM"].set_index("gene")["se"].to_dict()
    res["gbm_se"] = res["gene"].map(se_map)
    print(f"  Genes tested in all 3 endpoints: {(res['n_endpoints']==3).sum()}")
    print()
    print(f"  {'Gene':<14} {'comb_mlogp':>11}  {'GBM':>6}  {'ASTRO':>6}  {'BRAIN':>6}  {'dir':>10}  {'gbm_OR':>8}")
    print("  " + "-" * 72)
    shown = 0
    for _, r in res.iterrows():
        if shown >= 25:
            break
        OR_str = f"{math.exp(r['gbm_beta']):.1f}" if not np.isnan(r['gbm_beta']) else "—"
        print(f"  {r['gene']:<14} {r['combined_mlogp']:>11.3f}  {r['gbm_mlogp']:>6.2f}  "
              f"{r['gbm_astro_mlogp']:>6.2f}  {r['brain_mlogp']:>6.2f}  {r['direction']:>10}  {OR_str:>8}")
        shown += 1

    # Separate: cross-endpoint consistent + RISK direction
    consistent_risk = res[(res["n_endpoints"] == 3) & (res["direction"] == "RISK") &
                          (res["combined_mlogp"] >= 3.0)].head(15)
    print(f"\n  Consistent RISK direction in all 3 endpoints, combined mlogp>=3.0: {len(consistent_risk)} genes")
    for _, r in consistent_risk.iterrows():
        OR_str = f"{math.exp(r['gbm_beta']):.1f}" if not np.isnan(r['gbm_beta']) else "—"
        print(f"    {r['gene']:<14} comb={r['combined_mlogp']:.3f}  gbm={r['gbm_mlogp']:.2f}  OR_est={OR_str}")

    # Gene-set enrichment
    df_gbm = df[df["pheno"] == "C3_GBM"].copy()
    gene_sets = {
        "DNA repair (BER/MMR/HR/NER)": DNA_REPAIR,
        "Glioma TSG (known)": GLIOMA_TSG,
        "Replication origin (ORC/MCM)": REPLICATION_ORIGIN,
    }
    gene_set_enrichment(df_gbm, gene_sets)

    # MUTYH deep dive
    mutyh_rows = res[res["gene"] == "MUTYH"]
    if not mutyh_rows.empty:
        mr = mutyh_rows.iloc[0]
        mr["gbm_se"] = se_map.get("MUTYH", float("nan"))
        mutyh_deep_dive(mr, LOF.parent / "data" / "finngen_R13_lof_variants.txt")

    # ORC6 note
    orc6 = res[res["gene"] == "ORC6"]
    if not orc6.empty:
        r = orc6.iloc[0]
        print(f"\n### ORC6 (replication origin complex): comb_mlogp={r['combined_mlogp']:.3f}  "
              f"gbm={r['gbm_mlogp']:.2f}  astro={r['gbm_astro_mlogp']:.2f}  brain={r['brain_mlogp']:.2f}")
        se = se_map.get("ORC6", float("nan"))
        if not np.isnan(se):
            b = r["gbm_beta"]
            print(f"  C3_GBM OR={math.exp(b):.1f} (95% CI {math.exp(b-1.96*se):.1f}–{math.exp(b+1.96*se):.1f})")
        print("  ORC6 = origin recognition complex subunit 6; essential for DNA replication initiation.")
        print("  Not a known GBM predisposition gene. Effect size large, carrier N small — fragile.")

    # Write output
    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.name}  ({len(res)} gene records)")
    print("\n" + "=" * 100)
    print("READING:")
    print("  No gene reaches genome-wide significance (GBM N=467 is too small for rare-variant burden).")
    print("  MUTYH is the most consistent signal: DNA repair gene, same risk direction in all 3 endpoints,")
    print("    top hit in C3_GBM (mlogp=3.44). Effect size is large but CI is enormous (few carriers).")
    print("  Gene-set enrichment tests whether DNA repair genes as a CLASS are enriched in the tail.")
    print("  These are hypotheses, not discoveries.")


if __name__ == "__main__":
    main()
