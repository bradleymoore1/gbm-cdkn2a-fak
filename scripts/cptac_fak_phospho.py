#!/usr/bin/env python3
"""
Push A3: CPTAC GBM proteogenomics — is the FAK pathway ACTIVE (phosphorylated) at
the protein level in CDKN2A/B-deleted GBM tumors?

RNA (TCGA) showed ITGAV mRNA up in null tumors. Protein/phospho is closer to
function. The key readout is FAK (PTK2) phospho-Y397 — the canonical
autophosphorylation/activation site. Higher pY397 = more active FAK.

Status from CPTAC WES CNV (log2 gene ratio). Deep loss (homdel-like) vs intact.
Hypothesis: null tumors have MORE active FAK => null pY397 > intact.
"""
import cptac, pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

DEEP = -1.0     # log2 ratio <= -1.0  -> deep (homdel-like) loss
INTACT_HI = -0.3  # log2 ratio > -0.3 -> retained
MODULE = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK","PXN","BCAR1","SRC"]

g = cptac.Gbm()
cnv  = g.get_CNV("bcm")
phos = g.get_phosphoproteomics("umich")
prot = g.get_proteomics("umich")

# flatten gene level for cnv (columns may be multiindex of one level=gene)
def gene_col(df, gene):
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        hits = [c for c in cols if c[0]==gene]
        return hits
    return [c for c in cols if c==gene]

# ---- classify tumors by CDKN2A/B deep loss ----
a = cnv[gene_col(cnv,"CDKN2A")[0]]
b = cnv[gene_col(cnv,"CDKN2B")[0]]
cls = pd.Series(index=cnv.index, dtype=object)
cls[(a<=DEEP)&(b<=DEEP)] = "null"
cls[(a>INTACT_HI)&(b>INTACT_HI)] = "intact"
cls = cls.dropna()
print(f"CPTAC GBM tumors: {(cls=='null').sum()} null (deep CDKN2A/B loss) / "
      f"{(cls=='intact').sum()} intact  [ambiguous excluded]")

def compare(values, label, alt="greater"):
    """alt='greater' tests null > intact (more phospho/protein in null)."""
    df = pd.DataFrame({"v":values}).join(cls.rename("status"), how="inner").dropna()
    n = df[df.status=="null"]["v"]; i = df[df.status=="intact"]["v"]
    if len(n)<4 or len(i)<4:
        print(f"  {label:28s} too few (n={len(n)}/{len(i)})"); return
    _,p2 = mannwhitneyu(n,i,alternative="two-sided")
    _,pg = mannwhitneyu(n,i,alternative=alt)
    flag = "  <== null higher" if (n.median()>i.median() and pg<0.05) else ""
    print(f"  {label:28s} null med={n.median():+.3f}(n={len(n)}) "
          f"intact med={i.median():+.3f}(n={len(i)}) 2s p={p2:.3g} null>intact p={pg:.3g}{flag}")

print("\n--- FAK (PTK2) phospho-sites: activation readout ---")
for site in ["Y397","Y576","Y577","Y576Y577","Y570"]:
    hits = [c for c in phos.columns if c[0]=="PTK2" and c[1]==site]
    for c in hits:
        compare(phos[c], f"PTK2 {site} [{c[2][:12]}]")

print("\n--- Module phospho-protein (max site per gene shown individually) ---")
for gene in MODULE:
    hits = [c for c in phos.columns if c[0]==gene]
    # average all sites per gene as a coarse 'pathway phospho' proxy
    if hits:
        avg = phos[hits].mean(axis=1)
        compare(avg, f"{gene} (mean phospho, {len(hits)} sites)")

print("\n--- Module total protein abundance (proteomics) ---")
for gene in MODULE:
    hits = gene_col(prot, gene)
    if hits:
        compare(prot[hits[0]], f"{gene} protein")

print("\n(Phospho hypothesis is one-sided null>intact = FAK pathway more active in deleted tumors.)")

# ---- tidy dump for figure (per-tumor key readouts) ----
out = pd.DataFrame(index=cls.index)
out["status"] = cls
py397 = [c for c in phos.columns if c[0]=="PTK2" and c[1]=="Y397"]
if py397: out["FAK_pY397"] = phos[py397[0]]
fakp = gene_col(prot,"PTK2");      out["FAK_protein"]   = prot[fakp[0]] if fakp else np.nan
ferm = gene_col(prot,"FERMT2");    out["FERMT2_protein"]= prot[ferm[0]] if ferm else np.nan
out = out.dropna(subset=["status"])
out.to_csv("cptac_fak_values.csv")
print("wrote cptac_fak_values.csv", out.shape)
