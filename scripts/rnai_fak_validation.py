#!/usr/bin/env python3
"""
Push A4: DepMap RNAi (DEMETER2 combined) orthogonal validation of the
focal-adhesion / FAK module dependency in CDKN2A/B-null lines.

RNAi (shRNA) is a DIFFERENT silencing technology than CRISPR. If the same
module is more essential in null lines under RNAi too, the dependency is not a
CRISPR-specific artifact (e.g. copy-number cutting toxicity at the deleted locus).

DEMETER2 score: more NEGATIVE = more essential / stronger dependency.
Hypothesis: null lines MORE dependent => null score < intact score.
"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

HOMDEL = 0.2
MODULE = {
    "core":  ["ITGAV","TLN1","VCL","FERMT2","ITGB5","PTK2","ILK"],
    "actin": ["ACTR2","ACTR3","ARPC1B","ARPC2","ARPC3","ARPC4","ARPC5"],
}
ALL_GENES = MODULE["core"] + MODULE["actin"]

# ---- CDKN2A/B status from DepMap CN ----
id_col = pd.read_csv("depmap/OmicsCNGene.csv", nrows=0).columns[0]
cn = pd.read_csv("depmap/OmicsCNGene.csv",
                 usecols=[id_col,"CDKN2A (1029)","CDKN2B (1030)"], index_col=id_col)
cn.columns = ["CDKN2A","CDKN2B"]
null = (cn["CDKN2A"]<HOMDEL)&(cn["CDKN2B"]<HOMDEL)
status = pd.Series(np.where(null,"null","intact"), index=cn.index)

model = pd.read_csv("depmap/Model.csv", usecols=["ModelID","CCLEName","OncotreeLineage"])
model = model.set_index("ModelID")
model["status"] = status
model = model.dropna(subset=["status","CCLEName"])
# map CCLEName -> (status, lineage)
ccle_map = model.set_index("CCLEName")[["status","OncotreeLineage"]]

# ---- Load RNAi matrix, select module rows ----
mat = pd.read_csv("rnai/D2_combined_gene_dep_scores.csv", index_col=0)
# index labels like 'ITGAV (3685)' ; build symbol->label
sym2lab = {}
for lab in mat.index:
    sym = lab.split(" (")[0]
    sym2lab.setdefault(sym, lab)
rows = {g: sym2lab[g] for g in ALL_GENES if g in sym2lab}
print("module genes found in RNAi:", list(rows.keys()))
sub = mat.loc[list(rows.values())].T   # lines x genes
sub.columns = list(rows.keys())

# ---- attach status/lineage by CCLE_ID (== CCLEName) ----
sub = sub.join(ccle_map, how="inner")
print(f"\nRNAi lines with CDKN2A/B status: {len(sub)} "
      f"({(sub.status=='null').sum()} null / {(sub.status=='intact').sum()} intact)")
cns = sub[sub["OncotreeLineage"]=="CNS/Brain"]
print(f"CNS/Brain lines: {len(cns)} ({(cns.status=='null').sum()} null / {(cns.status=='intact').sum()} intact)")

def test(frame, gene, label):
    n = frame[frame.status=="null"][gene].dropna()
    i = frame[frame.status=="intact"][gene].dropna()
    if len(n)<3 or len(i)<3:
        return f"  {gene:7s} [{label}] too few (n={len(n)}/{len(i)})"
    _,p2 = mannwhitneyu(n,i,alternative="two-sided")
    _,pl = mannwhitneyu(n,i,alternative="less")  # null MORE dependent (more negative)
    flag = "  <-- null more dependent" if (n.median()<i.median() and pl<0.05) else ""
    return (f"  {gene:7s} [{label}] null med={n.median():+.3f} (n={len(n)}) | "
            f"intact med={i.median():+.3f} (n={len(i)}) | 2-sided p={p2:.3g} | null<intact p={pl:.3g}{flag}")

for scope,frame in [("PAN-CANCER",sub),("CNS/Brain",cns)]:
    print(f"\n=== {scope} ===")
    for grp,genes in MODULE.items():
        print(f" -- {grp} --")
        for g in genes:
            if g in sub.columns:
                print(test(frame,g,scope[:3]))

sub.to_csv("rnai_fak_validation.csv")
print("\nwrote rnai_fak_validation.csv")
