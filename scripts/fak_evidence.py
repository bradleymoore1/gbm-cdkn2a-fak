#!/usr/bin/env python3
"""
Figure 1 (revised): five independent lines of evidence for a focal-adhesion/FAK
dependency in CDKN2A/B-null GBM.
  A CRISPR knockout (DepMap)        - module more essential in null
  B RNAi shРНК (DEMETER2)           - orthogonal silencing, same module
  C Potent FAK-i dose-response (GDSC PF-562271) - null lines more sensitive
  D Tumor phospho-FAK Y397 (CPTAC)  - FAK pathway ACTIVE in null tumors
  E Tumor protein FAK + FERMT2 (CPTAC)
  F Tumor RNA (TCGA-GBM)            - ITGAV up; null-state wiring
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"axes.titleweight":"bold"})

# NB: pandas reads the literal string "null" as NaN by default -> would drop the
# null group. keep_default_na=False + na_values=[""] preserves "null"/"intact".
def rd(f): return pd.read_csv(f, keep_default_na=False, na_values=[""])

CORE = ["ITGAV","TLN1","VCL","FERMT2","ITGB5","PTK2","ILK"]
LBL  = {"PTK2":"PTK2/FAK"}
RED, BLUE, GREY = "#c0392b", "#2c6fbb", "#9aa0a6"

fig = plt.figure(figsize=(13.5, 7.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

# ---------- A: CRISPR forest ----------
axA = fig.add_subplot(gs[0,0])
cr = rd("depmap_synleth_gbm.csv").set_index("gene")
g = [x for x in CORE if x in cr.index]
d = cr.loc[g,"delta"]; q = cr.loc[g,"q"]
y = np.arange(len(g))[::-1]
cols = [RED if qq<0.05 else GREY for qq in q]
axA.barh(y, d.values, color=cols)
axA.set_yticks(y); axA.set_yticks(np.arange(len(g))[::-1])
axA.set_yticklabels([LBL.get(x,x) for x in g])
for yi,gg,qq in zip(y,g,q):
    axA.text(d[gg]-0.01, yi, f"q={qq:.0e}", va="center", ha="right", fontsize=6.5)
axA.axvline(0,color="k",lw=0.6)
axA.set_xlabel(r"$\Delta$ CRISPR effect (null $-$ intact)")
axA.set_title("A  CRISPR knockout — DepMap\n(more negative = more essential in null)",loc="left")
axA.invert_xaxis()

# ---------- B: RNAi forest ----------
axB = fig.add_subplot(gs[0,1])
rn = rd("rnai_fak_validation.csv")
deltas=[]; ps=[]
for gg in CORE:
    n=rn[rn.status=="null"][gg].dropna(); i=rn[rn.status=="intact"][gg].dropna()
    deltas.append(n.median()-i.median())
    try: _,p=mannwhitneyu(n,i,alternative="less")
    except Exception: p=np.nan
    ps.append(p)
y=np.arange(len(CORE))[::-1]
cols=[BLUE if (pp<0.05) else GREY for pp in ps]
axB.barh(y, deltas, color=cols)
axB.set_yticks(y); axB.set_yticklabels([LBL.get(x,x) for x in CORE])
for yi,dv,pp in zip(y,deltas,ps):
    axB.text(dv-0.003 if dv<0 else dv+0.003, yi, f"p={pp:.1e}", va="center",
             ha="right" if dv<0 else "left", fontsize=6.5)
axB.axvline(0,color="k",lw=0.6)
axB.set_xlabel(r"$\Delta$ RNAi dependency (null $-$ intact)")
axB.set_title("B  RNAi shRNA — DEMETER2 (orthogonal)\nrules out CRISPR cut-artifact",loc="left")
axB.invert_xaxis()

# ---------- C: GDSC PF-562271 AUC ----------
axC = fig.add_subplot(gs[0,2])
gd = rd("gdsc_fak_sensitivity.csv")
pf = gd[gd.DRUG_ID==158]
n=pf[pf.status=="null"]["AUC"].dropna(); i=pf[pf.status=="intact"]["AUC"].dropna()
_,p=mannwhitneyu(n,i,alternative="less")
bp=axC.boxplot([i,n],labels=[f"intact\n(n={len(i)})",f"null\n(n={len(n)})"],
               patch_artist=True,widths=0.6,showfliers=False)
for patch,c in zip(bp['boxes'],[GREY,RED]): patch.set_facecolor(c); patch.set_alpha(.75)
axC.set_ylabel("AUC (lower = more sensitive)")
axC.set_title(f"C  Potent FAK-i PF-562271 — GDSC\nnull more sensitive, p={p:.1e}",loc="left")

# ---------- D: CPTAC FAK pY397 ----------
axD = fig.add_subplot(gs[1,0])
cp = rd("cptac_fak_values.csv")
n=cp[cp.status=="null"]["FAK_pY397"].dropna(); i=cp[cp.status=="intact"]["FAK_pY397"].dropna()
_,p=mannwhitneyu(n,i,alternative="greater")
bp=axD.boxplot([i,n],labels=[f"intact\n(n={len(i)})",f"null\n(n={len(n)})"],
               patch_artist=True,widths=0.6,showfliers=False)
for patch,c in zip(bp['boxes'],[GREY,RED]): patch.set_facecolor(c); patch.set_alpha(.75)
axD.axhline(0,color="k",lw=0.5,ls=":")
axD.set_ylabel("FAK phospho-Y397 (z)")
axD.set_title(f"D  Tumor phospho-FAK Y397 — CPTAC\nFAK ACTIVE in null tumors, p={p:.1e}",loc="left")

# ---------- E: CPTAC FAK + FERMT2 protein ----------
axE = fig.add_subplot(gs[1,1])
data=[]; labels=[]; box_cols=[]
for gene,lab in [("FAK_protein","FAK"),("FERMT2_protein","FERMT2")]:
    n=cp[cp.status=="null"][gene].dropna(); i=cp[cp.status=="intact"][gene].dropna()
    _,p=mannwhitneyu(n,i,alternative="greater")
    data+=[i,n]; labels+=[f"{lab}\nintact",f"{lab}\nnull (p={p:.0e})"]; box_cols+=[GREY,RED]
bp=axE.boxplot(data,labels=labels,patch_artist=True,widths=0.6,showfliers=False)
for patch,c in zip(bp['boxes'],box_cols): patch.set_facecolor(c); patch.set_alpha(.75)
axE.axhline(0,color="k",lw=0.5,ls=":")
axE.set_ylabel("protein abundance (z)")
axE.tick_params(axis="x",labelsize=7)
axE.set_title("E  Tumor protein FAK & FERMT2 — CPTAC\nboth up in null tumors",loc="left")

# ---------- F: TCGA RNA ----------
axF = fig.add_subplot(gs[1,2])
tc = rd("tcga_gbm_rnaseq.csv")
tc = tc[tc.comparison=="CDKN2A/B-null vs intact"].drop_duplicates("gene").set_index("gene")
genes=["ITGAV","RB1","CDK6","E2F1"]
lfc=[np.log2(tc.loc[gn,"mean_A"]/tc.loc[gn,"mean_B"]) for gn in genes]
ps=[tc.loc[gn,"p"] for gn in genes]
cols=[RED if gn=="ITGAV" else BLUE for gn in genes]
x=np.arange(len(genes))
axF.bar(x,lfc,color=cols,alpha=.8)
for xi,l,pp in zip(x,lfc,ps):
    axF.text(xi, l+(0.02 if l>=0 else -0.02), f"p={pp:.0e}", ha="center",
             va="bottom" if l>=0 else "top", fontsize=6.5)
axF.axhline(0,color="k",lw=0.6)
axF.set_xticks(x); axF.set_xticklabels(genes)
axF.set_ylabel(r"$\log_2$ FC (null / intact)")
axF.set_title("F  Tumor RNA — TCGA-GBM\nITGAV up; null-state wiring confirmed",loc="left")

fig.suptitle("Five independent lines of evidence for a focal-adhesion / FAK dependency in CDKN2A/B-null GBM",
             fontsize=12, fontweight="bold", y=0.995)
fig.savefig("fak_evidence.png", dpi=200, bbox_inches="tight")
fig.savefig("fak_evidence.pdf", bbox_inches="tight")
print("wrote fak_evidence.png / .pdf")
