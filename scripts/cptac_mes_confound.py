#!/usr/bin/env python3
"""
Push B: mechanism + confound control.

The focal-adhesion/FAK program is biologically tied to the MESENCHYMAL GBM state.
The danger: maybe CDKN2A/B-null tumors are just more mesenchymal, and FAK comes
along for free (confounded). Test directly in CPTAC-GBM transcriptomics:

  1. Are null tumors mesenchymal-skewed?           (MES score: null vs intact)
  2. Is the FAK module elevated in null tumors?     (FAK score: null vs intact)
  3. Does FAK track MES?                            (Spearman corr)
  4. KEY: does null status predict FAK module AFTER adjusting for MES?
     OLS  FAK ~ MES + is_null   -> p(is_null).  If significant, the FAK signal
     is NOT merely a mesenchymal-subtype artifact.

Signatures: MES = canonical mesenchymal-GBM markers (Verhaak/Neftel MES metaprogram).
FAK module = the focal-adhesion dependency genes.
"""
import cptac, pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr
import warnings; warnings.filterwarnings("ignore")

DEEP, INTACT_HI = -1.0, -0.3
MES = ["CHI3L1","CD44","VIM","SERPINE1","TGFBI","TIMP1","ANXA1","ANXA2","A2M",
       "LGALS1","LGALS3","CAV1","RELB","NAMPT","ADM","SOCS3","EMP1","CTSB",
       "IGFBP3","LOX","PLAUR","MT2A","S100A4","CLIC1","TNC"]
FAK = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK","PXN","BCAR1"]

g = cptac.Gbm()
cnv = g.get_CNV("bcm")
tx  = g.get_transcriptomics("washu")

def col_for(df, gene):
    return [c for c in df.columns if (c[0] if isinstance(c,tuple) else c)==gene]

# classify
a = cnv[col_for(cnv,"CDKN2A")[0]]; b = cnv[col_for(cnv,"CDKN2B")[0]]
cls = pd.Series(index=cnv.index, dtype=object)
cls[(a<=DEEP)&(b<=DEEP)] = "null"
cls[(a>INTACT_HI)&(b>INTACT_HI)] = "intact"
cls = cls.dropna()

def sig_score(genes):
    cols=[]
    for gn in genes:
        c=col_for(tx,gn)
        if c: cols.append(tx[c[0]])
    M=pd.concat(cols,axis=1)
    Z=(M-M.mean())/M.std()
    return Z.mean(axis=1)

mes = sig_score(MES); fak = sig_score(FAK)
df = pd.DataFrame({"MES":mes,"FAK":fak}).join(cls.rename("status"),how="inner").dropna()
df["is_null"] = (df.status=="null").astype(int)
print(f"tumors: {df.is_null.sum()} null / {(1-df.is_null).sum()} intact")

def mw(col):
    n=df[df.is_null==1][col]; i=df[df.is_null==0][col]
    _,p=mannwhitneyu(n,i,alternative="two-sided")
    _,pg=mannwhitneyu(n,i,alternative="greater")
    print(f"  {col}: null med={n.median():+.3f} intact med={i.median():+.3f} | 2s p={p:.3g} | null>intact p={pg:.3g}")

print("\n[1/2] signature scores, null vs intact:"); mw("MES"); mw("FAK")
rho,pr = spearmanr(df.MES, df.FAK)
print(f"\n[3] Spearman FAK vs MES: rho={rho:.3f} p={pr:.3g}  (expected positive)")

print("\n[4] OLS  FAK ~ MES + is_null  (does null add signal beyond mesenchymal state?)")
X = sm.add_constant(df[["MES","is_null"]])
m = sm.OLS(df.FAK, X).fit()
print(m.summary().tables[1])
b_null = m.params["is_null"]; p_null = m.pvalues["is_null"]
print(f"\n=> is_null coefficient = {b_null:+.3f}, p = {p_null:.3g}")
print("   (significant & positive => FAK elevation in null is NOT just mesenchymal confound)")

# also: is MES itself higher in null? interpret
df.to_csv("cptac_mes_confound.csv")
print("\nwrote cptac_mes_confound.csv")
