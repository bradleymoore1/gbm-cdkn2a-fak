#!/usr/bin/env python3
"""
MTAP-vs-CDKN2A/B separability, PAN-CANCER.

The CNS subset is 93% collinear (only n=4 CDKN2A/B-null & MTAP-intact), too few
to separate the two. Pan-cancer, 9p21 deletions vary in size: many lines delete
CDKN2A/B but SPARE MTAP, which breaks the collinearity and lets us ask cleanly
which gene the FAK dependency tracks.

Same groups, all lineages:
  grpA  CDKN2A/B-null & MTAP-null     (co-deleted)
  grpB  CDKN2A/B-null & MTAP-INTACT   (key separating group)
  grpC  CDKN2A/B-intact               (reference)
Plus PRMT5/MAT2A positive control (must track MTAP, not CDKN2A/B).
"""
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

HOM = 0.2
FAK = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK"]
CTRL = ["PRMT5","MAT2A"]

cnid = pd.read_csv("depmap/OmicsCNGene.csv", nrows=0).columns[0]
cn = pd.read_csv("depmap/OmicsCNGene.csv",
                 usecols=[cnid,"CDKN2A (1029)","CDKN2B (1030)","MTAP (4507)"],
                 index_col=cnid)
cn.columns=["CDKN2A","CDKN2B","MTAP"]

ceid = pd.read_csv("depmap/CRISPRGeneEffect.csv", nrows=0).columns[0]
ce_all = pd.read_csv("depmap/CRISPRGeneEffect.csv", nrows=0).columns
keep=[ceid]+[c for c in ce_all if any(c.startswith(f"{g} (") for g in FAK+CTRL)]
ce = pd.read_csv("depmap/CRISPRGeneEffect.csv", usecols=keep, index_col=ceid)
ce.columns=[c.split(" (")[0] for c in ce.columns]

df = cn.join(ce, how="inner")
df["cdkn_null"]=((df.CDKN2A<HOM)|(df.CDKN2B<HOM)).astype(int)
df["mtap_null"]=(df.MTAP<HOM).astype(int)

A=df[(df.cdkn_null==1)&(df.mtap_null==1)]
B=df[(df.cdkn_null==1)&(df.mtap_null==0)]
C=df[(df.cdkn_null==0)]
nnull=len(A)+len(B)
print(f"PAN-CANCER lines: {len(df)}")
print(f"  grpA CDKN2A/B-null & MTAP-null : n={len(A)}")
print(f"  grpB CDKN2A/B-null & MTAP-INTACT: n={len(B)}   <-- separating group")
print(f"  grpC CDKN2A/B-intact           : n={len(C)}")
print(f"  MTAP co-deletion among CDKN2A/B-null: {len(A)}/{nnull} = {len(A)/nnull:.0%}")
# correlation of the two indicators
r=np.corrcoef(df.cdkn_null,df.mtap_null)[0,1]
print(f"  phi(cdkn_null, mtap_null) = {r:.2f}  (lower = more separable)")

def mw(x,y,alt="less"):
    x=x.dropna();y=y.dropna()
    if len(x)<3 or len(y)<3: return np.nan,np.nan,len(x),len(y)
    _,p=mannwhitneyu(x,y,alternative=alt); return x.median()-y.median(),p,len(x),len(y)

print("\n=== (1) KEY: FAK in MTAP-INTACT null (grpB) vs intact (grpC); alt B<C ===")
for g in FAK:
    d,p,nb,nc=mw(B[g],C[g]); print(f"  {g:8s} delta={d:+.3f}  p={p:.3g}  (nB={nb},nC={nc}){' *' if p<0.05 else ''}")

print("\n=== (2) Within null: MTAP-null (A) vs MTAP-intact (B); two-sided ===")
for g in FAK:
    d,p,na,nb=mw(A[g],B[g],alt="two-sided"); print(f"  {g:8s} delta(A-B)={d:+.3f}  p={p:.3g}  (nA={na},nB={nb})")

print("\n=== (3) OLS effect ~ cdkn_null + mtap_null (separable now) ===")
X=sm.add_constant(df[["cdkn_null","mtap_null"]])
for g in FAK:
    m=sm.OLS(df[g],X,missing="drop").fit()
    fl=""
    if m.pvalues['cdkn_null']<0.05 and m.params['cdkn_null']<0: fl+=" cdkn*"
    if m.pvalues['mtap_null']<0.05 and m.params['mtap_null']<0: fl+=" mtap*"
    print(f"  {g:8s} cdkn b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.2g} | "
          f"mtap b={m.params['mtap_null']:+.3f} p={m.pvalues['mtap_null']:.2g}{fl}")

print("\n=== (4) POSITIVE CONTROL PRMT5/MAT2A (must load on mtap) ===")
for g in CTRL:
    m=sm.OLS(df[g],X,missing="drop").fit()
    print(f"  {g:8s} cdkn b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.2g} | "
          f"mtap b={m.params['mtap_null']:+.3f} p={m.pvalues['mtap_null']:.2g}")
