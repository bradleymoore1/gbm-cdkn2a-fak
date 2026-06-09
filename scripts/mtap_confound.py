#!/usr/bin/env python3
"""
Confound check: is the CDKN2A/B-null FAK dependency separable from MTAP
co-deletion?

MTAP sits immediately adjacent to CDKN2A on 9p21 and is co-deleted in most
CDKN2A/B-homozygous-deletion tumors. The canonical 9p21-deletion collateral
dependency in the literature is MTAP-loss -> PRMT5/MAT2A (NOT FAK). A reviewer
will ask whether our "CDKN2A/B-null" group is really an "MTAP-null" group and
whether the FAK signal is a 9p21/MTAP artifact.

Strategy (DepMap CRISPR, CNS lineage):
  Classify lines by relative CN < 0.2 (homozygous-deletion proxy):
    grpA  CDKN2A/B-null & MTAP-null     (co-deleted; the common case)
    grpB  CDKN2A/B-null & MTAP-INTACT   (CDKN2A/B-specific deletion; the key group)
    grpC  CDKN2A/B-intact               (reference)
  Tests:
    (1) FAK module: grpB vs grpC  -> if FAK still more essential, dependency is
        CDKN2A/B-driven, not MTAP-driven.
    (2) FAK module: grpA vs grpB  -> does MTAP status change FAK dependency
        within null lines? (no difference => FAK tracks CDKN2A/B, not MTAP)
    (3) OLS effect ~ cdkn_null + mtap_null per gene -> which indicator carries it.
    (4) POSITIVE CONTROL: PRMT5/MAT2A should be more essential in MTAP-null
        (grpA) than MTAP-intact -> validates the classification captures the
        known MTAP collateral lethality.
"""
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

HOM = 0.2
FAK = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK"]
CTRL = ["PRMT5","MAT2A"]   # positive control: MTAP-loss collateral dependency

def col(df, g): return [c for c in df.columns if c.split(" (")[0]==g][0]

# ---- copy number (4 cols only from the 1.4 GB file) ----
cnid = pd.read_csv("depmap/OmicsCNGene.csv", nrows=0).columns[0]
cn = pd.read_csv("depmap/OmicsCNGene.csv",
                 usecols=[cnid,"CDKN2A (1029)","CDKN2B (1030)","MTAP (4507)"],
                 index_col=cnid)
cn.columns = ["CDKN2A","CDKN2B","MTAP"]

# ---- CRISPR effect (id + FAK module + controls) ----
ceid = pd.read_csv("depmap/CRISPRGeneEffect.csv", nrows=0).columns[0]
want = [ceid]+[f"{g} (" for g in FAK+CTRL]
ce_all = pd.read_csv("depmap/CRISPRGeneEffect.csv", nrows=0).columns
keep = [ceid]+[c for c in ce_all if any(c.startswith(w) for w in [f"{g} (" for g in FAK+CTRL])]
ce = pd.read_csv("depmap/CRISPRGeneEffect.csv", usecols=keep, index_col=ceid)
ce.columns = [c.split(" (")[0] for c in ce.columns]

# ---- CNS lineage ----
mdl = pd.read_csv("depmap/Model.csv")
idc = "ModelID" if "ModelID" in mdl.columns else mdl.columns[0]
linc = "OncotreeLineage" if "OncotreeLineage" in mdl.columns else [c for c in mdl.columns if "ineage" in c][0]
cns = set(mdl[mdl[linc]=="CNS/Brain"][idc])

# ---- classify ----
df = cn.join(ce, how="inner")
df = df[df.index.isin(cns)]
df["cdkn_null"] = ((df.CDKN2A<HOM)|(df.CDKN2B<HOM)).astype(int)
df["mtap_null"] = (df.MTAP<HOM).astype(int)

A = df[(df.cdkn_null==1)&(df.mtap_null==1)]   # co-deleted
B = df[(df.cdkn_null==1)&(df.mtap_null==0)]   # CDKN2A/B-specific
C = df[(df.cdkn_null==0)]                       # intact

print(f"CNS lines: {len(df)} total")
print(f"  grpA CDKN2A/B-null & MTAP-null (co-del): n={len(A)}")
print(f"  grpB CDKN2A/B-null & MTAP-INTACT       : n={len(B)}  <-- key group")
print(f"  grpC CDKN2A/B-intact                   : n={len(C)}")
# co-deletion rate among null lines
nnull=len(A)+len(B)
print(f"  MTAP co-deletion rate among CDKN2A/B-null CNS lines: {len(A)}/{nnull} = {len(A)/nnull:.0%}")

def mw(x,y,alt="less"):
    x=x.dropna(); y=y.dropna()
    if len(x)<3 or len(y)<3: return np.nan,np.nan,len(x),len(y)
    _,p=mannwhitneyu(x,y,alternative=alt)
    return x.median()-y.median(), p, len(x), len(y)

print("\n=== (1) KEY: FAK dependency in MTAP-INTACT null lines (grpB) vs intact (grpC) ===")
print("    (negative delta = more essential in null; alt: B<C)")
for g in FAK:
    d,p,nb,nc = mw(B[g],C[g],alt="less")
    flag=" *" if (p<0.05) else ""
    print(f"  {g:8s} delta={d:+.3f}  p={p:.3g}  (nB={nb}, nC={nc}){flag}")

print("\n=== (2) Within null lines: MTAP-null (grpA) vs MTAP-intact (grpB), FAK ===")
print("    (if ~no difference, FAK tracks CDKN2A/B not MTAP; two-sided)")
for g in FAK:
    d,p,na,nb = mw(A[g],B[g],alt="two-sided")
    print(f"  {g:8s} delta(A-B)={d:+.3f}  p={p:.3g}  (nA={na}, nB={nb})")

print("\n=== (3) OLS  effect ~ cdkn_null + mtap_null  (which indicator carries FAK?) ===")
X = sm.add_constant(df[["cdkn_null","mtap_null"]])
for g in FAK:
    y=df[g]
    m=sm.OLS(y,X,missing="drop").fit()
    print(f"  {g:8s} cdkn_null b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.3g} | "
          f"mtap_null b={m.params['mtap_null']:+.3f} p={m.pvalues['mtap_null']:.3g}")

print("\n=== (4) POSITIVE CONTROL: PRMT5/MAT2A should track MTAP, not CDKN2A/B ===")
print("    MTAP-null (grpA) vs MTAP-intact-but-anything; expect more essential in MTAP-null")
mt_null = df[df.mtap_null==1]; mt_int = df[df.mtap_null==0]
for g in CTRL:
    d,p,nn,ni = mw(mt_null[g],mt_int[g],alt="less")
    flag=" *" if p<0.05 else ""
    print(f"  {g:8s} delta(MTAPnull-int)={d:+.3f}  p={p:.3g}  (n={nn}/{ni}){flag}")
print("    OLS for controls (which indicator carries PRMT5/MAT2A):")
for g in CTRL:
    m=sm.OLS(df[g],X,missing="drop").fit()
    print(f"  {g:8s} cdkn_null b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.3g} | "
          f"mtap_null b={m.params['mtap_null']:+.3f} p={m.pvalues['mtap_null']:.3g}")
