#!/usr/bin/env python3
"""
Is the CDKN2A/B-null GBM adhesion dependency really a YAP/TAZ-TEAD
MECHANOTRANSDUCTION dependency -- and is TEAD/YAP a better-positioned node than FAK?

Motivation (from the full CRISPR ranking in depmap_synleth_gbm.csv):
  WWTR1/TAZ  q=3.6e-4  is a STRONGER selective dependency than PTK2/FAK q=0.037,
  and TEAD1 (q=0.016) is also selectively essential. YAP/TAZ-TEAD is the
  transcriptional OUTPUT of focal-adhesion / mechanotransduction signalling, and
  TEAD palmitoyl-pocket inhibitors (VT3989, IK-930) are in active clinical trials.
  If the dependency is mechanotransduction, TEAD may be a cleaner, more novel drug
  node than FAK (which has already failed as a GBM monotherapy).

This script answers three questions, honestly:
  (Q1) GENOME-WIDE: where do YAP1/WWTR1/TEAD1-4 rank among all selective
       dependencies in CDKN2A/B-null CNS lines? (delta, p, BH-q, rank)
       FAK module reported alongside for a head-to-head; PRMT5/MAT2A as the
       MTAP-loss positive control; Hippo negative-regulators (NF2/LATS/STK/SAV/MOB)
       as an orientation check (knockout ACTIVATES YAP -> should NOT be selectively
       essential, may even favour growth).
  (Q2) MTAP SEPARATION (the lesson from FAK): is the YAP/TEAD dependency separable
       from the 9p21/MTAP co-deletion? groups A (null&MTAP-null), B (null&MTAP-intact),
       C (intact); B-vs-C, within-null A-vs-B, OLS effect~cdkn_null+mtap_null; CNS +
       pan-cancer; PRMT5 positive control.
  (Q3) Paralog caveat: YAP1<->WWTR1 and TEAD1-4 are redundant paralog families;
       single-gene CRISPR KO can be buffered, so individual-gene essentiality
       UNDERSTATES pathway dependence. Interpreted in the printout.

All DepMap 24Q4, CNS lineage. Hypothesis generation only.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

DATA = Path.home() / "finngen-triage" / "depmap"
HOM = 0.2          # relative-CN homozygous-deletion proxy
MIN_N = 5

YAP_TEAD = ["YAP1","WWTR1","TEAD1","TEAD2","TEAD3","TEAD4"]
HIPPO_NEG = ["NF2","LATS1","LATS2","STK3","STK4","SAV1","MOB1A","MOB1B"]  # YAP repressors
FAK = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK"]
CTRL = ["PRMT5","MAT2A"]
ALL_GENES = YAP_TEAD + HIPPO_NEG + FAK + CTRL

# ---------- CNS lineage (broad match, to match depmap_synleth_gbm.csv) ----------
mdl = pd.read_csv(DATA / "Model.csv", low_memory=False)
idc = next((c for c in mdl.columns if c.lower() in ("modelid","depmap_id","model_id")), mdl.columns[0])
lin = next((c for c in mdl.columns if "lineage" in c.lower()), None)
dis = next((c for c in mdl.columns if "disease" in c.lower() or "tissue" in c.lower()), None)
mask = pd.Series(False, index=mdl.index)
for c in (lin, dis):
    if c: mask |= mdl[c].fillna("").str.lower().str.contains("brain|glioma|glioblast|cns|nervous")
cns_ids = set(mdl[mask][idc].dropna())
print(f"CNS/brain lines (broad): {len(cns_ids)}")

# ---------- copy number: CDKN2A/B + MTAP (all lines) ----------
cnid = pd.read_csv(DATA / "OmicsCNGene.csv", nrows=0).columns[0]
cn = pd.read_csv(DATA / "OmicsCNGene.csv",
                 usecols=[cnid,"CDKN2A (1029)","CDKN2B (1030)","MTAP (4507)"], index_col=cnid)
cn.columns = ["CDKN2A","CDKN2B","MTAP"]
cn["cdkn_null"] = ((cn.CDKN2A < HOM) | (cn.CDKN2B < HOM)).astype(int)
cn["mtap_null"] = (cn.MTAP < HOM).astype(int)

# =====================================================================
# Q1 — GENOME-WIDE differential essentiality (CNS, null vs intact)
# =====================================================================
print("\nloading full CRISPRGeneEffect.csv (428 MB)...")
ge = pd.read_csv(DATA / "CRISPRGeneEffect.csv", index_col=0)
ge = ge.loc[ge.index.isin(cns_ids)]
cn_cns = cn.loc[cn.index.isin(ge.index)]
null_ids = cn_cns[cn_cns.cdkn_null==1].index
int_ids  = cn_cns[cn_cns.cdkn_null==0].index
null_ids = [m for m in null_ids if m in ge.index]
int_ids  = [m for m in int_ids  if m in ge.index]
print(f"CNS genome-wide test: {len(null_ids)} CDKN2A/B-null vs {len(int_ids)} intact")

nullM, intM = ge.loc[null_ids], ge.loc[int_ids]
rows=[]
for g in ge.columns:
    nv = nullM[g].dropna().values; iv = intM[g].dropna().values
    if len(nv)<MIN_N or len(iv)<MIN_N: continue
    try: _,p = mannwhitneyu(nv, iv, alternative="less")
    except ValueError: continue
    rows.append((g.split(" (")[0], float(np.mean(nv)-np.mean(iv)), float(p)))
res = pd.DataFrame(rows, columns=["gene","delta","p"])
res["q"] = multipletests(res.p.values, method="fdr_bh")[1]
res = res.sort_values("delta").reset_index(drop=True)          # most essential-in-null first
res["rank"] = np.arange(1, len(res)+1)
ngene = len(res); nsig = int((res.q<0.05).sum())
print(f"genome-wide: {ngene} genes tested, {nsig} selective dependencies at q<0.05\n")

def show(genes, header):
    print(f"=== {header} ===")
    print(f"  {'gene':<8}{'delta':>9}{'p':>11}{'q(BH)':>11}{'rank/'+str(ngene):>12}")
    sub = res[res.gene.isin(genes)].copy()
    # preserve requested order
    sub["o"]=sub.gene.map({g:i for i,g in enumerate(genes)}); sub=sub.sort_values("o")
    for _,r in sub.iterrows():
        flag = " *" if (r.q<0.05 and r.delta<0) else (" (pos)" if r.delta>0 else "")
        print(f"  {r.gene:<8}{r.delta:>9.3f}{r.p:>11.2e}{r.q:>11.2e}{int(r['rank']):>12}{flag}")
    print()

show(YAP_TEAD, "Q1a  YAP/TAZ-TEAD effectors (drug-target nodes)")
show(FAK,      "Q1b  FAK / focal-adhesion module (reference)")
show(HIPPO_NEG,"Q1c  Hippo NEGATIVE regulators (KO activates YAP; expect NOT essential)")
show(CTRL,     "Q1d  PRMT5 / MAT2A (MTAP-loss positive control)")

# dump genes-of-interest genome-wide stats for the figure
res[res.gene.isin(ALL_GENES)].to_csv("yap_genomewide_goi.csv", index=False)

# =====================================================================
# Q2 — MTAP separation for the YAP/TEAD module (CNS + pan-cancer)
# =====================================================================
ce_all = pd.read_csv(DATA / "CRISPRGeneEffect.csv", nrows=0).columns
keep = [ce_all[0]] + [c for c in ce_all if any(c.startswith(f"{g} (") for g in ALL_GENES)]
mod = pd.read_csv(DATA / "CRISPRGeneEffect.csv", usecols=keep, index_col=ce_all[0])
mod.columns = [c.split(" (")[0] for c in mod.columns]
df = cn.join(mod, how="inner")
df.assign(cns=df.index.isin(cns_ids).astype(int)).to_csv("yap_module_perline.csv")  # for figure

def mw(x,y,alt):
    x=x.dropna(); y=y.dropna()
    if len(x)<3 or len(y)<3: return np.nan,np.nan,len(x),len(y)
    _,p=mannwhitneyu(x,y,alternative=alt); return x.median()-y.median(),p,len(x),len(y)

def separation(frame, label, test_genes):
    A=frame[(frame.cdkn_null==1)&(frame.mtap_null==1)]
    B=frame[(frame.cdkn_null==1)&(frame.mtap_null==0)]
    C=frame[(frame.cdkn_null==0)]
    nnull=len(A)+len(B)
    phi=np.corrcoef(frame.cdkn_null,frame.mtap_null)[0,1]
    print(f"\n########## MTAP SEPARATION — {label} ##########")
    print(f"  grpA null&MTAP-null={len(A)}  grpB null&MTAP-INTACT={len(B)}  grpC intact={len(C)}"
          f"  | co-del {len(A)}/{nnull}={len(A)/max(nnull,1):.0%}  phi={phi:.2f}")
    print(f"  (1) B vs C  [alt B<C; negative delta = more essential in MTAP-intact null]")
    for g in test_genes:
        d,p,nb,nc=mw(B[g],C[g],"less"); print(f"      {g:7s} d={d:+.3f} p={p:.3g} (nB={nb},nC={nc}){' *' if p<0.05 else ''}")
    print(f"  (2) within null: A vs B  [two-sided; ~0 => not MTAP-driven]")
    for g in test_genes:
        d,p,na,nb=mw(A[g],B[g],"two-sided"); print(f"      {g:7s} d(A-B)={d:+.3f} p={p:.3g} (nA={na},nB={nb})")
    print(f"  (3) OLS effect ~ cdkn_null + mtap_null")
    X=sm.add_constant(frame[["cdkn_null","mtap_null"]])
    for g in test_genes + CTRL:
        m=sm.OLS(frame[g],X,missing="drop").fit()
        fl=""
        if m.pvalues['cdkn_null']<0.05 and m.params['cdkn_null']<0: fl+=" cdkn*"
        if m.pvalues['mtap_null']<0.05 and m.params['mtap_null']<0: fl+=" mtap*"
        tag = "  <-- MTAP ctrl" if g in CTRL else ""
        print(f"      {g:7s} cdkn b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.2g} | "
              f"mtap b={m.params['mtap_null']:+.3f} p={m.pvalues['mtap_null']:.2g}{fl}{tag}")

separation(df[df.index.isin(cns_ids)], "CNS", YAP_TEAD)
separation(df,                          "PAN-CANCER", YAP_TEAD)
print("\nDONE.")
