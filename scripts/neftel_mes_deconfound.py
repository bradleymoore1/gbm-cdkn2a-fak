#!/usr/bin/env python3
"""
Neftel 2019 (GSE131928) single-cell mesenchymal-confound deconvolution.

QUESTION (the mesenchymal confound, distinct from the MTAP confound):
  In bulk CPTAC, the FAK / YAP-TAZ-output signal was flat across CDKN2A/B status
  and fully explained by mesenchymal (MES) state (p=2e-6). At SINGLE-CELL
  resolution, is the FAK module separable from MES state, or is it merely a
  readout of mesenchymal identity?

DESIGN GUARDS (lessons carried in):
  * scRNA pseudoreplication: effective n = PATIENT count (~tens), NOT cell count.
    All inference is patient-level (pseudobulk per patient x state + a mixed model
    with patient random effect). Per-cell r is descriptive only.
  * Signatures are the paper's exact lists (MES 25, FAK 9) + Neftel AC/OPC/NPC.
  * Malignant cells are signature-filtered (no infercnv available); stated as such.
  * Neftel cohort is ~all IDH-wt and ~all CDKN2A/B-deleted -> it can deconfound
    the MESENCHYMAL state but CANNOT re-confirm the CDKN2A/B->FAK link (no intact
    contrast). Reported honestly, not overclaimed.

Usage: python3 neftel_mes_deconfound.py {10x|ss2}
"""
import sys, gzip, numpy as np, pandas as pd
import scanpy as sc, anndata as ad
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

COHORT = sys.argv[1] if len(sys.argv) > 1 else "10x"
DDIR = "/home/brad/finngen-triage/scdata/neftel"
F = {"10x":  f"{DDIR}/GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv.gz",
     "ss2":  f"{DDIR}/GSM3828672_Smartseq2_GBM_IDHwt_processed_TPM.tsv.gz"}[COHORT]
PAT_SPLIT = (lambda c: c.split("_")[0]) if COHORT=="10x" else (lambda c: c.split("-")[0])

# ---- signatures ----
# paper's exact MES + FAK (cptac_mes_confound.py); Neftel 2019 AC/OPC/NPC markers
MES = ["CHI3L1","CD44","VIM","SERPINE1","TGFBI","TIMP1","ANXA1","ANXA2","A2M",
       "LGALS1","LGALS3","CAV1","RELB","NAMPT","ADM","SOCS3","EMP1","CTSB",
       "IGFBP3","LOX","PLAUR","MT2A","S100A4","CLIC1","TNC"]
AC  = ["GFAP","AQP4","S100B","SPARCL1","MLC1","CLU","AGT","CST3","HOPX","GJA1"]
OPC = ["OLIG1","OLIG2","PDGFRA","CSPG4","SOX10","BCAN","APOD","OMG"]
NPC = ["DCX","SOX11","STMN2","DLL3","CD24","SOX4","TUBB3","STMN1","TCF4","DLL1"]
FAK = ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK","PXN","BCAR1"]
TAZ = ["CTGF","CYR61","ANKRD1","AMOTL2","CAV1","THBS1","AXL","TGFB2","F3","CCN1","CCN2"]
# TME (for malignant filtering)
IMM  = ["PTPRC","CD3D","CD3E","CD2","CD14","FCGR3A","AIF1","CSF1R","C1QA","C1QB",
        "C1QC","CX3CR1","P2RY12","TYROBP","ITGAM","CD68","LYZ","CD163"]
OLIG = ["PLP1","MBP","MOG","MAG","CLDN11","CNP"]
ENDO = ["CLDN5","PECAM1","VWF"]
STROMA = ["DCN","COL1A1","COL1A2","LUM","PDGFRB","RGS5","ACTA2","COL3A1"]  # fibroblast/pericyte
STATES = {"MES":MES,"AC":AC,"OPC":OPC,"NPC":NPC}

print(f"=== Neftel mesenchymal deconvolution | cohort={COHORT} ===")
print(f"loading {F.split('/')[-1]} ...")
df = pd.read_csv(F, sep="\t", index_col=0)          # genes x cells
df = df[~df.index.duplicated(keep="first")]
A = ad.AnnData(df.T.astype("float32"))              # cells x genes
del df
A.var_names_make_unique()
A.obs["patient"] = [PAT_SPLIT(c) for c in A.obs_names]
A.layers["raw"] = A.X.copy()
# 10x supp is LINEAR TPM (max ~3.5e4) -> log per Neftel; SS2 supp is ALREADY
# log2(TPM/10+1) (max ~15.5, no negatives) -> use as-is. Verified empirically.
if COHORT == "10x":
    A.X = np.log2(A.X/10.0 + 1.0)
print(f"  value scale: max={float(A.X.max()):.2f} (log2(TPM/10+1) space)")
print(f"  {A.n_obs} cells x {A.n_vars} genes, {A.obs.patient.nunique()} patients")

def score(name, genes):
    g = [x for x in genes if x in A.var_names]
    sc.tl.score_genes(A, g, score_name=name, ctrl_size=max(50,len(g)))
    return g

for nm, gl in {**STATES, "FAK":FAK, "TAZ":TAZ, "IMM":IMM, "OLIG":OLIG,
               "ENDO":ENDO, "STROMA":STROMA}.items():
    used = score(nm, gl)

# ---- malignant filter (signature-based; no CNV). STROMA exclusion guards the
#      key alternative explanation: non-malignant fibroblasts/pericytes are
#      themselves mesenchymal+high-FAK and would inflate the FAK~MES link. ----
tme = A.obs[["IMM","OLIG","ENDO","STROMA"]].max(axis=1)
tum = A.obs[["MES","AC","OPC","NPC"]].max(axis=1)
A.obs["malignant"] = (tum > tme) & (A.obs["IMM"] < 0.10) & (A.obs["STROMA"] < 0.10)
mal = A[A.obs.malignant].copy()
print(f"  malignant cells: {mal.n_obs}/{A.n_obs} "
      f"({100*mal.n_obs/A.n_obs:.0f}%) across {mal.obs.patient.nunique()} patients")
# keep patients with >=20 malignant cells (pseudobulk stability)
keep = mal.obs.patient.value_counts()
keep = keep[keep>=20].index.tolist()
mal = mal[mal.obs.patient.isin(keep)].copy()
print(f"  patients with >=20 malignant cells: {len(keep)}  -> EFFECTIVE INFERENTIAL n")
mal.obs["state"] = mal.obs[["MES","AC","OPC","NPC"]].idxmax(axis=1)
print("  state distribution:", mal.obs.state.value_counts().to_dict())

o = mal.obs
# ---------- (1) per-cell association (DESCRIPTIVE only) ----------
print("\n[1] PER-CELL association (descriptive; pseudoreplicated, NOT inferential):")
for ax in ["FAK","TAZ"]:
    rP,_ = pearsonr(o[ax], o["MES"]); rS,_ = spearmanr(o[ax], o["MES"])
    print(f"   {ax} vs MES: Pearson r={rP:+.3f} (R2={rP**2:.2f}) | Spearman rho={rS:+.3f}")

# ---------- (2) FAK/TAZ by Neftel state ----------
print("\n[2] FAK / TAZ score by assigned Neftel state (cell means):")
print(o.groupby("state")[["FAK","TAZ","MES"]].mean().round(3).to_string())

# ---------- (3) variance decomposition (per-cell R^2) ----------
print("\n[3] variance of FAK/TAZ explained by state scores (per-cell OLS R2):")
for ax in ["FAK","TAZ"]:
    r_full = smf.ols(f"{ax} ~ MES + AC + OPC + NPC", data=o).fit().rsquared
    r_mes  = smf.ols(f"{ax} ~ MES", data=o).fit().rsquared
    print(f"   {ax}: R2(MES only)={r_mes:.3f}  R2(4 states)={r_full:.3f}")

# ---------- (4) PSEUDOBULK per patient x state (honest n) ----------
print("\n[4] PSEUDOBULK (patient x state means) -- this respects pseudoreplication:")
pb = o.groupby(["patient","state"])[["FAK","TAZ","MES","AC","OPC","NPC"]].mean().reset_index()
pb["n_cells"] = o.groupby(["patient","state"]).size().values
print(f"   {len(pb)} patient x state units from {o.patient.nunique()} patients")
# patient-level: mean FAK vs MES-fraction and vs mean MES score
pat = o.groupby("patient").agg(meanFAK=("FAK","mean"), meanTAZ=("TAZ","mean"),
                               meanMES=("MES","mean"),
                               mesFrac=("state", lambda s:(s=="MES").mean())).reset_index()
for y,x in [("meanFAK","meanMES"),("meanFAK","mesFrac"),("meanTAZ","meanMES")]:
    if pat[x].nunique()>2:
        r,p = pearsonr(pat[y],pat[x])
        print(f"   patient-level {y} ~ {x}: r={r:+.3f} p={p:.3g} (n={len(pat)} patients)")

# ---------- (5) MIXED MODEL: FAK ~ MES + (1|patient) ----------
print("\n[5] MIXED MODEL  FAK ~ MES + (1|patient)  (cell-level, patient random effect):")
for ax in ["FAK","TAZ"]:
    try:
        m = smf.mixedlm(f"{ax} ~ MES", o, groups=o["patient"]).fit(reml=True)
        b = m.params.get("MES", np.nan); p = m.pvalues.get("MES", np.nan)
        # variance partition
        ve = m.scale                                   # residual var
        vg = float(m.cov_re.iloc[0,0]) if m.cov_re.size else 0.0
        fitted = m.fittedvalues
        r2m = 1 - np.var(o[ax]-fitted)/np.var(o[ax])    # approx conditional R2
        print(f"   {ax}: MES beta={b:+.3f} p={p:.3g} | var(patient RE)={vg:.4f} "
              f"resid var={ve:.4f} | approx R2={r2m:.3f}")
    except Exception as e:
        print(f"   {ax}: mixedlm failed ({e})")

# ---------- (6) residual FAK beyond MES: does it still track state? ----------
print("\n[6] DECONFOUNDING TEST -- FAK residual after removing MES, by state:")
o2 = o.copy()
o2["FAK_resid"] = smf.ols("FAK ~ MES", data=o2).fit().resid
res_state = o2.groupby("state")["FAK_resid"].mean().round(3)
print("   mean residual FAK by state (≈0 everywhere => FAK is pure MES readout):")
print("   ", res_state.to_dict())
# patient-level ANOVA-style: does residual FAK differ by state beyond noise?
pbr = o2.groupby(["patient","state"])["FAK_resid"].mean().reset_index()
try:
    aov = smf.ols("FAK_resid ~ C(state)", data=pbr).fit()
    print(f"   patient-pseudobulk residual-FAK ~ state: F-p={aov.f_pvalue:.3g} R2={aov.rsquared:.3f}")
except Exception as e:
    print(f"   residual ANOVA failed ({e})")

# save
out = f"/home/brad/finngen-triage/neftel_deconfound_{COHORT}.csv"
pb.to_csv(out, index=False)
pat.to_csv(out.replace(".csv","_patient.csv"), index=False)
print(f"\nwrote {out} and *_patient.csv")
print("=== done ===")
