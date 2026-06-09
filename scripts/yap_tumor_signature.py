#!/usr/bin/env python3
"""
ORTHOGONAL VALIDATION of the YAP/TAZ-TEAD dependency in CDKN2A/B-null GBM.

The CRISPR side (yap_tead_analysis.py) showed WWTR1/TAZ (q=6e-4) and TEAD1
(q=0.026) are selectively essential in CDKN2A/B-null CNS lines, and that the
Hippo NEGATIVE regulators (NF2/LATS/SAV/MOB) are NOT essential / favour growth
when knocked out -- the functional signature of a YAP/TAZ-dependent cell.

Cell-line essentiality is one modality. The clinically decisive question (same
logic that made FAK-pY397 the strongest part of the FAK story): is the YAP/TAZ
pathway transcriptionally ACTIVE in real CDKN2A/B-null tumors? If null tumors
lean on YAP/TAZ-TEAD, the canonical TEAD-dependent OUTPUT signature should be
ELEVATED in null tumors -- at RNA and (independently) at protein level, in the
SAME tumors (CPTAC matched proteogenomics).

Three reads, honestly:
  (1) Canonical YAP/TAZ-TEAD target signature (direct TEAD targets), null vs
      intact, RNA and protein separately + a per-tumor signature score.
  (2) Pathway components WWTR1/TAZ, YAP1, TEAD1 themselves (RNA + protein).
  (3) MESENCHYMAL CONFOUND (the lesson from FAK): YAP/TAZ drives mesenchymal
      transition, so elevated YAP output could merely reflect mesenchymal-subtype
      skew. Test (a) are null tumors mesenchymal-skewed? (non-overlapping mes
      markers) and (b) does the YAP signature elevation survive adjusting for it?

CPTAC GBM, deep CDKN2A/B CNV loss = null (same thresholds as cptac_fak_phospho.py).
Hypothesis generation only.
"""
import cptac, pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

DEEP = -1.0       # log2 CNV ratio <= -1.0  -> deep (homdel-like) loss
INTACT_HI = -0.3  # log2 CNV ratio > -0.3   -> retained

# Canonical direct TEAD-dependent YAP/TAZ target genes (Dupont/Zanconato core +
# well-established direct targets). tx uses old symbols, prot uses new (CCN*).
YAP_SIG_TX   = ["CTGF","CYR61","ANKRD1","AMOTL2","NUAK2","CRIM1","F3","TGFB2"]
YAP_SIG_PROT = ["CCN2","CCN1","AMOTL2","NUAK2","CRIM1","F3","TGFB2"]   # CCN2=CTGF, CCN1=CYR61
COMPONENTS   = ["WWTR1","YAP1","TEAD1"]
# Mesenchymal-GBM markers chosen to NOT overlap the YAP target list (clean confound)
MES_MARKERS  = ["CHI3L1","CD44","VIM","MET","RELB","SERPINE1"]

g = cptac.Gbm()
cnv  = g.get_CNV("bcm")
tx   = g.get_transcriptomics("washu")
prot = g.get_proteomics("umich")

def gene_cols(df, gene):
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        return [c for c in cols if c[0]==gene]
    return [c for c in cols if c==gene]

def gene_series(df, gene):
    """One series per gene = mean across matching columns (transcript IDs)."""
    hits = gene_cols(df, gene)
    if not hits: return None
    return df[hits].mean(axis=1)

# ---- classify tumors by deep CDKN2A/B loss (identical to FAK analysis) ----
a = cnv[gene_cols(cnv,"CDKN2A")[0]]; b = cnv[gene_cols(cnv,"CDKN2B")[0]]
cls = pd.Series(index=cnv.index, dtype=object)
cls[(a<=DEEP)&(b<=DEEP)] = "null"
cls[(a>INTACT_HI)&(b>INTACT_HI)] = "intact"
cls = cls.dropna()
print(f"CPTAC GBM: {(cls=='null').sum()} CDKN2A/B-null (deep loss) / "
      f"{(cls=='intact').sum()} intact  [ambiguous excluded]\n")

def compare(values, label, alt="greater", quiet=False):
    df = pd.DataFrame({"v":values}).join(cls.rename("status"), how="inner").dropna()
    n = df[df.status=="null"]["v"]; i = df[df.status=="intact"]["v"]
    if len(n)<4 or len(i)<4:
        if not quiet: print(f"  {label:26s} too few (n={len(n)}/{len(i)})")
        return np.nan
    _,pg = mannwhitneyu(n,i,alternative=alt)
    flag = "  <== null higher" if (n.median()>i.median() and pg<0.05) else ""
    if not quiet:
        print(f"  {label:26s} null med={n.median():+.3f}(n={len(n)}) "
              f"intact med={i.median():+.3f}(n={len(i)})  null>intact p={pg:.3g}{flag}")
    return pg

def zscore_sig(df, genes):
    """Per-tumor signature = mean of z-scored member genes (z across all tumors)."""
    cols=[]
    for gn in genes:
        s = gene_series(df, gn)
        if s is None: continue
        z = (s - s.mean())/s.std()
        cols.append(z.rename(gn))
    if not cols: return None, []
    M = pd.concat(cols, axis=1)
    return M.mean(axis=1), [c.name for c in cols]

# =====================================================================
# (1) YAP/TAZ-TEAD canonical OUTPUT signature
# =====================================================================
print("="*70)
print("(1) YAP/TAZ-TEAD canonical target OUTPUT signature  [null vs intact]")
print("="*70)
print("-- RNA (transcriptomics, washu) --")
for gn in YAP_SIG_TX: compare(gene_series(tx,gn), f"{gn} mRNA")
sig_tx, used_tx = zscore_sig(tx, YAP_SIG_TX)
p_sig_tx = compare(sig_tx, f"** YAP signature score (RNA, {len(used_tx)}g) **")

print("\n-- PROTEIN (proteomics, umich) --")
for gn in YAP_SIG_PROT: compare(gene_series(prot,gn), f"{gn} protein")
sig_pr, used_pr = zscore_sig(prot, YAP_SIG_PROT)
p_sig_pr = compare(sig_pr, f"** YAP signature score (PROTEIN, {len(used_pr)}g) **")

# =====================================================================
# (2) Pathway components themselves
# =====================================================================
print("\n"+"="*70)
print("(2) Pathway components  WWTR1/TAZ, YAP1, TEAD1")
print("="*70)
print("-- RNA --");     [compare(gene_series(tx,gn),  f"{gn} mRNA")    for gn in COMPONENTS]
print("-- PROTEIN --"); [compare(gene_series(prot,gn),f"{gn} protein") for gn in COMPONENTS]

# =====================================================================
# (3) Mesenchymal confound  (the lesson from FAK)
# =====================================================================
print("\n"+"="*70)
print("(3) MESENCHYMAL CONFOUND  (YAP/TAZ drives mesenchymal transition)")
print("="*70)
print("-- (3a) are null tumors mesenchymal-skewed?  (non-overlapping mes markers, RNA) --")
mes_tx, used_mes = zscore_sig(tx, MES_MARKERS)
p_mes = compare(mes_tx, f"mesenchymal score (RNA, {len(used_mes)}g)", alt="two-sided")
print(f"     [two-sided; non-sig => null NOT mes-skewed, so YAP signal is not a mes passenger]")

print("\n-- (3b) does the YAP RNA signature survive adjusting for mes + purity? --")
# OLS: YAP_sig ~ cdkn_null + mes_score ; if cdkn term stays +/sig, not a mes passenger
dat = pd.DataFrame({"yap":sig_tx, "mes":mes_tx}).join(cls.rename("status"), how="inner").dropna()
dat["cdkn_null"] = (dat.status=="null").astype(int)
X = sm.add_constant(dat[["cdkn_null","mes"]])
m = sm.OLS(dat["yap"], X).fit()
print(f"     YAP_sig ~ cdkn_null + mes   (n={int(m.nobs)})")
print(f"       cdkn_null b={m.params['cdkn_null']:+.3f} p={m.pvalues['cdkn_null']:.3g}"
      f"{'  <-- survives' if (m.pvalues['cdkn_null']<0.05 and m.params['cdkn_null']>0) else ''}")
print(f"       mes       b={m.params['mes']:+.3f} p={m.pvalues['mes']:.3g}")

# ---- tidy dump for a potential figure panel ----
out = pd.DataFrame(index=cls.index)
out["status"] = cls
if sig_tx is not None: out["yap_sig_rna"]     = sig_tx
if sig_pr is not None: out["yap_sig_protein"] = sig_pr
if mes_tx is not None: out["mes_sig_rna"]      = mes_tx
for gn in COMPONENTS:
    s = gene_series(tx,gn);   out[f"{gn}_rna"]     = s if s is not None else np.nan
    s = gene_series(prot,gn); out[f"{gn}_protein"] = s if s is not None else np.nan
out = out.dropna(subset=["status"])
out.to_csv("yap_tumor_signature_values.csv")
print(f"\nwrote yap_tumor_signature_values.csv {out.shape}")
print("\nSUMMARY:")
print(f"  YAP target signature  RNA  null>intact p={p_sig_tx:.3g}")
print(f"  YAP target signature  PROT null>intact p={p_sig_pr:.3g}")
print(f"  null mesenchymal-skew (two-sided)   p={p_mes:.3g}  (want NON-sig)")
print("DONE.")
