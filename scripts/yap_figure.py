#!/usr/bin/env python3
"""
Supplementary Figure S2: the transcriptional OUTPUT of the adhesion/FAK axis --
TAZ/TEAD1 is a co-essential, MTAP-separable, druggable node; honest about the
flat tumor output signature.

  A  Genome-wide CRISPR: YAP/TAZ-TEAD effectors (TAZ & TEAD1 essential;
     YAP1/TEAD2-4 buffered by paralog redundancy).
  B  Hippo NEGATIVE regulators are NOT essential / favour growth when knocked
     out -- the orientation control for a YAP/TAZ-dependent state.
  C  MTAP separation: TAZ dependency persists in MTAP-intact null lines.
  D  CPTAC tumors: TAZ protein selectively up (mes-independent), but the YAP
     transcriptional OUTPUT signature is flat (mesenchymal-confounded).
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.titleweight":"bold"})
RED, BLUE, GREEN, GREY = "#c0392b", "#2c6fbb", "#27865c", "#9aa0a6"

def rd(f): return pd.read_csv(f, keep_default_na=False, na_values=[""])
goi = rd("yap_genomewide_goi.csv").set_index("gene")
perline = rd("yap_module_perline.csv")
tum = rd("yap_tumor_signature_values.csv")

fig = plt.figure(figsize=(12.6, 7.4))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30)

# ---------- A: YAP/TAZ-TEAD effector forest ----------
axA = fig.add_subplot(gs[0,0])
order = ["WWTR1","TEAD1","YAP1","TEAD3","TEAD4","TEAD2"]
lbl = {"WWTR1":"WWTR1/TAZ"}
d = goi.loc[order,"delta"]; q = goi.loc[order,"q"]; rk = goi.loc[order,"rank"]
y = np.arange(len(order))[::-1]
cols = [RED if qq<0.05 else GREY for qq in q]
axA.barh(y, d.values, color=cols)
for yi,g in zip(y,order):
    txt = f"q={q[g]:.0e}  (#{int(rk[g])})" if q[g]<0.05 else f"q={q[g]:.2f}"
    axA.text(d[g]-0.004, yi, txt, va="center", ha="right", fontsize=6.6)
axA.axvline(0,color="k",lw=0.6)
axA.set_yticks(y); axA.set_yticklabels([lbl.get(g,g) for g in order])
axA.set_xlabel(r"$\Delta$ CRISPR effect (null $-$ intact)")
axA.set_xlim(-0.30, 0.04)
axA.set_title("A  YAP/TAZ-TEAD effectors (CNS, genome-wide)\nTAZ & TEAD1 essential; YAP1/TEAD2-4 buffered",loc="left")

# ---------- B: Hippo negative-regulator orientation ----------
axB = fig.add_subplot(gs[0,1])
negs = ["NF2","LATS2","SAV1","MOB1B","LATS1","MOB1A","STK4","STK3"]
db = goi.loc[negs,"delta"]; rkb = goi.loc[negs,"rank"]
y = np.arange(len(negs))[::-1]
# essential-favouring (delta<0 & q<0.05) would be red; here none -> grey, growth-favouring green
colsB = [RED if (goi.loc[g,"q"]<0.05 and goi.loc[g,"delta"]<0) else (GREEN if goi.loc[g,"delta"]>0 else GREY) for g in negs]
axB.barh(y, db.values, color=colsB)
for yi,g in zip(y,negs):
    axB.text(db[g]+ (0.004 if db[g]>=0 else -0.004), yi, f"#{int(rkb[g])}",
             va="center", ha="left" if db[g]>=0 else "right", fontsize=6.6)
axB.axvline(0,color="k",lw=0.6)
axB.set_yticks(y); axB.set_yticklabels(negs)
axB.set_xlabel(r"$\Delta$ CRISPR effect (null $-$ intact)")
axB.set_title("B  Hippo NEGATIVE regulators (YAP repressors)\nnot essential / favour growth = orientation control",loc="left")
axB.text(0.97,0.04,"green = KO favours growth\n(rank /17916)",transform=axB.transAxes,
         ha="right",va="bottom",fontsize=6.3,color=GREEN)

# ---------- C: MTAP separation (pan-cancer), TAZ ----------
axC = fig.add_subplot(gs[1,0])
B = perline[(perline.cdkn_null==1)&(perline.mtap_null==0)]["WWTR1"].dropna()
C = perline[(perline.cdkn_null==0)]["WWTR1"].dropna()
_,pC = mannwhitneyu(B, C, alternative="less")
bp = axC.boxplot([C, B], labels=[f"intact\n(n={len(C)})", f"MTAP-intact\nnull (n={len(B)})"],
                 patch_artist=True, widths=0.6, showfliers=False)
for patch,c in zip(bp['boxes'],[GREY,RED]): patch.set_facecolor(c); patch.set_alpha(.75)
axC.set_ylabel("WWTR1/TAZ CRISPR effect")
axC.set_title(f"C  TAZ dependency survives 9p21/MTAP separation\nMTAP-intact null vs intact, pan-cancer p={pC:.3f}",loc="left")

# ---------- D: CPTAC tumor: TAZ protein up, output signature flat ----------
axD = fig.add_subplot(gs[1,1])
def grp(col):
    return (tum[tum.status=="intact"][col].dropna(), tum[tum.status=="null"][col].dropna())
# mes-adjusted p for TAZ protein
sub = tum[["status","WWTR1_protein","mes_sig_rna"]].dropna().copy()
sub["cdkn_null"]=(sub.status=="null").astype(int)
m = sm.OLS(sub.WWTR1_protein, sm.add_constant(sub[["cdkn_null","mes_sig_rna"]])).fit()
p_taz_adj = m.pvalues["cdkn_null"]
i_taz,n_taz = grp("WWTR1_protein"); i_sig,n_sig = grp("yap_sig_rna")
_,p_sig = mannwhitneyu(n_sig, i_sig, alternative="greater")
data=[i_taz,n_taz,i_sig,n_sig]
labels=["intact","null","intact","null"]
bp = axD.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
for patch,c in zip(bp['boxes'],[GREY,RED,GREY,GREY]): patch.set_facecolor(c); patch.set_alpha(.75)
axD.axhline(0,color="k",lw=0.5,ls=":")
axD.set_ylabel("abundance / score (z)")
axD.set_xticks([1.5,3.5]); axD.set_xticklabels([f"TAZ protein\n(mes-adj p={p_taz_adj:.3f})",
                                                f"YAP output sig (RNA)\n(p={p_sig:.2f}, ns)"])
for x in [1,2,3,4]: axD.text(x, axD.get_ylim()[0], "", ha="center")
axD.axvline(2.5,color="k",lw=0.5,ls="--",alpha=0.4)
axD.set_title("D  Tumor (CPTAC): TAZ protein up (mes-independent),\nYAP transcriptional OUTPUT flat (mes-confounded)",loc="left")

fig.suptitle("Supplementary Figure S2 | TAZ/TEAD1: the co-essential, MTAP-separable, druggable OUTPUT node of the adhesion axis",
             fontsize=11.5, fontweight="bold", y=0.995)
fig.savefig("yap_tead_figS2.pdf", bbox_inches="tight")
fig.savefig("yap_tead_figS2.png", dpi=200, bbox_inches="tight")
print(f"wrote yap_tead_figS2.pdf/.png  | panelC p={pC:.4f}  TAZ-prot mes-adj p={p_taz_adj:.4f}  output-sig p={p_sig:.3f}")
