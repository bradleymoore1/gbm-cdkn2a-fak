#!/usr/bin/env python3
"""Figure: single-cell FAK<->mesenchymal coupling in GBM (Neftel 2019).
Reads the stroma-excluded pseudobulk CSVs written by neftel_mes_deconfound.py."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

D="/home/brad/finngen-triage"
pat={c:pd.read_csv(f"{D}/neftel_deconfound_{c}_patient.csv") for c in ["ss2","10x"]}
pb ={c:pd.read_csv(f"{D}/neftel_deconfound_{c}.csv")          for c in ["ss2","10x"]}
order=["MES","AC","OPC","NPC"]; col={"ss2":"#1f77b4","10x":"#ff7f0e"}

fig,ax=plt.subplots(1,3,figsize=(13,4.2))

# A: patient-level meanFAK vs meanMES
for c in ["ss2","10x"]:
    d=pat[c]; ax[0].scatter(d.meanMES,d.meanFAK,s=42,c=col[c],alpha=.8,
        edgecolor="k",lw=.4,label=f"{'Smart-seq2' if c=='ss2' else '10x'} (n={len(d)})")
d=pat["ss2"]; r,p=pearsonr(d.meanMES,d.meanFAK)
m,b=np.polyfit(d.meanMES,d.meanFAK,1); xs=np.linspace(d.meanMES.min(),d.meanMES.max(),50)
ax[0].plot(xs,m*xs+b,"-",c=col["ss2"],lw=2)
ax[0].set_xlabel("patient mean MES score"); ax[0].set_ylabel("patient mean FAK-module score")
ax[0].set_title(f"A. Tumor-level FAK $\\propto$ mesenchymal\nSmart-seq2 r={r:.2f}, p={p:.1e}")
ax[0].legend(fontsize=8,loc="upper left")

# B: FAK by Neftel state (patient x state pseudobulk, both cohorts pooled)
allpb=pd.concat([pb[c].assign(cohort=c) for c in pb])
data=[allpb[allpb.state==s].FAK.values for s in order]
bp=ax[1].boxplot(data,labels=order,patch_artist=True,showfliers=False,widths=.6)
for patch in bp["boxes"]: patch.set_facecolor("#cfe8ff")
for s_i,s in enumerate(order):
    sub=allpb[allpb.state==s]
    ax[1].scatter(np.random.normal(s_i+1,.06,len(sub)),sub.FAK,s=10,
                  c=[col[cc] for cc in sub.cohort],alpha=.6)
ax[1].axhline(0,ls=":",c="grey",lw=.8)
ax[1].set_ylabel("FAK-module score (patient$\\times$state mean)")
ax[1].set_title("B. FAK high in MES/AC, low in NPC/OPC\n(no residual after MES adjust)")

# C: per-cell vs tumor-level variance explained
cats=["per-cell\nR²(FAK~MES)","tumor-level\nr²(meanFAK~meanMES)"]
ss2_vals=[0.128, 0.809**2]; tenx_vals=[0.099, 0.506**2]
x=np.arange(2); w=.35
ax[2].bar(x-w/2,ss2_vals,w,color=col["ss2"],label="Smart-seq2")
ax[2].bar(x+w/2,tenx_vals,w,color=col["10x"],label="10x")
ax[2].set_xticks(x); ax[2].set_xticklabels(cats,fontsize=9)
ax[2].set_ylabel("variance explained"); ax[2].set_ylim(0,.75)
ax[2].set_title("C. Coupling is a tumor-level property\n(weak within single cells)")
ax[2].legend(fontsize=8)
for i,v in enumerate(ss2_vals): ax[2].text(i-w/2,v+.01,f"{v:.2f}",ha="center",fontsize=8)
for i,v in enumerate(tenx_vals): ax[2].text(i+w/2,v+.01,f"{v:.2f}",ha="center",fontsize=8)

plt.tight_layout()
out=f"{D}/neftel_mes_deconfound.png"; plt.savefig(out,dpi=160,bbox_inches="tight")
print("wrote",out)
