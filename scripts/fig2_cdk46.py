#!/usr/bin/env python3
"""Figure 2: the CDK4/6-inhibitor-eligible GBM subset — no survival benefit, and why (resistance map).

  Panel A: median OS, eligible vs other IDH-wt, in TCGA and MSK-IMPACT (both null).
  Panel B: CDK4/6i resistance lesions present in the eligible subset (% of eligible tumors),
           grouped by escape pathway — the novel trial-design element (~44% PTEN/PI3K).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HOME = Path.home() / "finngen-triage"
OUT_PNG = HOME / "fig2_cdk46.png"
OUT_PDF = HOME / "fig2_cdk46.pdf"

# Survival (median OS months) — eligible vs other IDH-wt
SURV = [
    # cohort, elig_os, other_os, logrank_p
    ("TCGA-GBM\n(n=244 elig)", 12.9, 15.3, 0.12),
    ("MSK-IMPACT\n(n=266 elig)", 21.9, 23.6, 0.84),
]

# Resistance pathway grouping/colors
PATHWAY = {
    "PTEN mut":               ("PI3K/PTEN", "#b21f2d"),
    "PTEN homdel":            ("PI3K/PTEN", "#b21f2d"),
    "PIK3R1 mut":             ("PI3K/PTEN", "#b21f2d"),
    "PIK3CA mut":             ("PI3K/PTEN", "#b21f2d"),
    "NF1 mut":                ("RAS/MAPK", "#e08214"),
    "KRAS mut":               ("RAS/MAPK", "#e08214"),
    "AKT1 mut":               ("PI3K/PTEN", "#b21f2d"),
    "MYC amp":                ("E2F-independent", "#7d3c98"),
    "CCNE1 amp":              ("CDK2 bypass", "#1f6f8b"),
    "YAP1 amp":               ("Hippo bypass", "#7d7d7d"),
    "FGFR1 amp":              ("RTK bypass", "#117a65"),
}
PATH_COLORS = {
    "PI3K/PTEN": "#b21f2d", "RAS/MAPK": "#e08214", "CDK2 bypass": "#1f6f8b",
    "E2F-independent": "#7d3c98", "Hippo bypass": "#7d7d7d", "RTK bypass": "#117a65",
}


def panel_a(ax):
    x = np.arange(len(SURV))
    w = 0.38
    elig = [s[1] for s in SURV]
    other = [s[2] for s in SURV]
    ax.bar(x - w/2, elig, w, label="CDK4/6i-eligible", color="#2471a3")
    ax.bar(x + w/2, other, w, label="other IDH-wt", color="#aeb6bf")
    for xi, s in zip(x, SURV):
        top = max(s[1], s[2])
        ax.text(xi, top + 0.8, f"log-rank p={s[3]:.2f}\n(n.s.)", ha="center",
                va="bottom", fontsize=7.5, color="#34495e")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SURV], fontsize=8.5)
    ax.set_ylabel("Median overall survival (months)", fontsize=9)
    ax.set_ylim(0, 30)
    ax.set_title("A. CDK4/6i eligibility confers NO survival benefit\n(eligible vs other IDH-wt; two cohorts)",
                 fontsize=9.5, loc="left", color="#2c3e50", pad=8)
    ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#bdc3c7")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_b(ax):
    df = pd.read_csv(HOME / "tcga_gbm_resistance.csv")
    d = {r["event"]: r for _, r in df.iterrows()}
    rows = [
        ("PTEN mut", d["PTEN_mut"]["pct_eligible"]),
        ("PTEN homdel", d["PTEN_homdel"]["pct_eligible"]),
        ("PIK3R1 mut", d["PIK3R1_mut"]["pct_eligible"]),
        ("PIK3CA mut", d["PIK3CA_mut"]["pct_eligible"]),
        ("NF1 mut", d["NF1_mut"]["pct_eligible"]),
        ("MYC amp", d["MYC_amp"]["pct_eligible"]),
        ("KRAS mut", d["KRAS_mut"]["pct_eligible"]),
        ("AKT1 mut", d["AKT1_mut"]["pct_eligible"]),
        ("CCNE1 amp", d["CCNE1_amp"]["pct_eligible"]),
        ("YAP1 amp", d["YAP1_amp"]["pct_eligible"]),
        ("FGFR1 amp", d["FGFR1_amp"]["pct_eligible"]),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [PATHWAY[l][1] for l in labels]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.68)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.1f}%", va="center", ha="left", fontsize=7.6, color="#2c3e50")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("% of CDK4/6i-eligible tumors carrying lesion", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("B. CDK4/6i-resistance lesions within the eligible subset\nPI3K/PTEN-axis escape dominates (red)",
                 fontsize=9.5, loc="left", color="#2c3e50", pad=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    seen, handles = set(), []
    for l in labels:
        p = PATHWAY[l][0]
        if p not in seen:
            seen.add(p)
            handles.append(mpatches.Patch(color=PATH_COLORS[p], label=p))
    ax.legend(handles=handles, fontsize=7.0, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="#bdc3c7")


def main():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.32})
    fig.patch.set_facecolor("white")
    for ax in (axa, axb):
        ax.set_facecolor("white")
    panel_a(axa)
    panel_b(axb)
    fig.suptitle("Figure 2. The CDK4/6-inhibitor-eligible IDH-wildtype GBM subset: large and definable, "
                 "but no survival benefit — explained by PI3K/PTEN-axis resistance",
                 fontsize=11, fontweight="bold", color="#2c3e50", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.005,
             "Eligible = IDH-wt, RB1-intact, CDK4/6-axis-activated (CDKN2A/B-deleted and/or CDK4/CCND-amplified). PTEN shown as "
             "mutation (34%) and homozygous deletion (12%) separately (overlap unquantified). A CDK4/6i trial in GBM should exclude or co-target the PI3K/PTEN axis.",
             fontsize=7.2, color="#7f8c8d", va="bottom")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}\nSaved: {OUT_PDF}")
    plt.close()


if __name__ == "__main__":
    main()
