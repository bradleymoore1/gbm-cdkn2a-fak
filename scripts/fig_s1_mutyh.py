#!/usr/bin/env python3
"""Supplementary Figure S1 for the combined GBM preprint.

Phenome-wide association forest plot for MUTYH p.Gly155Asp (rs587781864) in FinnGen R13.
Same data/layout as the standalone MUTYH figure, retitled as Supplementary Figure S1.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path.home() / "finngen-triage" / "GBM_preprint_figS1_mutyh.png"
OUT_PDF = Path.home() / "finngen-triage" / "GBM_preprint_figS1_mutyh.pdf"

# (label, OR, CI_lo, CI_hi, mlogp, group)
DATA = [
    ("GBM",                              24.9,  4.3, 145.8, 3.45, "glioma"),
    ("GBM + astrocytoma",                21.2,  3.7, 121.7, 3.22, "glioma"),
    ("Brain cancer (broad)",             31.7,  2.2, 447.1, 1.98, "glioma"),
    ("Digestive organ cancers",           3.4,  1.8,   6.3, 3.93, "map"),
    ("Rectal adenocarcinoma (mucinous)",  6.0,  2.2,  16.2, 3.39, "map"),
    ("Rectal adenocarcinoma",             5.5,  2.0,  15.5, 2.94, "map"),
    ("Colorectal cancer",                 3.3,  1.5,   7.3, 2.48, "map"),
    ("Colorectal adenocarcinoma",         3.2,  1.4,   7.6, 2.11, "map"),
    ("Any solid cancer",                  1.7,  1.2,   2.5, 2.41, "broad"),
    ("Any cancer",                        1.7,  1.2,   2.4, 2.24, "broad"),
]
COLORS = {"glioma": "#c0392b", "map": "#2471a3", "broad": "#717d7e"}
LABELS = {"glioma": "Glioma / CNS", "map": "MAP-expected GI/CRC", "broad": "Broad cancer"}


def make():
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    n = len(DATA)
    ypos, prev, gap = [], None, 0
    for i, (_, _, _, _, _, g) in enumerate(DATA):
        if prev is not None and g != prev:
            gap += 0.6
        ypos.append(n - i - gap); prev = g
    for i, (label, OR, lo, hi, mlogp, g) in enumerate(DATA):
        y = ypos[i]; c = COLORS[g]
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.scatter([OR], [y], color=c, s=55, zorder=3, edgecolors="white", linewidths=0.5)
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=9,
                transform=ax.get_yaxis_transform(), color="#2c3e50")
        ax.text(1.02, y, f"{OR:.1f} ({lo:.1f}–{hi:.0f})", ha="left", va="center",
                fontsize=8, transform=ax.get_yaxis_transform(), color=c, fontweight="bold")
        s = f"{mlogp:.2f}" + (" ✦" if mlogp >= 3.45 else "")
        ax.text(1.32, y, s, ha="left", va="center", fontsize=8,
                transform=ax.get_yaxis_transform(), color="#2c3e50")
    ax.axvline(1.0, color="#95a5a6", lw=1.0, linestyle="--", zorder=1)
    ax.text(1.02, n + 0.3, "OR (95% CI)", ha="left", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform(), color="#2c3e50", style="italic")
    ax.text(1.32, n + 0.3, "−log₁₀P", ha="left", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform(), color="#2c3e50", style="italic")
    gpos = {}
    for i, (_, _, _, _, _, g) in enumerate(DATA):
        gpos.setdefault(g, []).append(ypos[i])
    for g, yl in gpos.items():
        ymid = (min(yl) + max(yl)) / 2
        ax.annotate("", xy=(-0.27, min(yl) - 0.25), xytext=(-0.27, max(yl) + 0.25),
                    xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=COLORS[g], lw=1.5))
        ax.text(-0.29, ymid, LABELS[g], ha="right", va="center", fontsize=8.0,
                transform=ax.get_yaxis_transform(), color=COLORS[g], fontweight="bold", rotation=90)
    ax.set_xscale("log"); ax.set_xlim(0.7, 600)
    ax.set_xticks([1, 2, 5, 10, 25, 100, 500])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "100", "500"], fontsize=9)
    ax.set_xlabel("Odds ratio (95% CI)", fontsize=10, labelpad=8)
    ax.set_yticks([]); ax.set_ylim(min(ypos) - 0.8, max(ypos) + 1.0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for y in ypos:
        ax.axhline(y, color="#ecf0f1", lw=0.5, zorder=0)
    ax.set_title("Supplementary Figure S1. FinnGen R13 phenome-wide associations of\n"
                 "MUTYH p.Gly155Asp (rs587781864) across cancer phenotypes",
                 fontsize=10.5, pad=14, color="#2c3e50", loc="left")
    ax.legend(handles=[mpatches.Patch(color=COLORS[g], label=LABELS[g])
                       for g in ("glioma", "map", "broad")],
              fontsize=8.5, loc="lower right", frameon=True, framealpha=0.9, edgecolor="#bdc3c7")
    fig.text(0.05, 0.01,
             "✦ −log₁₀P ≥ 3.45 (C3_GBM primary signal). FinnGen R13: N=467 GBM cases, 372,626 total. "
             "Does not pass genome-wide Bonferroni (>4.98); replication-grade lead.\n"
             "MUTYH p.Gly155Asp: ClinVar Pathogenic/Likely-pathogenic for MAP (VCV000141595); "
             "gnomAD FIN enrichment 7.3× vs NFE; monoallelic carriers.",
             fontsize=7.5, color="#7f8c8d", va="bottom")
    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}\nSaved: {OUT_PDF}")
    plt.close()


if __name__ == "__main__":
    make()
