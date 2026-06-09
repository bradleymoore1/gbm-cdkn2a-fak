#!/usr/bin/env python3
"""Generate publication-quality figure for MUTYH p.Gly155Asp GBM paper.

Figure 1: Phenome-wide association forest plot for rs587781864 in FinnGen R13.
  - Three groups: Glioma/CNS, MAP-expected GI/CRC, Broad cancer
  - OR + 95% CI on log scale
  - mlogp annotated on right
  - Color-coded by group
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path.home() / "finngen-triage" / "mutyh_gbm_figure1.png"
OUT_PDF = Path.home() / "finngen-triage" / "mutyh_gbm_figure1.pdf"

# Data: (label, OR, CI_lo, CI_hi, mlogp, group)
# Computed from FinnGen R13 MUTYH.Mask1.0.01 burden stats
DATA = [
    # Glioma / CNS
    ("GBM",                          24.9,   4.3,   145.8,  3.45, "glioma"),
    ("GBM + astrocytoma",            21.2,   3.7,   121.7,  3.22, "glioma"),
    ("Brain cancer (broad)",         31.7,   2.2,   447.1,  1.98, "glioma"),
    # MAP-expected GI/CRC
    ("Digestive organ cancers",       3.4,   1.8,     6.3,  3.93, "map"),
    ("Rectal adenocarcinoma (mucinous)", 6.0, 2.2,  16.2,  3.39, "map"),
    ("Rectal adenocarcinoma",         5.5,   2.0,    15.5,  2.94, "map"),
    ("Colorectal cancer",             3.3,   1.5,     7.3,  2.48, "map"),
    ("Colorectal adenocarcinoma",     3.2,   1.4,     7.6,  2.11, "map"),
    # Broad / pan-cancer
    ("Any solid cancer",              1.7,   1.2,     2.5,  2.41, "broad"),
    ("Any cancer",                    1.7,   1.2,     2.4,  2.24, "broad"),
]

COLORS = {
    "glioma": "#c0392b",   # red
    "map":    "#2471a3",   # blue
    "broad":  "#717d7e",   # gray
}
LABELS = {
    "glioma": "Glioma / CNS",
    "map":    "MAP-expected GI/CRC",
    "broad":  "Broad cancer",
}

def make_figure():
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    n = len(DATA)
    # y positions: top to bottom, with gaps between groups
    y_positions = []
    group_order = []
    prev_group = None
    gap = 0
    for i, (label, OR, lo, hi, mlogp, group) in enumerate(DATA):
        if prev_group is not None and group != prev_group:
            gap += 0.6
        y_positions.append(n - i - gap)
        group_order.append(group)
        prev_group = group

    for i, (label, OR, lo, hi, mlogp, group) in enumerate(DATA):
        y = y_positions[i]
        color = COLORS[group]

        # CI line
        ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="round", zorder=2)
        # OR point
        ax.scatter([OR], [y], color=color, s=55, zorder=3, edgecolors="white", linewidths=0.5)

        # Phenotype label (left)
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=9,
                transform=ax.get_yaxis_transform(), color="#2c3e50")

        # OR text (right of CI)
        or_text = f"{OR:.1f} ({lo:.1f}–{hi:.0f})"
        ax.text(1.02, y, or_text, ha="left", va="center", fontsize=8,
                transform=ax.get_yaxis_transform(), color=color, fontweight="bold")

        # mlogp annotation (far right)
        sig_str = f"{mlogp:.2f}"
        if mlogp >= 3.45:
            sig_str += " ✦"
        ax.text(1.32, y, sig_str, ha="left", va="center", fontsize=8,
                transform=ax.get_yaxis_transform(), color="#2c3e50")

    # Vertical line at OR=1
    ax.axvline(x=1.0, color="#95a5a6", lw=1.0, linestyle="--", zorder=1)

    # Header text
    ax.text(1.02, n + 0.3, "OR (95% CI)", ha="left", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform(), color="#2c3e50", style="italic")
    ax.text(1.32, n + 0.3, "−log₁₀P", ha="left", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform(), color="#2c3e50", style="italic")

    # Group bracket lines on far left
    group_label_positions = {}
    for i, (label, OR, lo, hi, mlogp, group) in enumerate(DATA):
        group_label_positions.setdefault(group, []).append(y_positions[i])
    for group, ylist in group_label_positions.items():
        ymid = (min(ylist) + max(ylist)) / 2
        ymin = min(ylist) - 0.25
        ymax = max(ylist) + 0.25
        # bracket
        ax.annotate("", xy=(-0.27, ymin), xytext=(-0.27, ymax),
                    xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=COLORS[group], lw=1.5))
        ax.text(-0.29, ymid, LABELS[group], ha="right", va="center", fontsize=8.0,
                transform=ax.get_yaxis_transform(), color=COLORS[group],
                fontweight="bold", rotation=90)

    # X-axis: log scale
    ax.set_xscale("log")
    ax.set_xlim(0.7, 600)
    ax.set_xticks([1, 2, 5, 10, 25, 100, 500])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "100", "500"], fontsize=9)
    ax.set_xlabel("Odds ratio (95% CI)", fontsize=10, labelpad=8)

    # Y-axis: hide ticks, set limits
    ax.set_yticks([])
    ax.set_ylim(min(y_positions) - 0.8, max(y_positions) + 1.0)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Subtle horizontal guide lines
    for i, y in enumerate(y_positions):
        ax.axhline(y=y, color="#ecf0f1", lw=0.5, zorder=0)

    # Title
    ax.set_title(
        "Figure 1. FinnGen R13 phenome-wide associations of\n"
        "MUTYH p.Gly155Asp (rs587781864) across cancer phenotypes",
        fontsize=10.5, pad=14, color="#2c3e50", loc="left"
    )

    # Legend
    patches = [mpatches.Patch(color=COLORS[g], label=LABELS[g])
               for g in ["glioma", "map", "broad"]]
    ax.legend(handles=patches, fontsize=8.5, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#bdc3c7")

    # Footnote
    fig.text(0.05, 0.01,
             "* mlogp ≥ 3.45 (C3_GBM primary signal). FinnGen R13: N=467 GBM cases, 372,626 total.\n"
             "MUTYH p.Gly155Asp: ClinVar Pathogenic/Likely pathogenic for MAP (VCV000141595); "
             "gnomAD FIN enrichment 7.3× vs. NFE.",
             fontsize=7.5, color="#7f8c8d", va="bottom")

    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")
    print(f"Saved: {OUT_PDF}")
    plt.close()


if __name__ == "__main__":
    make_figure()
