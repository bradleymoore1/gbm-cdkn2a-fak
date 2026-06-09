#!/usr/bin/env python3
"""Front B -- TCGA-GBM SOMATIC driver landscape (cBioPortal public API).

WHY THIS, NOT FINNGEN, FOR GBM BIOLOGY:
  FinnGen is a GERMLINE biobank (inherited blood DNA) -> it can only find inherited
  PREDISPOSITION. Glioblastoma is overwhelmingly driven by SOMATIC mutations that arise
  in the tumor and are invisible to FinnGen. The therapeutic targets live here, in tumor
  sequencing. TCGA-GBM is the canonical public somatic cohort.

This reproduces the GBM somatic driver landscape from TCGA (PanCancer Atlas, 2018):
per-gene somatic mutation frequency + GISTIC copy-number amplification / deep-deletion.
Reproducing the known landscape (EGFR amp ~40%, PTEN/TP53/NF1 mut, CDKN2A/B homdel ~50%,
IDH1 mut rare in primary GBM) is the VALIDATION that the somatic pipeline is wired right
before we push it anywhere novel (druggable-target mapping is the next module).

GISTIC discrete CNA: -2 deep/homozygous deletion, -1 shallow del, 0 diploid,
                      +1 gain, +2 high amplification. We count +2 (amp) and -2 (homdel).

Caveats kept honest:
  * TERT PROMOTER mutations (the most common GBM alteration, ~80%) are NON-CODING and are
    typically ABSENT from exome mutation calls -- do not expect TERT here.
  * MGMT promoter METHYLATION (the key temozolomide-response biomarker) is a separate
    methylation profile, not mutation/CNA -- flagged but not fetched in this pass.
  * Mutation denominator = sequenced samples; CNA denominator = CNA samples; these differ.
    Combined 'altered%' is computed over the intersection (cnaseq) for an honest union.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE = f"{STUDY}_mutations"
CNA_PROFILE = f"{STUDY}_gistic"
SEQ_LIST = f"{STUDY}_sequenced"
CNA_LIST = f"{STUDY}_cna"
CNASEQ_LIST = f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_somatic_landscape.csv"

# Curated GBM driver panel. role drives CNA interpretation:
#   ONC -> amplification (+2) is the oncogenic event
#   TSG -> homozygous deletion (-2) is the oncogenic event
#   BOTH/? -> report both, no single expectation
PANEL = {
    # RTK / RAS / PI3K axis
    "EGFR": "ONC", "PDGFRA": "ONC", "MET": "ONC", "FGFR1": "ONC", "FGFR3": "ONC",
    "PIK3CA": "ONC", "PIK3R1": "TSG", "PTEN": "TSG", "NF1": "TSG", "KRAS": "ONC",
    "BRAF": "ONC", "AKT3": "ONC", "PTPN11": "ONC",
    # p53 pathway
    "TP53": "TSG", "MDM2": "ONC", "MDM4": "ONC",
    # RB / cell-cycle pathway
    "RB1": "TSG", "CDKN2A": "TSG", "CDKN2B": "TSG", "CDK4": "ONC", "CDK6": "ONC",
    "CCND2": "ONC",
    # chromatin / IDH / telomere / other recurrent GBM drivers
    "IDH1": "ONC", "IDH2": "ONC", "ATRX": "TSG", "TP53BP1": "?",
    "QKI": "TSG", "LZTR1": "TSG", "MYCN": "ONC", "SOX2": "ONC", "MYC": "ONC",
}

# Genes whose primary clinical signal is NOT mutation/CNA -- reported as context only.
CONTEXT = {
    "TERT": "promoter mutation (~80% GBM) is non-coding; usually absent from exome calls",
    "MGMT": "promoter METHYLATION predicts temozolomide response; separate methylation profile",
}


def post(path: str, body) -> list | dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}", data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(path: str) -> list | dict:
    req = urllib.request.Request(f"{API}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def sample_ids(list_id: str) -> set[str]:
    d = get(f"/sample-lists/{list_id}")
    return set(d.get("sampleIds", []))


def main():
    print("=" * 100)
    print("TCGA-GBM SOMATIC driver landscape  (cBioPortal PanCancer Atlas 2018)")
    print("Germline (FinnGen) cannot see these -- this is tumor DNA, where the targets are.")
    print("=" * 100)

    # --- cohort denominators ---
    seq = sample_ids(SEQ_LIST)
    cna = sample_ids(CNA_LIST)
    cnaseq = sample_ids(CNASEQ_LIST)
    print(f"cohort: {len(seq)} sequenced (mutation)  |  {len(cna)} CNA  |  "
          f"{len(cnaseq)} with BOTH (cnaseq, used for combined altered%)")

    # --- resolve genes -> entrez ---
    genes = list(PANEL)
    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", genes)
    sym2ez = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in resolved}
    ez2sym = {v: k for k, v in sym2ez.items()}
    missing = [g for g in genes if g not in sym2ez]
    if missing:
        print(f"  (unresolved symbols, skipped: {missing})")
    entrez = list(sym2ez.values())

    # --- fetch mutations (SUMMARY) over sequenced list ---
    muts = post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                {"sampleListId": SEQ_LIST, "entrezGeneIds": entrez})
    mut_samples: dict[str, set[str]] = {g: set() for g in genes}
    mut_types: dict[str, set[str]] = {g: set() for g in genes}
    for m in muts:
        g = ez2sym.get(m.get("entrezGeneId"))
        if g is None:
            continue
        mut_samples[g].add(m["sampleId"])
        mt = m.get("mutationType")
        if mt:
            mut_types[g].add(mt)

    # --- fetch discrete CNA (GISTIC) over cna list ---
    cnas = post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch"
                f"?discreteCopyNumberEventType=ALL",
                {"sampleListId": CNA_LIST, "entrezGeneIds": entrez})
    amp_samples: dict[str, set[str]] = {g: set() for g in genes}
    del_samples: dict[str, set[str]] = {g: set() for g in genes}
    for c in cnas:
        g = ez2sym.get(c.get("entrezGeneId"))
        if g is None:
            continue
        a = c.get("alteration")
        if a == 2:
            amp_samples[g].add(c["sampleId"])
        elif a == -2:
            del_samples[g].add(c["sampleId"])

    # --- per-gene frequencies ---
    rows = []
    n_seq, n_cna, n_cs = len(seq) or 1, len(cna) or 1, len(cnaseq) or 1
    for g in genes:
        ms, amps, dels = mut_samples[g], amp_samples[g], del_samples[g]
        # combined altered set restricted to samples profiled for BOTH (cnaseq)
        altered_cs = {s for s in cnaseq if s in ms or s in amps or s in dels}
        rows.append({
            "gene": g, "role": PANEL[g],
            "mut_n": len(ms), "mut_pct": 100 * len(ms) / n_seq,
            "amp_n": len(amps), "amp_pct": 100 * len(amps) / n_cna,
            "homdel_n": len(dels), "homdel_pct": 100 * len(dels) / n_cna,
            "altered_pct": 100 * len(altered_cs) / n_cs,
            "mut_types": ";".join(sorted(mut_types[g])),
        })

    rows.sort(key=lambda r: -r["altered_pct"])

    # --- write CSV ---
    cols = ["gene", "role", "altered_pct", "mut_n", "mut_pct", "amp_n", "amp_pct",
            "homdel_n", "homdel_pct", "mut_types"]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: round(r[k], 1) for k in
                                ("altered_pct", "mut_pct", "amp_pct", "homdel_pct")}})

    # --- report ---
    print(f"\n{'gene':<9}{'role':>5}{'altered%':>10}{'mut%':>8}{'amp%':>8}{'homdel%':>9}   key event")
    print("-" * 90)
    for r in rows:
        if r["role"] == "ONC":
            key = f"amp {r['amp_pct']:.0f}%" if r["amp_pct"] >= r["homdel_pct"] else f"del {r['homdel_pct']:.0f}%"
            if r["mut_pct"] > max(r["amp_pct"], 5):
                key = f"mut {r['mut_pct']:.0f}% / " + key
        elif r["role"] == "TSG":
            key = f"homdel {r['homdel_pct']:.0f}%" if r["homdel_pct"] >= r["mut_pct"] else f"mut {r['mut_pct']:.0f}%"
            if r["mut_pct"] > 5 and r["homdel_pct"] > 5:
                key = f"mut {r['mut_pct']:.0f}% + homdel {r['homdel_pct']:.0f}%"
        else:
            key = "mixed"
        print(f"{r['gene']:<9}{r['role']:>5}{r['altered_pct']:>9.1f}%{r['mut_pct']:>7.1f}%"
              f"{r['amp_pct']:>7.1f}%{r['homdel_pct']:>8.1f}%   {key}")

    print(f"\nwrote {OUT.name} ({len(rows)} driver genes)")
    print("\ncontext genes (not mutation/CNA-driven, flagged for honesty):")
    for g, note in CONTEXT.items():
        print(f"  {g}: {note}")
    print("\nSanity check vs known TCGA-GBM: expect EGFR amp ~40-50% & mut ~25%, "
          "CDKN2A/B homdel ~50-60%,\nPTEN mut ~30%, TP53 mut ~28%, NF1 ~15%, "
          "PDGFRA amp ~10%, IDH1 mut LOW in primary GBM (~5%).")
    print("If those line up, the somatic pipeline is trustworthy -> next: map drivers to "
          "druggable\ntargets (Open Targets / DGIdb), honest about BBB penetrance & the GBM trial graveyard.")


if __name__ == "__main__":
    main()
