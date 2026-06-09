#!/usr/bin/env python3
"""IDH-wildtype GBM: tumor mutational burden (TMB) analysis and the hypermutator tail.

GBM is generally immunotherapy-resistant (CheckMate-143, other trials negative). But a small
subset of GBMs develop a HYPERMUTATOR phenotype -- massively elevated TMB from mismatch-repair
deficiency or, crucially, TMZ-induced mutagenesis. These hypermutated tumors are:
  (1) More immunogenic (more neoantigens -> potentially visible to immune checkpoint therapy)
  (2) Often MGMT-unmethylated (TMZ causes mutagenesis precisely because DNA repair is intact
      and creates C->T transitions in MGMT-unmethylated tumors under TMZ pressure)
  (3) Potentially the GBM subgroup most likely to respond to nivolumab/pembrolizumab

This analysis computes TMB for all TCGA-GBM IDH-wt patients, identifies the hypermutated tail
(conventional threshold: >10 mut/Mb; GBM hypermutators often >100 mut/Mb), and cross-tabs with
MGMT status and CDK4/6-eligibility to understand where immunotherapy might complement CDK4/6i.

NOTE on TMB calculation: TCGA used WES (~30 Mb coding region; we use 30 Mb as denominator).
We count ALL non-silent somatic mutations per patient. TCGA TMB is systematically compared by
dividing total non-silent somatic mutations / 30 (Mb) -> mut/Mb.

HONEST CAVEATS: TCGA GBM patients were largely pre-treatment or treated with Stupp (TMZ+RT),
NOT with checkpoint inhibitors. High TMB here is a PREDICTIVE HYPOTHESIS (neoantigens should
exist), not a proven clinical benefit in GBM. TMB predicts immunotherapy benefit in some cancers
(NSCLC, TMB-high approval) but GBM's immunosuppressive microenvironment may still limit response.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2dist
from scipy.stats import mannwhitneyu

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE = f"{STUDY}_mutations"
CNA_PROFILE = f"{STUDY}_gistic"
SEQ_LIST, CNA_LIST, CNASEQ_LIST = f"{STUDY}_sequenced", f"{STUDY}_cna", f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_tmb.csv"

WES_MB = 30.0  # approximate WES coding region size
TMB_HIGH_THRESHOLD = 10.0  # mut/Mb (conventional threshold; GBM hypermutators often >>10)
TMB_EXTREME_THRESHOLD = 100.0  # mut/Mb (extreme hypermutators, likely MMRd or post-TMZ)

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}
CDK_GENES = ["IDH1", "IDH2", "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND2", "RB1"]
PATHWAY = [("CDKN2A", "homdel"), ("CDKN2B", "homdel"),
           ("CDK4", "amp"), ("CDK6", "amp"), ("CCND2", "amp")]
MGMT_SOURCES = [("lgggbm_tcga_pub", "MGMT_PROMOTER_STATUS"), ("gbm_tcga_pub2013", "MGMT_STATUS")]


def get(path):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(API + path, headers={"Accept": "application/json"}), timeout=120))


def post(path, body):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))


def has(d, g, kinds):
    return bool(d.get(g, set()) & set(kinds))


def km_median(times, events):
    t = np.asarray(times, float); e = np.asarray(events, int)
    at_risk = len(t); S = 1.0; median = None
    for ut in sorted(set(t)):
        d_n = int(np.sum((t == ut) & (e == 1)))
        c = int(np.sum((t == ut) & (e == 0)))
        if at_risk > 0 and d_n > 0:
            S *= 1 - d_n / at_risk
            if median is None and S <= 0.5:
                median = ut
        at_risk -= (d_n + c)
    return median


def logrank(t1, e1, t2, e2):
    t1, e1, t2, e2 = map(lambda x: np.asarray(x, float), (t1, e1, t2, e2))
    O1 = E1 = V = 0.0
    for ut in sorted(set(np.concatenate([t1, t2]))):
        r1 = int(np.sum(t1 >= ut)); r2 = int(np.sum(t2 >= ut)); r = r1 + r2
        d1 = int(np.sum((t1 == ut) & (e1 == 1)))
        d = d1 + int(np.sum((t2 == ut) & (e2 == 1)))
        if r >= 2 and d > 0:
            O1 += d1; E1 += d * r1 / r
            V += d * (r1 / r) * (1 - r1 / r) * (r - d) / (r - 1)
    chi2 = (O1 - E1) ** 2 / V if V > 0 else 0.0
    return chi2, float(chi2dist.sf(chi2, 1))


def main():
    print("=" * 100)
    print("IDH-WILDTYPE GBM: tumor mutational burden (TMB) + hypermutator analysis")
    print(f"TMB = non-silent somatic mutations / {WES_MB} Mb (WES exome)")
    print(f"Thresholds: TMB-high >{TMB_HIGH_THRESHOLD} mut/Mb | extreme >{TMB_EXTREME_THRESHOLD} mut/Mb")
    print("=" * 100)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    cnaseq_pat = {samp2pat[s] for s in set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
                  if s in samp2pat}

    cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    clin: dict[str, dict] = {}
    for r in cd:
        clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]

    def os_of(pid):
        d = clin.get(pid, {})
        try:
            m = float(d["OS_MONTHS"])
        except (KeyError, TypeError, ValueError):
            return None
        ev = 1 if d.get("OS_STATUS", "").startswith("1") or "DECEASED" in d.get("OS_STATUS", "").upper() else 0
        return (m, ev)

    # use precomputed TMB from cBioPortal sample clinical data (avoids fetching all mutations)
    print("fetching precomputed TMB (TMB_NONSYNONYMOUS) and MUTATION_COUNT from sample attributes...")
    samp_cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=SAMPLE&pageSize=2000000")
    samp_tmb: dict[str, float] = {}   # patientId -> TMB (mut/Mb)
    samp_mc: dict[str, int] = {}      # patientId -> mutation count (raw)
    for r in samp_cd:
        pid = samp2pat.get(r["sampleId"])
        if not pid:
            continue
        if r["clinicalAttributeId"] == "TMB_NONSYNONYMOUS":
            try:
                samp_tmb[pid] = float(r["value"])
            except (ValueError, TypeError):
                pass
        elif r["clinicalAttributeId"] == "MUTATION_COUNT":
            try:
                samp_mc[pid] = int(float(r["value"]))
            except (ValueError, TypeError):
                pass
    print(f"  TMB_NONSYNONYMOUS: {len(samp_tmb)} patients | MUTATION_COUNT: {len(samp_mc)} patients")

    # fetch targeted genes for eligibility
    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", CDK_GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    cdk_muts = post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                    {"sampleListId": SEQ_LIST, "entrezGeneIds": entrez})
    alt: dict[str, dict[str, set]] = {}
    for m in cdk_muts:
        g = ez2sym.get(m.get("entrezGeneId"))
        if g and m.get("mutationType") not in SILENT:
            pid = samp2pat.get(m["sampleId"])
            if pid:
                alt.setdefault(pid, {}).setdefault(g, set()).add("mut")
    cnas = post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
                {"sampleListId": CNA_LIST, "entrezGeneIds": entrez})
    for c in cnas:
        g = ez2sym.get(c.get("entrezGeneId"))
        pid = samp2pat.get(c["sampleId"])
        if g and pid:
            if c.get("alteration") == 2:
                alt.setdefault(pid, {}).setdefault(g, set()).add("amp")
            elif c.get("alteration") == -2:
                alt.setdefault(pid, {}).setdefault(g, set()).add("homdel")

    # MGMT
    mgmt: dict[str, int] = {}
    for study, attr in MGMT_SOURCES:
        smp = get(f"/studies/{study}/samples?pageSize=100000")
        s2p = {s["sampleId"]: s["patientId"] for s in smp}
        data = get(f"/studies/{study}/clinical-data?clinicalDataType=SAMPLE&attributeId={attr}&pageSize=2000000")
        for r in data:
            if r["clinicalAttributeId"] != attr:
                continue
            v = (r["value"] or "").strip().upper()
            pid = s2p.get(r["sampleId"])
            if pid and pid not in mgmt and v in ("METHYLATED", "UNMETHYLATED"):
                mgmt[pid] = 1 if v == "METHYLATED" else 0

    cohort = [p for p in cnaseq_pat if os_of(p) is not None]

    def is_idh_mut(d): return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])
    def pathway_on(d): return any(has(d, g, [k]) for g, k in PATHWAY)
    def eligible(d): return (not is_idh_mut(d)) and pathway_on(d) and not has(d, "RB1", ["mut", "homdel"])

    idh_wt = [p for p in cohort if not is_idh_mut(alt.get(p, {}))]
    # prefer TMB_NONSYNONYMOUS (mut/Mb directly); fall back to MUTATION_COUNT/30Mb
    tmb = {p: samp_tmb.get(p, samp_mc.get(p, 0) / WES_MB) for p in idh_wt}

    all_tmb = sorted(tmb.values())
    tmb_high = [p for p in idh_wt if tmb[p] >= TMB_HIGH_THRESHOLD]
    tmb_extreme = [p for p in idh_wt if tmb[p] >= TMB_EXTREME_THRESHOLD]
    tmb_low = [p for p in idh_wt if tmb[p] < TMB_HIGH_THRESHOLD]

    print(f"\nIDH-wt GBM cohort: {len(idh_wt)} patients with TMB data")
    print(f"TMB distribution (mut/Mb):")
    print(f"  median: {np.median(all_tmb):.1f}")
    print(f"  mean:   {np.mean(all_tmb):.1f}")
    print(f"  p95:    {np.percentile(all_tmb,95):.1f}")
    print(f"  max:    {max(all_tmb):.1f}")
    print(f"  TMB-high (>{TMB_HIGH_THRESHOLD}):    n={len(tmb_high)} ({100*len(tmb_high)/len(idh_wt):.1f}%)")
    print(f"  TMB-extreme (>{TMB_EXTREME_THRESHOLD}):  n={len(tmb_extreme)} ({100*len(tmb_extreme)/len(idh_wt):.1f}%)")

    # cross-tab TMB-high with MGMT and eligibility
    elig_set = {p for p in idh_wt if eligible(alt.get(p, {}))}
    meth_set = {p for p in idh_wt if mgmt.get(p) == 1}
    unmeth_set = {p for p in idh_wt if mgmt.get(p) == 0}

    print(f"\nTMB-high cross-tab (within IDH-wt):")
    for lab, grp in [("MGMT-methylated", meth_set & set(idh_wt)),
                     ("MGMT-unmethylated", unmeth_set & set(idh_wt)),
                     ("CDK4/6-eligible", elig_set),
                     ("not CDK4/6-eligible", set(idh_wt) - elig_set)]:
        grp = list(grp)
        if not grp: continue
        n_hi = sum(1 for p in grp if tmb[p] >= TMB_HIGH_THRESHOLD)
        print(f"  {lab:<30} n={len(grp):>3}  TMB-high={n_hi} ({100*n_hi/len(grp):.1f}%)")

    # of highest clinical interest: CDK4/6-eligible + MGMT-unmethylated and their TMB
    elig_unmeth = [p for p in idh_wt if p in elig_set and p in unmeth_set]
    elig_unmeth_hi = [p for p in elig_unmeth if tmb[p] >= TMB_HIGH_THRESHOLD]
    print(f"\nFOCUS -- CDK4/6-eligible + MGMT-unmethylated (n={len(elig_unmeth)}, the highest-need group):")
    print(f"  TMB-high (>{TMB_HIGH_THRESHOLD} mut/Mb): n={len(elig_unmeth_hi)} ({100*len(elig_unmeth_hi)/max(len(elig_unmeth),1):.1f}%)")
    print(f"  These patients have: (1) rational CDK4/6i target, (2) TMZ helps least,")
    print(f"  (3) if TMB-high -> also potentially immunotherapy-eligible (neoantigens present).")
    print(f"  = the 'triple-play' subgroup for combination therapy reasoning.")

    # OS by TMB-high vs low within IDH-wt
    print(f"\nOS: TMB-high vs TMB-low within IDH-wt")
    if len(tmb_high) >= 5 and len(tmb_low) >= 5:
        th, eh = zip(*[os_of(p) for p in tmb_high]); tl, el = zip(*[os_of(p) for p in tmb_low])
        mh, ml = km_median(th, eh), km_median(tl, el)
        _, pval = logrank(th, eh, tl, el)
        fh = f"{mh:.1f}" if mh else "NA"; fl = f"{ml:.1f}" if ml else "NA"
        print(f"  TMB-high (>10 mut/Mb)  n={len(tmb_high):>3}  median OS = {fh} mo")
        print(f"  TMB-low  (<10 mut/Mb)  n={len(tmb_low):>3}  median OS = {fl} mo  p={pval:.4f}")
        print(f"  (longer TMB-high OS = hypermutators sometimes have different biology; shorter = more aggressive)")

    # identify the extreme outliers
    extreme_sorted = sorted(tmb_extreme, key=lambda p: -tmb[p])
    print(f"\nTMB-extreme (>{TMB_EXTREME_THRESHOLD} mut/Mb) — likely MMRd or post-TMZ hypermutators:")
    for p in extreme_sorted[:10]:
        m_val = "MGMT-M" if p in meth_set else "MGMT-UM" if p in unmeth_set else "MGMT-?"
        e_val = "CDK-elig" if p in elig_set else "CDK-other"
        print(f"  {p}  TMB={tmb[p]:.0f} mut/Mb  {m_val}  {e_val}")

    # write output
    rows = [{"patient": p, "tmb_per_mb": round(tmb[p], 2),
             "tmb_high": int(tmb[p] >= TMB_HIGH_THRESHOLD),
             "tmb_extreme": int(tmb[p] >= TMB_EXTREME_THRESHOLD),
             "cdk46_eligible": int(p in elig_set),
             "mgmt_methylated": mgmt.get(p, "")} for p in idh_wt]
    rows.sort(key=lambda r: -r["tmb_per_mb"])
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient","tmb_per_mb","tmb_high","tmb_extreme",
                                           "cdk46_eligible","mgmt_methylated"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 100)
    print("READING: TMB-high GBMs are candidates for checkpoint immunotherapy (more neoantigens).")
    print("MGMT-unmethylated + post-TMZ = highest risk of hypermutation (C->T transitions from TMZ).")
    print("The 'triple-play' subgroup (CDK4/6-eligible + MGMT-unmethylated + TMB-high) is tiny but")
    print("rationally targets all three axes simultaneously: cell-cycle (abemaciclib) + immune (checkpoint).")
    print("TCGA patients weren't treated with immunotherapy -> OS here is observational baseline only.")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
