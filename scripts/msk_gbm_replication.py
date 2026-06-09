#!/usr/bin/env python3
"""INDEPENDENT REPLICATION: CDK4/6i-eligible subset + MGMT in MSK Glioma 2019 cohort.

All findings in tcga_gbm_cdk_eligible.py and tcga_gbm_mgmt.py were derived from TCGA-GBM
(PanCancer Atlas). Science requires replication in an independent cohort.

This script runs the identical eligibility + MGMT analysis in the MSK Glioma 2019 cohort
(Clin Cancer Res 2019, N=1004 total, N=539 GBM). MSK used MSK-IMPACT panel sequencing +
GISTIC CNA -- different institution, different sequencing platform, different patient population.
Agreement = the thesis is robust and generalizable. Disagreement = TCGA-specific artifact.

We compare directly:
  TCGA: eligible = 69.5% of IDH-wt GBM, MGMT-eligibility OR = 0.83 (p=0.52, independent)
  MSK:  <computed here>

HONEST CAVEATS: MSK-IMPACT is a panel (not WES); smaller mutations may differ by coverage.
GISTIC CNA calls are more similar across platforms. Unadjusted KM/log-rank for survival.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2dist
from scipy.stats import fisher_exact

API = "https://www.cbioportal.org/api"
STUDY = "glioma_mskcc_2019"
MUT_PROFILE = f"{STUDY}_mutations"
CNA_PROFILE = f"{STUDY}_gistic"
ALL_LIST = f"{STUDY}_all"
GBM_HISTOLOGY = "Glioblastoma Multiforme"
OUT = Path.home() / "finngen-triage" / "msk_gbm_replication.csv"

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}
GENES = ["IDH1", "IDH2", "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND2", "RB1"]
PATHWAY = [("CDKN2A", "homdel"), ("CDKN2B", "homdel"),
           ("CDK4", "amp"), ("CDK6", "amp"), ("CCND2", "amp")]


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
        d = int(np.sum((t == ut) & (e == 1)))
        c = int(np.sum((t == ut) & (e == 0)))
        if at_risk > 0 and d > 0:
            S *= 1 - d / at_risk
            if median is None and S <= 0.5:
                median = ut
        at_risk -= (d + c)
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
    print("=" * 104)
    print("INDEPENDENT REPLICATION (MSK Glioma 2019, N_GBM=539): CDK4/6i-eligible + MGMT analysis")
    print("TCGA reference: eligible=69.5% of IDH-wt GBM | MGMT-eligibility OR=0.83 (independent)")
    print("=" * 104)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}

    # clinical: patient-level OS, sample-level histology + MGMT
    pat_cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    pat_clin: dict[str, dict] = {}
    for r in pat_cd:
        pat_clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]
    samp_cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=SAMPLE&pageSize=2000000")
    samp_clin: dict[str, dict] = {}
    for r in samp_cd:
        samp_clin.setdefault(r["sampleId"], {})[r["clinicalAttributeId"]] = r["value"]

    # filter to GBM samples
    gbm_samps = {s["sampleId"] for s in samples
                 if samp_clin.get(s["sampleId"], {}).get("CANCER_TYPE_DETAILED") == GBM_HISTOLOGY}
    gbm_pats = {samp2pat[s] for s in gbm_samps if s in samp2pat}
    print(f"\nGBM samples: {len(gbm_samps)}, unique patients: {len(gbm_pats)}")

    def os_of(pid):
        d = pat_clin.get(pid, {})
        try:
            m = float(d["OS_MONTHS"])
        except (KeyError, TypeError, ValueError):
            return None
        ev = 1 if d.get("OS_STATUS", "").startswith("1") or "DECEASED" in d.get("OS_STATUS", "").upper() else 0
        return (m, ev)

    # resolve genes
    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    # fetch alterations using sample-list approach (all samples, filter in Python)
    alt: dict[str, dict[str, set]] = {}
    print("fetching mutations...")
    muts = post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                {"sampleListId": ALL_LIST, "entrezGeneIds": entrez})
    for m in muts:
        if m["sampleId"] not in gbm_samps:
            continue
        g = ez2sym.get(m.get("entrezGeneId"))
        if g and m.get("mutationType") not in SILENT:
            pid = samp2pat.get(m["sampleId"])
            if pid:
                alt.setdefault(pid, {}).setdefault(g, set()).add("mut")
    print("fetching CNA...")
    cnas = post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
                {"sampleListId": ALL_LIST, "entrezGeneIds": entrez})
    pat_has_cna = set()
    for c in cnas:
        if c["sampleId"] not in gbm_samps:
            continue
        g = ez2sym.get(c.get("entrezGeneId"))
        pid = samp2pat.get(c["sampleId"])
        if not g or not pid:
            continue
        pat_has_cna.add(pid)
        if c.get("alteration") == 2:
            alt.setdefault(pid, {}).setdefault(g, set()).add("amp")
        elif c.get("alteration") == -2:
            alt.setdefault(pid, {}).setdefault(g, set()).add("homdel")

    cohort = [p for p in gbm_pats if p in pat_has_cna and os_of(p) is not None]
    N = len(cohort)
    print(f"GBM analysis cohort (CNA + OS): {N} patients\n")

    def is_idh_mut(d): return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])
    def pathway_on(d): return any(has(d, g, [k]) for g, k in PATHWAY)
    def eligible(p):
        d = alt.get(p, {})
        return (not is_idh_mut(d)) and pathway_on(d) and not has(d, "RB1", ["mut", "homdel"])

    idh_wt = [p for p in cohort if not is_idh_mut(alt.get(p, {}))]
    idh_mut = [p for p in cohort if is_idh_mut(alt.get(p, {}))]
    elig = [p for p in idh_wt if eligible(p)]
    pct = lambda n: f"{100*n/N:5.1f}%"
    pctw = lambda n: f"{100*n/len(idh_wt):5.1f}%" if idh_wt else "NA"

    print("COHORT BREAKDOWN (compare to TCGA in brackets)")
    print(f"  IDH-wildtype              {len(idh_wt):>4}  ({pct(len(idh_wt))} of cohort)  [TCGA: 93.6%]")
    print(f"  IDH-mutant                {len(idh_mut):>4}  ({pct(len(idh_mut))} of cohort)  [TCGA:  6.4%]")

    rb1_blocked = sum(1 for p in idh_wt if pathway_on(alt.get(p, {})) and has(alt.get(p, {}), "RB1", ["mut", "homdel"]))
    no_pathway = sum(1 for p in idh_wt if not pathway_on(alt.get(p, {})))
    print(f"\nWithin IDH-wt (compare to TCGA):")
    print(f"  CDK4/6 pathway ON         {len(idh_wt)-no_pathway:>4}  ({pctw(len(idh_wt)-no_pathway)} of IDH-wt)  [TCGA: 71.5%]")
    print(f"    RB1 lost (block)         {rb1_blocked:>4}")
    print(f"    ELIGIBLE                 {len(elig):>4}  ({pctw(len(elig))} of IDH-wt)  [TCGA: 69.5%]")
    print(f"  no CDK4/6 lesion          {no_pathway:>4}  ({pctw(no_pathway)} of IDH-wt)  [TCGA: 28.5%]")
    print(f"\n>>> MSK CDK4/6i-ELIGIBLE: {len(elig)} = {pct(len(elig))} of GBM cohort, {pctw(len(elig))} of IDH-wt")
    print(f"    TCGA reference:          65.1% of GBM, 69.5% of IDH-wt")

    # qualifying lesion breakdown
    print("\nQualifying lesion in eligible (MSK vs TCGA):")
    for g, k in PATHWAY:
        n = sum(1 for p in elig if has(alt.get(p, {}), g, [k]))
        pct_e = 100*n/len(elig) if elig else 0
        print(f"  {g} {k:<8} {n:>3} ({pct_e:4.1f}% of eligible)")

    # MGMT
    wt_mgmt = {p: (pat_clin.get(p, {}).get("MGMT_STATUS") or "").strip().upper()
               for p in idh_wt}
    wt_mgmt = {p: ("METHYLATED" if v == "METHYLATED" else "UNMETHYLATED" if v == "UNMETHYLATED" else None)
               for p, v in wt_mgmt.items()}
    wt_mgmt_valid = {p: v for p, v in wt_mgmt.items() if v is not None}
    n_meth = sum(1 for v in wt_mgmt_valid.values() if v == "METHYLATED")
    print(f"\nMGMT in IDH-wt GBM: {len(wt_mgmt_valid)} with calls ({100*len(wt_mgmt_valid)/len(idh_wt):.0f}%)")
    print(f"  Methylated: {n_meth} ({100*n_meth/max(len(wt_mgmt_valid),1):.0f}%)  [TCGA: 43% methylated in IDH-wt]")

    # Fisher: MGMT x eligibility (compare to TCGA OR=0.83, p=0.52)
    elig_set = set(elig)
    a = sum(1 for p in wt_mgmt_valid if p in elig_set and wt_mgmt_valid[p] == "METHYLATED")
    b = sum(1 for p in wt_mgmt_valid if p in elig_set and wt_mgmt_valid[p] == "UNMETHYLATED")
    c = sum(1 for p in wt_mgmt_valid if p not in elig_set and wt_mgmt_valid[p] == "METHYLATED")
    d = sum(1 for p in wt_mgmt_valid if p not in elig_set and wt_mgmt_valid[p] == "UNMETHYLATED")
    orr, pfish = fisher_exact([[a, b], [c, d]])
    print(f"\nMGMT x CDK4/6-eligibility (Fisher exact): OR={orr:.2f} p={pfish:.3f}")
    print(f"  eligible:methylated={a} eligible:unmethylated={b}  other:methylated={c} other:unmethylated={d}")
    print(f"  TCGA reference: OR=0.83 p=0.52 (independent)")
    print(f"  -> {'INDEPENDENT (replicates TCGA)' if pfish >= 0.05 else 'ASSOCIATED (discordant with TCGA)'}")

    # Survival
    print("\nSURVIVAL (KM, within IDH-wt):")
    rows = []
    def survline(grp_a, grp_b, la, lb, tcga_a=None, tcga_b=None, tcga_p=None):
        if len(grp_a) < 5 or len(grp_b) < 5:
            print(f"  too few for {la}")
            return
        ta, ea = zip(*[os_of(p) for p in grp_a]); tb, eb = zip(*[os_of(p) for p in grp_b])
        ma, mb = km_median(ta, ea), km_median(tb, eb)
        _, pval = logrank(ta, ea, tb, eb)
        fa = f"{ma:.1f}" if ma is not None else "NA"
        fb = f"{mb:.1f}" if mb is not None else "NA"
        print(f"  {la:<38} n={len(grp_a):>3} med={fa:>5} mo")
        print(f"  {lb:<38} n={len(grp_b):>3} med={fb:>5} mo  p={pval:.4f}")
        if tcga_a is not None:
            print(f"    TCGA: {tcga_a:.1f} vs {tcga_b:.1f} mo, p={tcga_p:.4f}")
        rows.append({"comparison": la+" vs "+lb, "n_a": len(grp_a), "med_a_mo": round(ma,1) if ma else None,
                     "n_b": len(grp_b), "med_b_mo": round(mb,1) if mb else None, "logrank_p": round(pval,4)})

    other_wt = [p for p in idh_wt if p not in elig_set]
    survline(elig, other_wt, "CDK4/6i-eligible (IDH-wt)", "other IDH-wt",
             tcga_a=12.9, tcga_b=15.3, tcga_p=0.12)
    meth_wt = [p for p in idh_wt if wt_mgmt_valid.get(p) == "METHYLATED"]
    unmeth_wt = [p for p in idh_wt if wt_mgmt_valid.get(p) == "UNMETHYLATED"]
    survline(meth_wt, unmeth_wt, "MGMT methylated (IDH-wt)", "MGMT unmethylated",
             tcga_a=14.9, tcga_b=12.7, tcga_p=0.087)

    # 4-way table
    elig_meth = [p for p in wt_mgmt_valid if p in elig_set and wt_mgmt_valid[p] == "METHYLATED"]
    elig_unmeth = [p for p in wt_mgmt_valid if p in elig_set and wt_mgmt_valid[p] == "UNMETHYLATED"]
    other_meth = [p for p in wt_mgmt_valid if p not in elig_set and wt_mgmt_valid[p] == "METHYLATED"]
    other_unmeth = [p for p in wt_mgmt_valid if p not in elig_set and wt_mgmt_valid[p] == "UNMETHYLATED"]
    print("\n4-WAY OS TABLE (MSK vs TCGA in brackets):")
    for lab, grp, tcga_med in [("elig+methylated", elig_meth, 14.9),
                                 ("elig+unmethylated", elig_unmeth, 12.6),
                                 ("other+methylated", other_meth, 15.3),
                                 ("other+unmethylated", other_unmeth, 13.3)]:
        if len(grp) >= 5:
            t, e = zip(*[os_of(p) for p in grp]); med = km_median(t, e)
            print(f"  {lab:<22} n={len(grp):>3} med={med:.1f} mo  [TCGA:{tcga_med:.1f}]" if med else
                  f"  {lab:<22} n={len(grp):>3} med=NA")
        rows.append({"comparison": lab, "n_a": len(grp), "med_a_mo": None,
                     "n_b": None, "med_b_mo": None, "logrank_p": None})

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["comparison","n_a","med_a_mo","n_b","med_b_mo","logrank_p"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 104)
    print("REPLICATION VERDICT: does the MSK cohort reproduce TCGA's CDK4/6i-eligible fraction")
    print("and the MGMT-eligibility independence? If yes -> thesis is robust across institutions.")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
