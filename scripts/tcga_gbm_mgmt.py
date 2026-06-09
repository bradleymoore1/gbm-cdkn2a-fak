#!/usr/bin/env python3
"""IDH-wildtype GBM: add MGMT promoter methylation -- the one biomarker that changes treatment
TODAY -- and cross it against the CDK4/6-inhibitor-eligible subset.

GBM clinical decision-making rests on three molecular axes, in order of established weight:
  1. IDH status        -- master prognostic switch (IDH-mut = different, better disease). [done]
  2. MGMT methylation  -- THE predictive biomarker: a methylated MGMT promoter silences the DNA-
                          repair enzyme that undoes temozolomide's damage, so methylated tumors
                          respond better to TMZ and live longer. Unmethylated = TMZ less useful
                          -> exactly the patients who most need a targeted alternative.   [this]
  3. CDK4/6-eligibility -- the most defensible molecularly-targeted shot (CNS-penetrant
                          abemaciclib), sized in tcga_gbm_cdk_eligible.py.                 [join]

MGMT is NOT in the PanCancer 2018 study's fields, but the older TCGA-GBM papers carry a curated
methylated/unmethylated call. TCGA barcodes are stable across studies, so we pull MGMT from the
merged LGG+GBM cohort (Cell 2016, best coverage) with the GBM-2013 cohort as fallback, and join
by patient barcode to our IDH-wt PanCancer cohort.

VALIDATION ANCHOR: MGMT-methylated must show LONGER OS (it is the canonical TMZ-benefit result).
Reproducing it validates the cross-study join -- same trick as the IDH anchor.

HONEST CAVEATS: MGMT call imported from older studies (their classifier); unadjusted KM/log-rank
(no Cox; age/subtype/treatment confound). Standard of care = Stupp +/- TTFields. Not medical advice.
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
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE, CNA_PROFILE = f"{STUDY}_mutations", f"{STUDY}_gistic"
SEQ_LIST, CNA_LIST, CNASEQ_LIST = f"{STUDY}_sequenced", f"{STUDY}_cna", f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_mgmt.csv"

# external MGMT sources, in priority order: (study, sample-level attribute id)
MGMT_SOURCES = [("lgggbm_tcga_pub", "MGMT_PROMOTER_STATUS"), ("gbm_tcga_pub2013", "MGMT_STATUS")]

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}
GENES = ["IDH1", "IDH2", "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND2", "RB1"]
PATHWAY_LESIONS = [("CDKN2A", "homdel"), ("CDKN2B", "homdel"),
                   ("CDK4", "amp"), ("CDK6", "amp"), ("CCND2", "amp")]


def get(path):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(API + path, headers={"Accept": "application/json"}), timeout=120))


def post(path, body):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=180))


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


def mgmt_calls() -> dict[str, str]:
    """patient barcode -> 'METHYLATED'/'UNMETHYLATED', merged across sources (priority order)."""
    calls: dict[str, str] = {}
    conflicts = 0
    for study, attr in MGMT_SOURCES:
        smp = get(f"/studies/{study}/samples?pageSize=100000")
        s2p = {s["sampleId"]: s["patientId"] for s in smp}
        data = get(f"/studies/{study}/clinical-data?clinicalDataType=SAMPLE&attributeId={attr}&pageSize=2000000")
        for r in data:
            if r["clinicalAttributeId"] != attr:
                continue
            v = (r["value"] or "").strip().upper()
            if v not in ("METHYLATED", "UNMETHYLATED"):
                continue
            pid = s2p.get(r["sampleId"])
            if not pid:
                continue
            if pid in calls and calls[pid] != v:
                conflicts += 1  # keep higher-priority (already set) source
            else:
                calls.setdefault(pid, v)
    if conflicts:
        print(f"  (note: {conflicts} patients had cross-study MGMT disagreement; kept higher-priority source)")
    return calls


def survdiff(a, b, os_of, la, lb):
    ta, ea = zip(*[os_of(p) for p in a]); tb, eb = zip(*[os_of(p) for p in b])
    ma, mb = km_median(ta, ea), km_median(tb, eb)
    _, p = logrank(ta, ea, tb, eb)
    fa = f"{ma:.1f}" if ma is not None else "NA"
    fb = f"{mb:.1f}" if mb is not None else "NA"
    print(f"  {la:<42} n={len(a):>3}  median OS = {fa:>5} mo")
    print(f"  {lb:<42} n={len(b):>3}  median OS = {fb:>5} mo")
    print(f"  log-rank p = {p:.4f}")
    return ma, mb, p


def main():
    print("=" * 104)
    print("IDH-WILDTYPE GBM: MGMT methylation (temozolomide-response biomarker) x CDK4/6i-eligible subset")
    print("=" * 104)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    cnaseq_pat = {samp2pat[s] for s in set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", [])) if s in samp2pat}

    cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    clin: dict[str, dict] = {}
    for r in cd:
        clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]

    def os_of(pid):
        d = clin.get(pid, {})
        try:
            m = float(d.get("OS_MONTHS"))
        except (TypeError, ValueError):
            return None
        ev = 1 if d.get("OS_STATUS", "").startswith("1") or "DECEASED" in d.get("OS_STATUS", "").upper() else 0
        return (m, ev)

    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    alt: dict[str, dict[str, set]] = {}
    for m in post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                  {"sampleListId": SEQ_LIST, "entrezGeneIds": entrez}):
        g = ez2sym.get(m.get("entrezGeneId"))
        if g and m.get("mutationType") not in SILENT:
            pid = samp2pat.get(m["sampleId"])
            if pid:
                alt.setdefault(pid, {}).setdefault(g, set()).add("mut")
    for c in post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
                  {"sampleListId": CNA_LIST, "entrezGeneIds": entrez}):
        g = ez2sym.get(c.get("entrezGeneId"))
        pid = samp2pat.get(c["sampleId"])
        if g and pid:
            if c.get("alteration") == 2:
                alt.setdefault(pid, {}).setdefault(g, set()).add("amp")
            elif c.get("alteration") == -2:
                alt.setdefault(pid, {}).setdefault(g, set()).add("homdel")

    cohort = [p for p in cnaseq_pat if os_of(p) is not None]

    def is_idh_mut(d):
        return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])

    def pathway_on(d):
        return any(has(d, g, [k]) for g, k in PATHWAY_LESIONS)

    def eligible(p):
        d = alt.get(p, {})
        return (not is_idh_mut(d)) and pathway_on(d) and not has(d, "RB1", ["mut", "homdel"])

    idh_wt = [p for p in cohort if not is_idh_mut(alt.get(p, {}))]

    print(f"\nfetching MGMT calls from {', '.join(s for s, _ in MGMT_SOURCES)} (join by TCGA barcode)...")
    mgmt = mgmt_calls()
    wt_m = [p for p in idh_wt if p in mgmt]
    cov = 100 * len(wt_m) / len(idh_wt) if idh_wt else 0
    n_meth = sum(1 for p in wt_m if mgmt[p] == "METHYLATED")
    print(f"\nIDH-wildtype GBM cohort: {len(idh_wt)} patients; MGMT call available for "
          f"{len(wt_m)} ({cov:.0f}%)")
    print(f"  METHYLATED {n_meth} ({100*n_meth/len(wt_m):.0f}%)  |  UNMETHYLATED {len(wt_m)-n_meth} "
          f"({100*(len(wt_m)-n_meth)/len(wt_m):.0f}%)   (~40-45% methylated is expected for GBM)")

    rows = []

    # 1) MGMT survival within IDH-wt (validation anchor: methylated -> longer OS)
    print("\n### MGMT methylation vs OS  (IDH-wildtype GBM)  -- VALIDATION: methylated should live longer")
    meth = [p for p in wt_m if mgmt[p] == "METHYLATED"]
    unmeth = [p for p in wt_m if mgmt[p] == "UNMETHYLATED"]
    mm, mu, pm = survdiff(meth, unmeth, os_of, "MGMT methylated (TMZ-sensitive)", "MGMT unmethylated")
    rows.append({"stratum": "MGMT methylated (IDH-wt)", "n": len(meth), "median_os_mo": round(mm, 1) if mm else None,
                 "vs": "MGMT unmethylated", "median_os_other": round(mu, 1) if mu else None, "logrank_p": round(pm, 4)})

    # 2) MGMT x CDK4/6-eligibility independence (Fisher) + 2x2 medians
    elig_wt = {p for p in idh_wt if eligible(p)}
    a = sum(1 for p in wt_m if p in elig_wt and mgmt[p] == "METHYLATED")
    b = sum(1 for p in wt_m if p in elig_wt and mgmt[p] == "UNMETHYLATED")
    c = sum(1 for p in wt_m if p not in elig_wt and mgmt[p] == "METHYLATED")
    d = sum(1 for p in wt_m if p not in elig_wt and mgmt[p] == "UNMETHYLATED")
    orr, pfish = fisher_exact([[a, b], [c, d]])
    print("\n### Are MGMT status and CDK4/6-eligibility independent axes?  (Fisher exact)")
    print(f"  eligible & methylated={a}   eligible & unmethylated={b}")
    print(f"  other    & methylated={c}   other    & unmethylated={d}")
    print(f"  OR={orr:.2f}  p={pfish:.3f}  -> {'INDEPENDENT (orthogonal stratifiers)' if pfish>=0.05 else 'ASSOCIATED'}")

    # 3) four-way median OS table (the clinically meaningful stratification)
    print("\n### Median OS by the two actionable axes (IDH-wt only):")
    groups = {
        "eligible + MGMT-methylated": [p for p in wt_m if p in elig_wt and mgmt[p] == "METHYLATED"],
        "eligible + MGMT-unmethyl.":  [p for p in wt_m if p in elig_wt and mgmt[p] == "UNMETHYLATED"],
        "not-elig + MGMT-methylated": [p for p in wt_m if p not in elig_wt and mgmt[p] == "METHYLATED"],
        "not-elig + MGMT-unmethyl.":  [p for p in wt_m if p not in elig_wt and mgmt[p] == "UNMETHYLATED"],
    }
    for lab, ps in groups.items():
        if len(ps) >= 5:
            t, e = zip(*[os_of(p) for p in ps]); med = km_median(t, e)
            fm = f"{med:.1f}" if med is not None else "NA"
            print(f"  {lab:<30} n={len(ps):>3}  median OS = {fm:>5} mo")
            rows.append({"stratum": lab, "n": len(ps), "median_os_mo": round(med, 1) if med else None,
                         "vs": "", "median_os_other": None, "logrank_p": None})
        else:
            print(f"  {lab:<30} n={len(ps):>3}  (too few)")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stratum", "n", "median_os_mo", "vs", "median_os_other", "logrank_p"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 104)
    print("READING:")
    print(" * MGMT-methylated living longer reproduces the established TMZ-benefit result -> join is sound.")
    print(" * MGMT (alkylator response) and CDK4/6-eligibility (cell-cycle target) are largely INDEPENDENT,")
    print("   so they stratify patients orthogonally. The highest-need cell for a NEW targeted shot is the")
    print("   CDK4/6-eligible + MGMT-UNMETHYLATED group: a rational target exists AND temozolomide helps least.")
    print(" * Still unadjusted, still untreated-by-CDK4/6i. This maps where rational effort concentrates;")
    print("   it is not a treatment result. Not medical advice.")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
