#!/usr/bin/env python3
"""IDH-wildtype GBM: how big is the *rational* CDK4/6-inhibitor-eligible subset, and what is
its natural prognosis?

This is the concrete brick behind the CDK4/6-axis thesis. In primary IDH-wildtype GBM the most
defensible molecularly-selected shot is a CNS-penetrant CDK4/6 inhibitor (abemaciclib), but only
for tumors that meet ALL THREE conditions:
  (1) IDH-WILDTYPE        -- primary GBM (IDH-mutant is a different, better-prognosis disease).
  (2) CDK4/6 PATHWAY ON   -- a lesion that drives CDK4/6 kinase activity: CDKN2A or CDKN2B
                             homozygous deletion (loss of the INK4 brake), OR amplification of
                             CDK4 / CDK6 / CCND2.
  (3) RB1 INTACT          -- RB1 is the substrate downstream of CDK4/6; if RB1 is lost, inhibiting
                             CDK4/6 does nothing. RB1 loss is a hard negative-selection marker.

We quantify the eligible fraction of the TCGA-GBM cohort (and within IDH-wt), enumerate WHY the
rest are ineligible, and read the eligible subset's overall survival.

CRITICAL HONESTY: TCGA patients were NOT treated with CDK4/6 inhibitors (the data predates that
use). So the survival readout here is the *natural history* of the eligible molecular profile, a
baseline -- NOT evidence the drug helps. In fact CDKN2A/B deletion is a known poor-prognosis
marker, so the eligible group may show equal-or-WORSE untreated OS. The value of this analysis is
sizing the addressable, biologically-rational trial subset, not claiming benefit. Unadjusted
KM/log-rank (no Cox); age/subtype confound. Standard of care remains Stupp +/- TTFields.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2dist

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE, CNA_PROFILE = f"{STUDY}_mutations", f"{STUDY}_gistic"
SEQ_LIST, CNA_LIST, CNASEQ_LIST = f"{STUDY}_sequenced", f"{STUDY}_cna", f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_cdk_eligible.csv"

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}
GENES = ["IDH1", "IDH2", "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND2", "RB1"]
PATHWAY_LESIONS = [  # (gene, kind, human label) -- any one turns the CDK4/6 axis on
    ("CDKN2A", "homdel", "CDKN2A homozygous deletion"),
    ("CDKN2B", "homdel", "CDKN2B homozygous deletion"),
    ("CDK4", "amp", "CDK4 amplification"),
    ("CDK6", "amp", "CDK6 amplification"),
    ("CCND2", "amp", "CCND2 amplification"),
]


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


def main():
    print("=" * 104)
    print("IDH-WILDTYPE GBM: rational CDK4/6-inhibitor-eligible subset (pathway-ON + RB1-intact) + natural OS")
    print("Eligibility = IDH-wt AND (CDKN2A/B homdel or CDK4/CDK6/CCND2 amp) AND RB1 intact.")
    print("TCGA patients were NOT given CDK4/6i -> survival = natural history of the profile, NOT drug benefit.")
    print("=" * 104)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    cnaseq_samp = set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
    cnaseq_pat = {samp2pat[s] for s in cnaseq_samp if s in samp2pat}

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
        st = d.get("OS_STATUS", "")
        ev = 1 if st.startswith("1") or "DECEASED" in st.upper() else 0
        return (m, ev)

    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    alt: dict[str, dict[str, set]] = {}
    muts = post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                {"sampleListId": SEQ_LIST, "entrezGeneIds": entrez})
    for m in muts:
        g = ez2sym.get(m.get("entrezGeneId"))
        if g is None or m.get("mutationType") in SILENT:
            continue
        pid = samp2pat.get(m["sampleId"])
        if pid:
            alt.setdefault(pid, {}).setdefault(g, set()).add("mut")
    cnas = post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
                {"sampleListId": CNA_LIST, "entrezGeneIds": entrez})
    for c in cnas:
        g = ez2sym.get(c.get("entrezGeneId"))
        if g is None:
            continue
        pid = samp2pat.get(c["sampleId"])
        if not pid:
            continue
        if c.get("alteration") == 2:
            alt.setdefault(pid, {}).setdefault(g, set()).add("amp")
        elif c.get("alteration") == -2:
            alt.setdefault(pid, {}).setdefault(g, set()).add("homdel")

    cohort = [p for p in cnaseq_pat if os_of(p) is not None]
    N = len(cohort)
    print(f"\nanalysis cohort (mutation + CNA + OS): {N} patients\n")

    # classify each patient
    def pathway_on(d):
        return any(has(d, g, [k]) for g, k, _ in PATHWAY_LESIONS)

    def is_idh_mut(d):
        return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])

    idh_wt = [p for p in cohort if not is_idh_mut(alt.get(p, {}))]
    idh_mut = [p for p in cohort if is_idh_mut(alt.get(p, {}))]
    eligible, rb1_lost_block, no_pathway = [], [], []
    for p in idh_wt:
        d = alt.get(p, {})
        if not pathway_on(d):
            no_pathway.append(p)
        elif has(d, "RB1", ["mut", "homdel"]):
            rb1_lost_block.append(p)
        else:
            eligible.append(p)

    pct = lambda n: f"{100*n/N:5.1f}%"
    pctw = lambda n: f"{100*n/len(idh_wt):5.1f}%" if idh_wt else "  NA"
    print("COHORT BREAKDOWN")
    print(f"  IDH-wildtype (primary GBM)      {len(idh_wt):>4}  ({pct(len(idh_wt))} of cohort)")
    print(f"  IDH-mutant (different disease)  {len(idh_mut):>4}  ({pct(len(idh_mut))} of cohort)  -- excluded from eligibility\n")
    print("WITHIN IDH-WILDTYPE -> CDK4/6i eligibility funnel")
    print(f"  CDK4/6 pathway ON               {len(idh_wt)-len(no_pathway):>4}  ({pctw(len(idh_wt)-len(no_pathway))} of IDH-wt)")
    print(f"    - of those, RB1 LOST (block)  {len(rb1_lost_block):>4}  -> ineligible (drug acts upstream of RB1)")
    print(f"    - RB1 intact  => ELIGIBLE     {len(eligible):>4}")
    print(f"  no CDK4/6 pathway lesion        {len(no_pathway):>4}  ({pctw(len(no_pathway))} of IDH-wt)  -- no rational target here\n")
    print(f">>> RATIONAL CDK4/6i-ELIGIBLE SUBSET: {len(eligible)} patients "
          f"= {pct(len(eligible))} of full cohort, {pctw(len(eligible))} of IDH-wildtype GBM\n")

    # which lesion qualifies the eligible patients
    print("Qualifying lesion among the ELIGIBLE (a tumor can carry more than one):")
    for g, k, lab in PATHWAY_LESIONS:
        n = sum(1 for p in eligible if has(alt.get(p, {}), g, [k]))
        print(f"  {lab:<32} {n:>4}  ({100*n/len(eligible):4.1f}% of eligible)" if eligible else f"  {lab}")

    # natural-history survival: eligible vs other IDH-wt
    print("\nNATURAL-HISTORY OVERALL SURVIVAL (untreated by CDK4/6i; baseline only)")
    other_wt = [p for p in idh_wt if p not in set(eligible)]
    rows = []
    if len(eligible) >= 5 and len(other_wt) >= 5:
        te, ee = zip(*[os_of(p) for p in eligible])
        to, eo = zip(*[os_of(p) for p in other_wt])
        me, mo = km_median(te, ee), km_median(to, eo)
        _, pval = logrank(te, ee, to, eo)
        fe = f"{me:.1f}" if me is not None else "NA"
        fo = f"{mo:.1f}" if mo is not None else "NA"
        print(f"  eligible (IDH-wt, pathway-on, RB1-intact)  n={len(eligible):>3}  median OS = {fe} mo")
        print(f"  other IDH-wt                               n={len(other_wt):>3}  median OS = {fo} mo")
        print(f"  log-rank p = {pval:.4f}")
        print("  (If eligible OS is similar-or-worse, that is EXPECTED -- CDKN2A/B loss is a poor-prognosis")
        print("   marker. It does NOT argue against the drug; these patients never received it.)")
        rows.append({"subset": "CDK4/6i-eligible (IDH-wt, pathway-on, RB1-intact)", "n": len(eligible),
                     "median_os_mo": round(me, 1) if me else None, "compare": "vs other IDH-wt",
                     "median_os_other": round(mo, 1) if mo else None, "logrank_p": round(pval, 4)})
    # IDH anchor sanity check (must show big benefit)
    if len(idh_mut) >= 5:
        tm, em = zip(*[os_of(p) for p in idh_mut])
        twt, ewt = zip(*[os_of(p) for p in idh_wt])
        mm, mwt = km_median(tm, em), km_median(twt, ewt)
        _, pidh = logrank(tm, em, twt, ewt)
        print(f"\n  [sanity anchor] IDH-mut median OS = {mm:.1f} mo vs IDH-wt {mwt:.1f} mo, "
              f"log-rank p = {pidh:.2e} (must be large -> code trustworthy)")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subset", "n", "median_os_mo", "compare",
                                           "median_os_other", "logrank_p"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 104)
    print("BOTTOM LINE: this sizes the addressable, biologically-rational CDK4/6i trial subset in")
    print("IDH-wildtype GBM and reads its untreated baseline survival. It is a TARGETING fraction,")
    print("not a treatment result. A real test still needs RB1-intact confirmation + a CNS-penetrant")
    print("agent (abemaciclib) in a biomarker-selected trial. Not medical advice.")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
