#!/usr/bin/env python3
"""GBM somatic drivers -> CLINICAL correlates: survival stratification + co-occurrence.

Connects the validated TCGA-GBM driver landscape to OUTCOMES. Two questions:
  (1) SURVIVAL: does carrying a given driver alteration change overall survival (OS)?
      Kaplan-Meier median OS + log-rank p, altered vs wild-type, per driver/pathway.
  (2) ARCHITECTURE: which drivers CO-OCCUR vs are MUTUALLY EXCLUSIVE (Fisher exact)?
      Mutual exclusivity => same pathway / redundant; co-occurrence => cooperating hits
      (combination-therapy rationale).

VALIDATION ANCHOR: IDH1-mutant tumors must show a LARGE OS benefit (IDH-mutant glioma is a
biologically distinct, better-prognosis disease). If we reproduce that, the survival code is
trustworthy -- same logic as validating the landscape against known frequencies.

HONEST CAVEATS:
  * No lifelines installed -> KM median + log-rank implemented directly (numpy/scipy). We do
    NOT compute a multivariable Cox model, so these are UNADJUSTED. AGE and transcriptomic
    SUBTYPE are dominant confounders in GBM; an unadjusted OS difference is hypothesis-level,
    not a causal/independent prognostic claim.
  * Analysis restricted to patients with BOTH mutation+CNA data AND OS (clean denominator).
  * MGMT promoter methylation (the key temozolomide-response biomarker) is NOT in this study's
    clinical attributes -> not included here (needs the methylation profile).
"""
from __future__ import annotations

import csv
import json
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2dist
from scipy.stats import fisher_exact

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE, CNA_PROFILE = f"{STUDY}_mutations", f"{STUDY}_gistic"
SEQ_LIST, CNA_LIST, CNASEQ_LIST = f"{STUDY}_sequenced", f"{STUDY}_cna", f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_clinical.csv"

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}

# genes to fetch (entrez resolved below)
GENES = ["IDH1", "TP53", "PTEN", "NF1", "RB1", "PIK3CA", "PIK3R1", "ATRX",
         "EGFR", "CDK4", "CDK6", "CCND2", "MDM2", "MDM4", "PDGFRA", "MET",
         "CDKN2A", "CDKN2B"]

# survival groups: label -> (predicate over a patient's alteration dict)
# alteration dict per patient: {gene: set of {'mut','amp','homdel'}}
def has(d, g, kinds):
    return bool(d.get(g, set()) & set(kinds))

SURV_GROUPS = {
    "IDH1 mutation (anchor)":      lambda d: has(d, "IDH1", ["mut"]),
    "EGFR amplification":          lambda d: has(d, "EGFR", ["amp"]),
    "EGFR mutation":               lambda d: has(d, "EGFR", ["mut"]),
    "CDKN2A/B homozygous del":     lambda d: has(d, "CDKN2A", ["homdel"]) or has(d, "CDKN2B", ["homdel"]),
    "TP53 mutation":               lambda d: has(d, "TP53", ["mut"]),
    "PTEN loss (mut/homdel)":      lambda d: has(d, "PTEN", ["mut", "homdel"]),
    "NF1 loss (mut/homdel)":       lambda d: has(d, "NF1", ["mut", "homdel"]),
    "RB1 loss (mut/homdel)":       lambda d: has(d, "RB1", ["mut", "homdel"]),
    "PDGFRA amplification":        lambda d: has(d, "PDGFRA", ["amp"]),
    "MDM2/4 amplification":        lambda d: has(d, "MDM2", ["amp"]) or has(d, "MDM4", ["amp"]),
    "CDK4/6 axis (amp or CDKN2A/B del)":
        lambda d: any(has(d, g, ["amp"]) for g in ("CDK4", "CDK6", "CCND2"))
                  or has(d, "CDKN2A", ["homdel"]) or has(d, "CDKN2B", ["homdel"]),
    "PI3K axis (PIK3CA/R1 mut or PTEN loss)":
        lambda d: has(d, "PIK3CA", ["mut"]) or has(d, "PIK3R1", ["mut"]) or has(d, "PTEN", ["mut", "homdel"]),
}

# genes for the co-occurrence matrix (any alteration)
COOCCUR_GENES = ["EGFR", "PTEN", "TP53", "NF1", "RB1", "PIK3CA", "PIK3R1", "PDGFRA",
                 "CDK4", "MDM2", "MDM4", "CDKN2A", "IDH1", "ATRX"]


def get(path):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(API + path, headers={"Accept": "application/json"}), timeout=120))


def post(path, body):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=180))


def km_median(times, events):
    """Kaplan-Meier median survival. times months, events 1=death/0=censor."""
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
    print("=" * 108)
    print("TCGA-GBM clinical correlates: survival stratification + driver co-occurrence")
    print("UNADJUSTED KM/log-rank (no Cox); AGE & SUBTYPE confound -> hypothesis-level, not independent prognosis.")
    print("=" * 108)

    # sample -> patient
    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    cnaseq_samp = set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
    cnaseq_pat = {samp2pat[s] for s in cnaseq_samp if s in samp2pat}

    # clinical (patient level)
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

    # resolve genes + fetch alterations
    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    alt: dict[str, dict[str, set]] = {}  # patient -> gene -> {kinds}
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

    # analysis cohort = cnaseq patients with OS
    cohort = [p for p in cnaseq_pat if os_of(p) is not None]
    print(f"survival cohort (both data types + OS): {len(cohort)} patients\n")

    # --- survival by group ---
    print(f"{'group':<42}{'n_alt':>6}{'n_wt':>6}{'med_alt':>9}{'med_wt':>8}{'logrank_p':>11}  signal")
    print("-" * 108)
    rows = []
    for label, pred in SURV_GROUPS.items():
        a = [p for p in cohort if pred(alt.get(p, {}))]
        w = [p for p in cohort if p not in set(a)]
        if len(a) < 5 or len(w) < 5:
            print(f"{label:<42}{len(a):>6}{len(w):>6}   (too few altered to test)")
            continue
        ta, ea = zip(*[os_of(p) for p in a]); tw, ew = zip(*[os_of(p) for p in w])
        ma, mw = km_median(ta, ea), km_median(tw, ew)
        _, p = logrank(ta, ea, tw, ew)
        ma = round(ma, 1) if ma is not None else None
        mw = round(mw, 1) if mw is not None else None
        better = (ma or 0) > (mw or 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        arrow = "longer OS" if better else "shorter OS"
        rows.append({"group": label, "n_alt": len(a), "n_wt": len(w),
                     "median_os_alt": ma, "median_os_wt": mw, "logrank_p": round(p, 5),
                     "direction": arrow})
        fa = f"{ma:.1f}" if ma is not None else "NA"
        fw = f"{mw:.1f}" if mw is not None else "NA"
        print(f"{label:<42}{len(a):>6}{len(w):>6}{fa:>9}{fw:>8}{p:>11.2e}  {sig:<3} {arrow if sig else ''}")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "n_alt", "n_wt", "median_os_alt",
                                           "median_os_wt", "logrank_p", "direction"])
        w.writeheader(); w.writerows(rows)

    # --- subtype median OS ---
    print("\n### Median OS by transcriptomic SUBTYPE")
    sub: dict[str, list] = {}
    for p in cohort:
        s = clin.get(p, {}).get("SUBTYPE") or "NA"
        sub.setdefault(s, []).append(p)
    for s, ps in sorted(sub.items(), key=lambda kv: -len(kv[1])):
        t, e = zip(*[os_of(p) for p in ps])
        med = km_median(t, e)
        print(f"  {s:<28} n={len(ps):>3}  median OS = {med:.1f} mo" if med is not None
              else f"  {s:<28} n={len(ps):>3}  median OS = NA")

    # --- co-occurrence / mutual exclusivity ---
    print("\n### Driver co-occurrence / mutual exclusivity (Fisher exact, cnaseq cohort)")
    altered = {g: {p for p in cohort if alt.get(p, {}).get(g)} for g in COOCCUR_GENES}
    N = len(cohort)
    pairs = []
    for g1, g2 in combinations(COOCCUR_GENES, 2):
        A, B = altered[g1], altered[g2]
        a = len(A & B); b = len(A - B); c = len(B - A); d = N - a - b - c
        if min(len(A), len(B)) < 8:
            continue
        orr, p = fisher_exact([[a, b], [c, d]])
        pairs.append((g1, g2, a, orr, p))
    co = sorted([x for x in pairs if x[3] > 1 and x[4] < 0.1], key=lambda x: x[4])[:8]
    ex = sorted([x for x in pairs if x[3] < 1 and x[4] < 0.1], key=lambda x: x[4])[:8]
    print("  CO-OCCURRING (cooperating hits; OR>1):")
    for g1, g2, a, orr, p in co:
        print(f"    {g1:>7} + {g2:<7}  both={a:<3} OR={orr:>4.2f} p={p:.3f}")
    print("  MUTUALLY EXCLUSIVE (same pathway / redundant; OR<1):")
    for g1, g2, a, orr, p in ex:
        print(f"    {g1:>7} x {g2:<7}  both={a:<3} OR={orr:>4.2f} p={p:.3f}")

    print("\n" + "=" * 108)
    print("READING: IDH1-mut should show markedly LONGER OS (validation). CDKN2A/B-del typically")
    print("SHORTER OS. Mutual exclusivity among RTKs (EGFR/PDGFRA) or p53-axis (TP53 vs MDM2/4)")
    print("reflects pathway redundancy. All UNADJUSTED -- age/subtype confound; treat as hypotheses.")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
