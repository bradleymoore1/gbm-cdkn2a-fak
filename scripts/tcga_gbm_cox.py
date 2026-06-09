#!/usr/bin/env python3
"""IDH-wildtype GBM: age-adjusted Cox proportional hazards model.

Every survival result in tcga_gbm_cdk_eligible.py and tcga_gbm_mgmt.py was UNADJUSTED.
Age is GBM's dominant prognostic confounder (older patients do worse regardless of molecular
profile). This script closes that gap with a proper multivariable Cox model:

  OS ~ age + MGMT_methylated + CDK4/6_eligible   (within IDH-wildtype)

If MGMT and CDK4/6-eligibility retain independent signal after age adjustment, the stratification
is real. If they disappear, the unadjusted results were age artifacts.

Also runs the full-cohort model including IDH status as a covariate (sanity: IDH-mut must show
protective HR, large effect, consistent with the log-rank anchor).

Requires: lifelines (pip install lifelines)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE, CNA_PROFILE = f"{STUDY}_mutations", f"{STUDY}_gistic"
SEQ_LIST, CNA_LIST, CNASEQ_LIST = f"{STUDY}_sequenced", f"{STUDY}_cna", f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_cox.csv"

SILENT = {"Silent", "Synonymous", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "Intron", "RNA", "IGR"}
GENES = ["IDH1", "IDH2", "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND2", "RB1"]
MGMT_SOURCES = [("lgggbm_tcga_pub", "MGMT_PROMOTER_STATUS"), ("gbm_tcga_pub2013", "MGMT_STATUS")]
PATHWAY = [("CDKN2A", "homdel"), ("CDKN2B", "homdel"),
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


def mgmt_calls(cohort_pids):
    calls = {}
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
            if pid and pid not in calls:
                calls[pid] = 1 if v == "METHYLATED" else 0
    return calls


def main():
    print("=" * 100)
    print("IDH-wildtype GBM: age-adjusted Cox PH model (lifelines)")
    print("OS ~ age + MGMT_methylated + CDK4/6_eligible   (within IDH-wt)")
    print("=" * 100)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    cnaseq_pat = {samp2pat[s] for s in set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
                  if s in samp2pat}

    cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    clin: dict[str, dict] = {}
    for r in cd:
        clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]

    def os_age_of(pid):
        d = clin.get(pid, {})
        try:
            m = float(d["OS_MONTHS"])
            age = float(d["AGE"])
        except (KeyError, TypeError, ValueError):
            return None
        ev = 1 if d.get("OS_STATUS", "").startswith("1") or "DECEASED" in d.get("OS_STATUS", "").upper() else 0
        return (m, ev, age)

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

    cohort = {p for p in cnaseq_pat if os_age_of(p) is not None}
    print(f"patients with cnaseq + OS + age: {len(cohort)}")

    mgmt = mgmt_calls(cohort)

    def is_idh_mut(d): return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])
    def pathway_on(d): return any(has(d, g, [k]) for g, k in PATHWAY)
    def eligible(d): return (not is_idh_mut(d)) and pathway_on(d) and not has(d, "RB1", ["mut", "homdel"])

    rows = []
    for p in cohort:
        oa = os_age_of(p)
        if oa is None:
            continue
        m, ev, age = oa
        d = alt.get(p, {})
        rows.append({
            "patient": p, "T": m, "E": ev, "age": age,
            "idh_mut": int(is_idh_mut(d)),
            "mgmt_methylated": mgmt.get(p, float("nan")),
            "cdk46_eligible": int(eligible(d)),
        })
    df = pd.DataFrame(rows).set_index("patient")
    print(f"full cohort df: {len(df)}")

    def run_cox(data, label, show_cols):
        print(f"\n### {label}  (n={len(data)})")
        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col="T", event_col="E")
            cph.print_summary(columns=["coef", "exp(coef)", "exp(coef) lower 95%",
                                       "exp(coef) upper 95%", "p"])
            print(f"  concordance: {cph.concordance_index_:.3f}")
            return cph
        except Exception as e:
            print(f"  FAILED: {e}")
            return None

    # Model 1: full cohort, IDH + age + MGMT + eligibility
    full = df.dropna(subset=["mgmt_methylated"])
    run_cox(full[["T", "E", "age", "idh_mut", "mgmt_methylated", "cdk46_eligible"]],
            "FULL COHORT (IDH-wt + IDH-mut): OS ~ age + IDH_mut + MGMT_methylated + CDK4/6_eligible",
            ["age", "idh_mut", "mgmt_methylated", "cdk46_eligible"])

    # Model 2: IDH-wt only (the clinically relevant model for treatment-selection)
    idh_wt_df = df[df["idh_mut"] == 0].drop(columns=["idh_mut"])
    wt_complete = idh_wt_df.dropna(subset=["mgmt_methylated"])
    cph2 = run_cox(wt_complete[["T", "E", "age", "mgmt_methylated", "cdk46_eligible"]],
                   "IDH-WILDTYPE ONLY: OS ~ age + MGMT_methylated + CDK4/6_eligible",
                   ["age", "mgmt_methylated", "cdk46_eligible"])

    # Model 3: within IDH-wt, CDK4/6 eligibility only (unadjusted for MGMT) for comparison
    run_cox(idh_wt_df.dropna(subset=["T", "E", "age"])[["T", "E", "age", "cdk46_eligible"]],
            "IDH-WILDTYPE: OS ~ age + CDK4/6_eligible  (MGMT-agnostic)",
            ["age", "cdk46_eligible"])

    if cph2 is not None:
        res = cph2.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
        res.columns = ["covariate", "coef", "HR", "HR_lo95", "HR_hi95", "p"]
        res.to_csv(OUT, index=False)

    print("\n" + "=" * 100)
    print("READING:")
    print(" * age HR > 1 (each year older -> worse OS) = internal sanity check.")
    print(" * MGMT_methylated HR < 1 = methylated is protective (expected, TMZ-benefit).")
    print(" * CDK4/6_eligible HR ~ 1 = eligible (CDKN2A/B-del) has similar or slightly worse UNTREATED OS")
    print("   (expected -- CDKN2A/B loss is poor-prognosis; these patients never received CDK4/6i in this study).")
    print(" * If MGMT effect survives age adjustment -> the biomarker signal is real.")
    print(" * Concordance > 0.6 = model has meaningful discriminatory ability.")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
