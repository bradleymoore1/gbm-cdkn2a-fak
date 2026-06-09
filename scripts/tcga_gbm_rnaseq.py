#!/usr/bin/env python3
"""TCGA GBM: RNA-seq expression in CDK4/6-eligible vs other, and FAK/integrin pathway.

Two questions:
 1. Is the integrin-FAK-Arp2/3 pathway (top DepMap synthetic-lethal signal) transcriptionally
    upregulated in CDKN2A/B-deleted GBM tumors? If yes, the CRISPR dependency is corroborated.
 2. What are the most differentially expressed genes between:
      a) CDK4/6-eligible + MGMT-unmethylated vs. everything else (the clinical target population)
      b) CDKN2A/B-null vs. CDKN2A/B-intact (within IDH-wt)

Uses: gbm_tcga_pan_can_atlas_2018_rna_seq_v2_mrna (bulk RNA-seq, log2(RPKM+1))
API: cBioPortal /molecular-profiles/{profile}/molecular-data/fetch  POST with sampleIds + entrezGeneIds
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE = f"{STUDY}_mutations"
CNA_PROFILE = f"{STUDY}_gistic"
RNA_PROFILE = f"{STUDY}_rna_seq_v2_mrna"
SEQ_LIST = f"{STUDY}_sequenced"
CNA_LIST = f"{STUDY}_cna"
CNASEQ_LIST = f"{STUDY}_cnaseq"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_rnaseq.csv"

SILENT = {"Silent","Synonymous","3'UTR","5'UTR","3'Flank","5'Flank","Intron","RNA","IGR"}
DRIVER_GENES = ["IDH1","IDH2","CDKN2A","CDKN2B","CDK4","CDK6","CCND2","RB1"]
MGMT_SOURCES = [("lgggbm_tcga_pub","MGMT_PROMOTER_STATUS"),("gbm_tcga_pub2013","MGMT_STATUS")]

# Target gene sets for differential expression
FAK_INTEGRIN = ["ITGAV","ITGB5","FERMT2","TLN1","VCL","PTK2","ILK","RAC1",
                "ACTR2","ACTR3","ARPC2","ARPC3","ARPC4","CRK"]
TAZ_HIPPO    = ["WWTR1","YAP1","TEAD1","TEAD4","CYR61","CTGF","AMOTL2"]
AP1          = ["JUN","JUNB","JUND","FOSL1","FOSL2","FOS","FOSB"]
CELL_CYCLE   = ["CDK4","CDK6","CCND1","CCND2","CCND3","RB1","E2F1","E2F2","E2F3","CDKN1A","CDKN1B"]
BET_CHROM    = ["BRD2","BRD4","MYC","ACTL6A"]

ALL_TARGET_GENES = list(set(FAK_INTEGRIN + TAZ_HIPPO + AP1 + CELL_CYCLE + BET_CHROM))


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
                calls[pid] = v
    return calls


def main():
    print("=" * 100)
    print("TCGA GBM: RNA-seq differential expression — FAK/integrin pathway in CDKN2A/B-null tumors")
    print("=" * 100)

    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}
    pat2samp = {}
    for s in samples:
        pat2samp.setdefault(s["patientId"], []).append(s["sampleId"])

    cnaseq_sids = set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
    cnaseq_pat = {samp2pat[s] for s in cnaseq_sids if s in samp2pat}

    cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    clin: dict[str, dict] = {}
    for r in cd:
        clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]

    def has_os(pid):
        d = clin.get(pid, {})
        try:
            float(d["OS_MONTHS"])
            return True
        except (KeyError, TypeError, ValueError):
            return False

    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", DRIVER_GENES)
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

    cohort = {p for p in cnaseq_pat if has_os(p)}
    mgmt = mgmt_calls(cohort)

    def is_idh_mut(d): return has(d, "IDH1", ["mut"]) or has(d, "IDH2", ["mut"])
    def cdkn2ab_null(d): return has(d, "CDKN2A", ["homdel"]) or has(d, "CDKN2B", ["homdel"])
    def pathway_on(d): return any(has(d, g, [k]) for g, k in
        [("CDKN2A","homdel"),("CDKN2B","homdel"),("CDK4","amp"),("CDK6","amp"),("CCND2","amp")])
    def eligible(d): return (not is_idh_mut(d)) and pathway_on(d) and not has(d, "RB1", ["mut","homdel"])

    idh_wt = {p for p in cohort if not is_idh_mut(alt.get(p, {}))}
    elig_pats = {p for p in idh_wt if eligible(alt.get(p, {}))}
    cdkn2ab_null_pats = {p for p in idh_wt if cdkn2ab_null(alt.get(p, {}))}

    print(f"IDH-wt cohort: {len(idh_wt)}  eligible: {len(elig_pats)}  CDKN2A/B-null: {len(cdkn2ab_null_pats)}")
    wt_with_mgmt = {p for p in idh_wt if p in mgmt}
    elig_unmeth = {p for p in wt_with_mgmt if p in elig_pats and mgmt[p] == "UNMETHYLATED"}
    print(f"  elig+MGMT-unmeth: {len(elig_unmeth)}  (clinical target population)")

    # Fetch RNA-seq for target genes
    rna_genes_resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", ALL_TARGET_GENES)
    rna_ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in rna_genes_resolved}
    rna_entrez = list(rna_ez2sym)
    print(f"\nFetching RNA-seq for {len(rna_entrez)} target genes...")

    # Get sample IDs for IDH-wt patients that have RNA-seq data
    # Use all samples in cnaseq list, then filter to those with RNA data
    rna_data_raw = post(
        f"/molecular-profiles/{RNA_PROFILE}/molecular-data/fetch?projection=SUMMARY",
        {"sampleListId": CNASEQ_LIST, "entrezGeneIds": rna_entrez})

    # Build: sample_id -> gene -> expression
    samp_expr: dict[str, dict[str, float]] = {}
    for r in rna_data_raw:
        sid = r.get("sampleId")
        g = rna_ez2sym.get(r.get("entrezGeneId"))
        val = r.get("value")
        if sid and g and val is not None:
            try:
                samp_expr.setdefault(sid, {})[g] = float(val)
            except (TypeError, ValueError):
                pass

    # Map to patients
    pat_expr: dict[str, dict[str, float]] = {}
    for sid, gdict in samp_expr.items():
        pid = samp2pat.get(sid)
        if pid:
            pat_expr[pid] = gdict

    print(f"Patients with RNA-seq data: {len(pat_expr)}")
    idh_wt_rna = {p for p in idh_wt if p in pat_expr}
    print(f"IDH-wt with RNA-seq: {len(idh_wt_rna)}")

    def compare_groups(groupA, groupB, label_a, label_b, genes):
        """Mann-Whitney U for each gene between two groups."""
        groupA_rna = {p for p in groupA if p in pat_expr}
        groupB_rna = {p for p in groupB if p in pat_expr}
        print(f"\n### {label_a} (n={len(groupA_rna)}) vs {label_b} (n={len(groupB_rna)})")
        results = []
        for gene in genes:
            a_vals = [pat_expr[p][gene] for p in groupA_rna if gene in pat_expr[p]]
            b_vals = [pat_expr[p][gene] for p in groupB_rna if gene in pat_expr[p]]
            if len(a_vals) < 5 or len(b_vals) < 5:
                continue
            delta = np.mean(a_vals) - np.mean(b_vals)
            try:
                _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            except ValueError:
                continue
            results.append({"gene": gene, "mean_A": np.mean(a_vals), "mean_B": np.mean(b_vals),
                             "delta": delta, "p": p})
        if not results:
            print("  no results (insufficient data)")
            return None
        res = pd.DataFrame(results)
        _, q, _, _ = multipletests(res["p"].values, method="fdr_bh")
        res["q"] = q
        res = res.sort_values("p")
        print(f"  {'Gene':<12} {'mean_A':>8} {'mean_B':>8} {'delta':>8} {'p':>10} {'q':>10}")
        for _, r in res.iterrows():
            flag = " ***" if r["q"] < 0.05 else (" *" if r["p"] < 0.05 else "")
            print(f"  {r['gene']:<12} {r['mean_A']:>8.2f} {r['mean_B']:>8.2f} {r['delta']:>8.2f} "
                  f"{r['p']:>10.4f} {r['q']:>10.4f}{flag}")
        return res

    # Comparison 1: CDKN2A/B-null vs intact (IDH-wt) — tests whether FAK pathway is upregulated
    cdkn2ab_intact_rna = idh_wt_rna - cdkn2ab_null_pats
    print("\n" + "=" * 80)
    print("COMPARISON 1: Does CDKN2A/B deletion upregulate integrin-FAK pathway at the RNA level?")
    print("(This is orthogonal validation of the DepMap CRISPR synthetic-lethality signal)")
    print("=" * 80)
    res1 = compare_groups(cdkn2ab_null_pats, cdkn2ab_intact_rna,
                          "CDKN2A/B-null (IDH-wt)", "CDKN2A/B-intact (IDH-wt)",
                          FAK_INTEGRIN + TAZ_HIPPO + AP1 + CELL_CYCLE)

    # Comparison 2: elig+unmeth vs rest within IDH-wt with MGMT call
    rest = wt_with_mgmt - elig_unmeth
    print("\n" + "=" * 80)
    print("COMPARISON 2: CDK4/6-eligible + MGMT-unmethylated vs other IDH-wt")
    print("(Characterizing the clinical target population transcriptomically)")
    print("=" * 80)
    res2 = compare_groups(elig_unmeth, rest,
                          "elig+unmeth", "other IDH-wt",
                          FAK_INTEGRIN + TAZ_HIPPO + AP1 + CELL_CYCLE + BET_CHROM)

    # Save combined
    rows = []
    for label, res in [("CDKN2A/B-null vs intact", res1), ("elig+unmeth vs other", res2)]:
        if res is not None:
            r = res.copy(); r["comparison"] = label; rows.append(r)
    if rows:
        pd.concat(rows).to_csv(OUT, index=False)
        print(f"\nwrote {OUT.name}")

    print("\n" + "=" * 100)
    print("READING:")
    print("  If FAK/integrin genes are UPREGULATED in CDKN2A/B-null tumors -> the DepMap CRISPR")
    print("    synthetic-lethality signal is corroborated by orthogonal RNA evidence.")
    print("  If YAP/TAZ (WWTR1) is upregulated in CDKN2A/B-null -> confirms the resistance mechanism.")
    print("  If AP-1 (JUN/FOSL1) is upregulated -> consistent with CDK4/6 hyperactivity -> E2F -> AP-1.")
    print("  *** = q<0.05 after BH correction within comparison.")


if __name__ == "__main__":
    main()
