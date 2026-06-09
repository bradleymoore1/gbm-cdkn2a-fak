#!/usr/bin/env python3
"""Two analyses in one pass (shared API calls):

 A. Mutational signatures proxy — SBS18 (oxidative damage / MUTYH-deficiency):
    SBS18 is predominantly C>A transversions. We can't compute exact SBS18 decomposition
    without trinucleotide reference context, but we CAN compute:
      - C>A fraction per patient (crude SBS18 proxy)
      - Overall C>A enrichment in CDKN2A/B-null vs intact tumors
    If MUTYH deficiency is genuinely contributing, null tumors should show higher C>A load.

 B. CDK4/6i resistance co-alteration landscape:
    Of the 244 CDK4/6i-eligible IDH-wt GBMs, what fraction already have co-alterations
    that predict resistance to CDK4/6i (from breast cancer CDK4/6i resistance literature)?
    Known resistance mechanisms:
      - CCNE1 amplification → CDK2 bypass (most common in breast)
      - CDK2 amplification
      - PIK3CA/PIK3R1 activating mutation → PI3K/mTOR escape
      - YAP1 amplification → Hippo bypass
      - FGFR1/2 amplification → RTK bypass
      - RB1 loss (already excluded in eligibility but double-check)
      - MYC amplification → E2F-independent proliferation
    We already know CDKN2A/B status, CDK4/CDK6/CCND2/RB1 from previous analysis.
    This extends to the resistance set.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact

API = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
MUT_PROFILE  = f"{STUDY}_mutations"
CNA_PROFILE  = f"{STUDY}_gistic"
SEQ_LIST     = f"{STUDY}_sequenced"
CNA_LIST     = f"{STUDY}_cna"
CNASEQ_LIST  = f"{STUDY}_cnaseq"
OUT_SIG  = Path.home() / "finngen-triage" / "tcga_gbm_signature_proxy.csv"
OUT_RES  = Path.home() / "finngen-triage" / "tcga_gbm_resistance.csv"

SILENT = {"Silent","Synonymous","3'UTR","5'UTR","3'Flank","5'Flank","Intron","RNA","IGR"}

# Genes for CDK4/6i eligibility + resistance
DRIVER_GENES    = ["IDH1","IDH2","CDKN2A","CDKN2B","CDK4","CDK6","CCND2","RB1"]
RESISTANCE_GENES = ["CCNE1","CDK2","PIK3CA","PIK3R1","YAP1","FGFR1","FGFR2","MYC",
                    "PTEN","TP53","EGFR","AKT1","AKT2","AKT3","MTOR","KRAS","NF1"]
ALL_GENES = list(set(DRIVER_GENES + RESISTANCE_GENES))

MGMT_SOURCES = [("lgggbm_tcga_pub","MGMT_PROMOTER_STATUS"),("gbm_tcga_pub2013","MGMT_STATUS")]


def get(path):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(API + path, headers={"Accept":"application/json"}), timeout=120))

def post(path, body):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Accept":"application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=180))

def has(d, g, kinds):
    return bool(d.get(g, set()) & set(kinds))


def main():
    print("=" * 100)
    print("TCGA GBM: (A) Mutational signature proxy (SBS18/C>A) + (B) CDK4/6i resistance co-alterations")
    print("=" * 100)

    # ── Setup ──────────────────────────────────────────────────────────────────
    samples = get(f"/studies/{STUDY}/samples?pageSize=100000")
    samp2pat = {s["sampleId"]: s["patientId"] for s in samples}

    cnaseq_sids = set(get(f"/sample-lists/{CNASEQ_LIST}").get("sampleIds", []))
    cnaseq_pat  = {samp2pat[s] for s in cnaseq_sids if s in samp2pat}

    cd = get(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&pageSize=2000000")
    clin: dict[str,dict] = {}
    for r in cd: clin.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]

    def has_os(pid):
        try: float(clin.get(pid,{})["OS_MONTHS"]); return True
        except: return False

    cohort = {p for p in cnaseq_pat if has_os(p)}

    # ── Gene alteration data ───────────────────────────────────────────────────
    resolved = post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", ALL_GENES)
    ez2sym = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in resolved}
    entrez = list(ez2sym)

    alt: dict[str,dict[str,set]] = {}
    # mutations
    mut_list = post(f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
                    {"sampleListId": SEQ_LIST, "entrezGeneIds": entrez})
    for m in mut_list:
        g   = ez2sym.get(m.get("entrezGeneId"))
        pid = samp2pat.get(m["sampleId"])
        if g and pid and m.get("mutationType") not in SILENT:
            alt.setdefault(pid,{}).setdefault(g,set()).add("mut")
    # CNA
    for c in post(f"/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
                  {"sampleListId": CNA_LIST, "entrezGeneIds": entrez}):
        g   = ez2sym.get(c.get("entrezGeneId"))
        pid = samp2pat.get(c["sampleId"])
        if g and pid:
            if c.get("alteration") ==  2: alt.setdefault(pid,{}).setdefault(g,set()).add("amp")
            elif c.get("alteration") == -2: alt.setdefault(pid,{}).setdefault(g,set()).add("homdel")

    # ── Patient classification ─────────────────────────────────────────────────
    def is_idh_mut(d):  return has(d,"IDH1",["mut"]) or has(d,"IDH2",["mut"])
    def cdkn2ab_null(d): return has(d,"CDKN2A",["homdel"]) or has(d,"CDKN2B",["homdel"])
    def pathway_on(d):
        return any(has(d,g,[k]) for g,k in
                   [("CDKN2A","homdel"),("CDKN2B","homdel"),("CDK4","amp"),("CDK6","amp"),("CCND2","amp")])
    def eligible(d): return (not is_idh_mut(d)) and pathway_on(d) and not has(d,"RB1",["mut","homdel"])

    idh_wt = {p for p in cohort if not is_idh_mut(alt.get(p,{}))}
    null_pats  = {p for p in idh_wt if cdkn2ab_null(alt.get(p,{}))}
    elig_pats  = {p for p in idh_wt if eligible(alt.get(p,{}))}

    # ── MGMT calls ────────────────────────────────────────────────────────────
    mgmt = {}
    for study, attr in MGMT_SOURCES:
        smp = get(f"/studies/{study}/samples?pageSize=100000")
        s2p = {s["sampleId"]: s["patientId"] for s in smp}
        data = get(f"/studies/{study}/clinical-data?clinicalDataType=SAMPLE&attributeId={attr}&pageSize=2000000")
        for r in data:
            if r["clinicalAttributeId"] != attr: continue
            v = (r["value"] or "").strip().upper()
            if v in ("METHYLATED","UNMETHYLATED"):
                pid = s2p.get(r["sampleId"])
                if pid and pid not in mgmt: mgmt[pid] = v
    elig_unmeth = {p for p in elig_pats if mgmt.get(p) == "UNMETHYLATED"}

    print(f"cohort={len(cohort)}  IDH-wt={len(idh_wt)}  CDKN2A/B-null={len(null_pats)}")
    print(f"CDK4/6i-eligible={len(elig_pats)}  elig+unmeth={len(elig_unmeth)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # A. MUTATIONAL SIGNATURE PROXY (C>A / SBS18 proxy)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("A. SBS18 proxy: C>A transversion fraction (MUTYH/oxidative damage signature)")
    print("   Fetching ALL non-silent mutations from TCGA GBM...")
    print("=" * 80)

    # Fetch all mutations (no gene filter = pass empty list or use a broad fetch)
    # cBioPortal requires entrezGeneIds in the POST body; for all genes, use the
    # sample-list endpoint which returns pre-fetched data in chunks.
    # Strategy: fetch mutations for all sequenced samples, all genes (omit entrezGeneIds = all)
    all_muts = post(
        f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=SUMMARY",
        {"sampleListId": SEQ_LIST})  # no entrezGeneIds = all genes

    print(f"  Total mutation records fetched: {len(all_muts)}")

    # Tally per patient: total mutations, C>A (G in ref, T in alt OR C in ref, A in alt)
    pat_mut: dict[str, dict] = {}
    TRANSITION = {"C>T","G>A","A>G","T>C"}  # common transitions
    for m in all_muts:
        pid = samp2pat.get(m.get("sampleId"))
        if not pid or pid not in idh_wt: continue
        ref = (m.get("referenceAllele") or "").upper()
        alt_a = (m.get("variantAllele") or m.get("proteinChange") or "").upper()
        mtype = m.get("mutationType","")
        if mtype in SILENT: continue

        # Normalise to pyrimidine strand: C>A and G>T are the same; also capture as string
        change = None
        if   ref == "C" and alt_a == "A": change = "C>A"
        elif ref == "G" and alt_a == "T": change = "C>A"   # complement
        elif ref == "C" and alt_a == "T": change = "C>T"
        elif ref == "G" and alt_a == "A": change = "C>T"
        elif ref == "C" and alt_a == "G": change = "C>G"
        elif ref == "G" and alt_a == "C": change = "C>G"
        elif ref == "T" and alt_a == "A": change = "T>A"
        elif ref == "A" and alt_a == "T": change = "T>A"
        elif ref == "T" and alt_a == "C": change = "T>C"
        elif ref == "A" and alt_a == "G": change = "T>C"
        elif ref == "T" and alt_a == "G": change = "T>G"
        elif ref == "A" and alt_a == "C": change = "T>G"
        else: change = "other"

        rec = pat_mut.setdefault(pid, {"total":0,"C>A":0,"C>T":0,"C>G":0,"T>A":0,"T>C":0,"T>G":0})
        rec["total"] += 1
        if change: rec[change] = rec.get(change,0) + 1

    print(f"  Patients with mutation data (IDH-wt): {len(pat_mut)}")

    # Compute C>A fraction
    rows_sig = []
    for pid, rec in pat_mut.items():
        tot = rec["total"]
        if tot < 5: continue
        ca_frac = rec.get("C>A",0) / tot
        rows_sig.append({
            "patient": pid,
            "total_muts": tot,
            "CA_count": rec.get("C>A",0),
            "CA_fraction": round(ca_frac,4),
            "CT_fraction": round(rec.get("C>T",0)/tot,4),
            "cdkn2ab_null": int(pid in null_pats),
            "eligible":     int(pid in elig_pats),
        })
    df_sig = pd.DataFrame(rows_sig)

    null_ca  = df_sig[df_sig["cdkn2ab_null"]==1]["CA_fraction"].values
    intact_ca = df_sig[df_sig["cdkn2ab_null"]==0]["CA_fraction"].values
    print(f"\n  CDKN2A/B-null: n={len(null_ca)}  median C>A fraction={np.median(null_ca):.3f}")
    print(f"  CDKN2A/B-intact: n={len(intact_ca)}  median C>A fraction={np.median(intact_ca):.3f}")
    if len(null_ca) >= 5 and len(intact_ca) >= 5:
        _, p = mannwhitneyu(null_ca, intact_ca, alternative="two-sided")
        delta = np.median(null_ca) - np.median(intact_ca)
        print(f"  delta (null-intact) = {delta:+.4f}   Mann-Whitney p = {p:.4f}")
        if p < 0.05:
            print("  SIGNIFICANT — CDKN2A/B-null GBMs show different C>A burden (SBS18-consistent)")
        else:
            print("  Not significant — no C>A enrichment in CDKN2A/B-null vs intact")

    # Also: are TMB-high tumors enriched in C>A? (hypermutation is SBS11/TMZ, not SBS18)
    print("\n  Mutation type breakdown (all IDH-wt GBM):")
    for col in ["C>A","C>T","C>G","T>A","T>C","T>G"]:
        cnt_col = col.replace(">","") + "_count"
        tot_col = df_sig["CA_count"].sum() * 0  # trick
        frac_mean = df_sig["CA_fraction"].mean() if col == "C>A" else df_sig[col.replace(">","_frac") if col.replace(">","_frac") in df_sig.columns else "CT_fraction"].mean()
    # Simpler: compute overall fractions
    total_all = df_sig["total_muts"].sum()
    ca_all = df_sig["CA_count"].sum()
    print(f"  C>A overall: {ca_all}/{total_all} = {100*ca_all/total_all:.1f}% of all non-silent mutations")

    df_sig.to_csv(OUT_SIG, index=False)
    print(f"\n  wrote {OUT_SIG.name}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # B. CDK4/6i RESISTANCE CO-ALTERATION LANDSCAPE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("B. CDK4/6i resistance co-alteration landscape in CDK4/6i-eligible IDH-wt GBM")
    print("   (applying breast cancer CDK4/6i resistance mechanisms to this population)")
    print("=" * 80)

    RESISTANCE_EVENTS = {
        "CCNE1_amp":    ("CCNE1", "amp",    "CDK2 bypass — most common breast CDK4/6i resistance"),
        "PIK3CA_mut":   ("PIK3CA","mut",    "PI3K/mTOR escape — second most common"),
        "PIK3R1_mut":   ("PIK3R1","mut",    "PI3K pathway (PI3K regulatory subunit)"),
        "YAP1_amp":     ("YAP1",  "amp",    "Hippo bypass — WWTR1/YAP1 our #3 DepMap hit"),
        "FGFR1_amp":    ("FGFR1", "amp",    "RTK bypass"),
        "FGFR2_amp":    ("FGFR2", "amp",    "RTK bypass"),
        "MYC_amp":      ("MYC",   "amp",    "E2F-independent proliferation"),
        "NF1_mut":      ("NF1",   "mut",    "RAS/MAPK escape (NF1 loss -> RAS active)"),
        "PTEN_mut":     ("PTEN",  "mut",    "PI3K pathway (PTEN loss)"),
        "PTEN_homdel":  ("PTEN",  "homdel", "PI3K pathway (PTEN loss)"),
        "KRAS_mut":     ("KRAS",  "mut",    "RAS/MAPK escape"),
        "AKT1_mut":     ("AKT1",  "mut",    "AKT activation"),
    }

    print(f"\n  CDK4/6i-eligible IDH-wt patients: {len(elig_pats)}")
    print(f"  elig+MGMT-unmethylated: {len(elig_unmeth)}")

    rows_res = []
    any_resist = set()

    print(f"\n  {'Resistance event':<20} {'N in elig':>10} {'%':>6}  {'in elig+unmeth':>14} {'%':>6}  {'Mechanism'}")
    print("  " + "-" * 90)

    for event, (gene, atype, mechanism) in RESISTANCE_EVENTS.items():
        n_elig  = sum(1 for p in elig_pats  if has(alt.get(p,{}), gene, [atype]))
        n_unm   = sum(1 for p in elig_unmeth if has(alt.get(p,{}), gene, [atype]))
        pct_e   = 100*n_elig  / len(elig_pats)   if elig_pats   else 0
        pct_u   = 100*n_unm   / len(elig_unmeth) if elig_unmeth else 0
        print(f"  {event:<20} {n_elig:>10}  {pct_e:>5.1f}%  {n_unm:>14}  {pct_u:>5.1f}%  {mechanism[:55]}")
        rows_res.append({"event":event,"gene":gene,"alt_type":atype,
                         "n_eligible":n_elig,"pct_eligible":round(pct_e,1),
                         "n_elig_unmeth":n_unm,"pct_elig_unmeth":round(pct_u,1),
                         "mechanism":mechanism})
        for p in elig_pats:
            if has(alt.get(p,{}), gene, [atype]):
                any_resist.add(p)

    # Compound: any resistance event
    n_any  = len(any_resist & elig_pats)
    n_any_u = len(any_resist & elig_unmeth)
    pct_any  = 100*n_any  / len(elig_pats)
    pct_any_u= 100*n_any_u/ len(elig_unmeth)
    print(f"\n  {'ANY resistance event':<20} {n_any:>10}  {pct_any:>5.1f}%  {n_any_u:>14}  {pct_any_u:>5.1f}%")
    print(f"  Patients eligible AND likely resistant (pre-treatment): {n_any}/{len(elig_pats)} ({pct_any:.0f}%)")

    clean_elig  = elig_pats  - any_resist
    clean_unmeth = elig_unmeth - any_resist
    print(f"\n  CLEAN eligible (no known resistance event): {len(clean_elig)}/{len(elig_pats)} ({100*len(clean_elig)/len(elig_pats):.0f}%)")
    print(f"  CLEAN elig+unmeth: {len(clean_unmeth)}/{len(elig_unmeth)} ({100*len(clean_unmeth)/len(elig_unmeth):.0f}%)")
    print(f"\n  → Realistic trial population = CLEAN elig+unmeth = {len(clean_unmeth)} patients")
    print(f"    = {100*len(clean_unmeth)/len(idh_wt):.0f}% of all IDH-wt GBM")

    pd.DataFrame(rows_res).to_csv(OUT_RES, index=False)
    print(f"\n  wrote {OUT_RES.name}")

    print("\n" + "=" * 100)
    print("READING:")
    print("A: If CDKN2A/B-null GBMs show elevated C>A fraction -> somatic MUTYH-deficiency")
    print("   signature supports germline MUTYH finding from FinnGen.")
    print("   C>T dominates in GBM (TMZ-induced, SBS11/SBS alkylating) so C>A fraction is diluted.")
    print("B: Pre-existing resistance co-alterations shrink the realistically responsive fraction.")
    print("   Clean elig+unmeth = the honest trial population size.")


if __name__ == "__main__":
    main()
