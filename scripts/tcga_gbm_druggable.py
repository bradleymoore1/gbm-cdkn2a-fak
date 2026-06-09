#!/usr/bin/env python3
"""GBM somatic drivers -> druggable targets (DGIdb) with HONEST GBM clinical reality.

Takes the validated TCGA-GBM driver landscape (tcga_gbm_somatic.py) and asks, per driver:
is there a drug, and -- crucially -- has it ever worked IN GBM? GBM is a trial graveyard.
A long DGIdb drug list is meaningless without two filters that kill most GBM drugs:
  (1) BLOOD-BRAIN BARRIER penetrance -- most TKIs/antibodies don't reach tumor at dose.
  (2) BIOMARKER SELECTION -- unselected trials failed; the few wins are subset-selected.

So we pull DGIdb gene->drug interactions, then overlay a curated, sourced GBM-reality note
per target. The goal is a truthful map of where leverage actually exists (mostly small,
biomarker-defined subsets), NOT a list of "targetable" genes that already failed in trials.

NOTHING here is a cure or a treatment recommendation. Standard of care remains maximal safe
resection + radiotherapy + temozolomide (Stupp) + tumor-treating fields; median OS ~15-20 mo.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

DGIDB = "https://dgidb.org/api/graphql"
LAND = Path.home() / "finngen-triage" / "tcga_gbm_somatic_landscape.csv"
OUT = Path.home() / "finngen-triage" / "tcga_gbm_druggable.csv"
MIN_ALTERED = 3.0  # only map drivers altered in >=3% of the cohort

# Curated, honest GBM clinical reality per target. status: LIVE (rational, biomarker-selected,
# CNS-active option exists), FAILED (tried in GBM, didn't work -- usually BBB/heterogeneity),
# SUBSET (works only in a rare molecular subset), BIOMARKER (not a drug target itself).
GBM_REALITY = {
    "EGFR": ("FAILED", "Most-attacked, least-conquered GBM target. EGFR TKIs (erlotinib/gefitinib/"
             "afatinib) failed-poor BBB penetration + heterogeneity. Depatux-m ADC failed ph3 "
             "(INTELLANCE); rindopepimut EGFRvIII vaccine failed ph3 (ACT IV). EGFRvIII CAR-T early, "
             "antigen escape."),
    "CDK4": ("LIVE", "CDK4/6 inhibitors (abemaciclib is CNS-penetrant; palbo/ribo). Rational when "
             "CDK4/6-amplified AND RB1 intact AND CDKN2A/B deleted. Among the more defensible live shots."),
    "CDK6": ("LIVE", "Same CDK4/6i axis (abemaciclib CNS-active). Requires intact RB1."),
    "CDKN2A": ("BIOMARKER", "Homozygous-deletion is the lesion (no direct drug); sensitizes to CDK4/6i "
               "and is a poor-prognosis marker. Target the pathway (CDK4/6), not the gene."),
    "CDKN2B": ("BIOMARKER", "Co-deleted with CDKN2A on 9p21; same CDK4/6-pathway rationale."),
    "MDM2": ("SUBSET", "MDM2 inhibitors (milademetan, navtemadlin/AMG-232) restore p53 -- ONLY rational "
             "in TP53-wild-type, MDM2-amplified tumors. Early-phase; CNS exposure a question."),
    "MDM4": ("SUBSET", "Same p53-restoration rationale as MDM2 in TP53-WT tumors; fewer clinical agents."),
    "PDGFRA": ("FAILED", "Imatinib/dasatinib and multi-TKIs failed in unselected GBM; the PDGFRA-amplified/"
               "mutant subset has rarely been tested prospectively with a CNS-active agent."),
    "PIK3CA": ("FAILED", "PI3K-AKT-mTOR axis. Buparlisib failed (poor penetration); paxalisib (BBB-penetrant "
               "PI3K/mTOR) is in trials. Pathway redundancy limits single-agent benefit."),
    "PIK3R1": ("FAILED", "Same PI3K axis; regulatory-subunit loss activates signaling. Same caveats as PIK3CA."),
    "PTEN": ("FAILED", "Loss activates PI3K/AKT/mTOR -- not directly druggable; downstream mTOR (everolimus) "
             "gave minimal GBM benefit."),
    "TP53": ("FAILED", "Loss-of-function; not directly druggable. (When intact + MDM2-amp, target MDM2.)"),
    "NF1": ("SUBSET", "Loss activates RAS/MEK; MEK inhibitors (selumetinib/trametinib) have a rationale and "
            "real activity in NF1-driven low-grade glioma/pNF -- weaker evidence in GBM."),
    "RB1": ("BIOMARKER", "Loss -> resistance to CDK4/6i (RB1 must be intact for CDK4/6i to work). A negative "
            "selection marker, not a target."),
    "BRAF": ("SUBSET", "BRAF V600E (rare in GBM; epithelioid GBM/pediatric): dabrafenib+trametinib gives REAL "
             "responses -- FDA tumor-agnostic approval. A genuine win in the small V600E+ subset."),
    "FGFR1": ("SUBSET", "FGFR-altered (incl. FGFR3-TACC3 fusion): erdafitinib/pemigatinib/futibatinib in trials; "
              "some responses in FGFR-fusion glioma."),
    "FGFR3": ("SUBSET", "FGFR3-TACC3 fusion is the actionable lesion; FGFR inhibitors show subset responses."),
    "MET": ("SUBSET", "MET amplification/exon14: crizotinib/capmatinib -- anecdotal/early responses in MET-altered "
            "glioma; CNS exposure variable."),
    "IDH1": ("SUBSET", "Defines IDH-MUTANT glioma (NOT primary IDH-wildtype GBM). Vorasidenib (BBB-penetrant) FDA-"
             "approved 2024 for grade-2 IDH-mut glioma (INDIGO) -- a real recent win, but a different disease "
             "than your father's primary GBM if it was IDH-wildtype."),
    "IDH2": ("SUBSET", "IDH2-mutant (rarer); enasidenib/vorasidenib rationale -- lower-grade IDH-mut glioma, not "
             "primary GBM."),
    "KRAS": ("FAILED", "RAS pathway; no approved GBM-active RAS drug for the alleles seen here."),
    "ATRX": ("BIOMARKER", "Loss -> alternative-lengthening-of-telomeres phenotype; a marker, not yet a target."),
    "MYCN": ("FAILED", "MYC-family amplification; 'undruggable' TF -- indirect strategies (BET/Aurora) experimental."),
    "MYC": ("FAILED", "Undruggable TF; BET-inhibitor strategies experimental, none GBM-approved."),
    "SOX2": ("FAILED", "Stemness TF; not directly druggable."),
    "CCND2": ("LIVE", "Cyclin D2 amplification feeds CDK4/6 -- same CDK4/6i rationale (RB1 intact)."),
    "AKT3": ("FAILED", "PI3K/AKT axis amplification; AKT inhibitors (ipatasertib) not GBM-validated."),
    "PTPN11": ("FAILED", "SHP2 (RAS pathway); SHP2 inhibitors experimental, not GBM-validated."),
    "LZTR1": ("BIOMARKER", "RAS-pathway regulator; loss stabilizes RAS. No direct drug."),
    "QKI": ("BIOMARKER", "RNA-binding tumor suppressor; not druggable."),
}


def dgidb(genes: list[str]) -> dict:
    q = ("query($names:[String!]!){ genes(names:$names){ nodes{ name interactions{ "
         "drug{ name approved } interactionScore interactionTypes{ type } } } } }")
    body = json.dumps({"query": q, "variables": {"names": genes}}).encode()
    req = urllib.request.Request(
        DGIDB, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    out: dict[str, list] = {}
    for node in (((d.get("data") or {}).get("genes") or {}).get("nodes") or []):
        ints = []
        for it in node.get("interactions") or []:
            dr = it.get("drug") or {}
            ints.append({"drug": dr.get("name"), "approved": dr.get("approved"),
                         "score": it.get("interactionScore") or 0.0,
                         "types": ",".join(t.get("type") for t in (it.get("interactionTypes") or []) if t.get("type"))})
        ints.sort(key=lambda x: (-int(bool(x["approved"])), -x["score"]))
        out[node["name"]] = ints
    return out


def main():
    print("=" * 115)
    print("GBM somatic drivers -> druggable targets (DGIdb) + HONEST GBM clinical reality")
    print("GBM is a trial graveyard. A drug existing != it works in GBM. Two killers: BBB penetrance + heterogeneity.")
    print("=" * 115)

    drivers = [r for r in csv.DictReader(LAND.open()) if float(r["altered_pct"]) >= MIN_ALTERED]
    drivers.sort(key=lambda r: -float(r["altered_pct"]))
    genes = [r["gene"] for r in drivers]
    di = dgidb(genes)

    rows = []
    order = {"LIVE": 0, "SUBSET": 1, "BIOMARKER": 2, "FAILED": 3, "?": 4}
    for r in drivers:
        g = r["gene"]
        status, note = GBM_REALITY.get(g, ("?", "no curated GBM note"))
        ints = di.get(g, [])
        approved = [i for i in ints if i["approved"]]
        top = (approved or ints)[:4]
        drugs = ", ".join(f"{i['drug']}{'*' if i['approved'] else ''}" for i in top) or "-"
        rows.append({"gene": g, "altered_pct": r["altered_pct"], "status": status,
                     "n_drugs": len(ints), "n_approved": len(approved),
                     "top_drugs": drugs, "gbm_note": note})

    rows.sort(key=lambda r: (order.get(r["status"], 9), -float(r["altered_pct"])))

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "altered_pct", "status", "n_drugs",
                                           "n_approved", "top_drugs", "gbm_note"])
        w.writeheader()
        w.writerows(rows)

    cur = None
    for r in rows:
        if r["status"] != cur:
            cur = r["status"]
            banner = {"LIVE": "LIVE -- rational, biomarker-selected, a CNS-active option exists",
                      "SUBSET": "SUBSET -- works only in a rare molecular subset (real responses there)",
                      "BIOMARKER": "BIOMARKER -- the lesion guides therapy but is not itself a drug target",
                      "FAILED": "FAILED / WEAK -- tried in GBM, didn't deliver (usually BBB + heterogeneity)",
                      "?": "uncurated"}[cur]
            print(f"\n### {banner}")
        print(f"  {r['gene']:<8} altered {float(r['altered_pct']):>5.1f}%  "
              f"drugs:{r['n_drugs']:>3} (approved {r['n_approved']:>2})  {r['top_drugs'][:52]}")
        print(f"           {r['gbm_note']}")

    print("\n" + "=" * 115)
    print("BOTTOM LINE (honest):")
    print(" * The biggest, most frequent lesions (EGFR amp 44%, PTEN/TP53 loss) are the ones that have")
    print("   repeatedly FAILED in trials -- not for lack of drugs, but BBB penetrance + tumor heterogeneity.")
    print(" * The genuine leverage is in SMALL, biomarker-defined subsets with CNS-active agents:")
    print("     - CDK4/6-amp + RB1-intact + CDKN2A/B-deleted -> abemaciclib (CNS-penetrant)")
    print("     - TP53-WT + MDM2/4-amp -> MDM2 inhibitors (early-phase)")
    print("     - BRAF V600E -> dabrafenib+trametinib (real responses; tumor-agnostic approval)")
    print("     - FGFR3-TACC3 fusion / NTRK fusion -> FGFR / TRK inhibitors (subset responses)")
    print("     - IDH-MUTANT lower-grade glioma -> vorasidenib (2024) -- a DIFFERENT disease than IDH-WT GBM")
    print(" * '*' marks FDA-approved drugs. Approved-for-something != effective-in-GBM. None of this is")
    print("   medical advice; it is a research map of where rational, testable shots actually remain.")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
