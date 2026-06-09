#!/usr/bin/env python3
"""GBM germline founder candidates -> gnomAD v4 validation + ARTIFACT triage.

Front A's scan produced rare, large-effect glioma-associated germline variants. At 467-1987
cases these are HYPOTHESIS-GENERATING, not findings. This module is the first hard filter:
for each top candidate (outside known glioma loci) we query gnomAD v4 and ask three things:

  1. ARTIFACT?  Does gnomAD flag the site as segdup (segmental duplication) / lcr (low-
     complexity) or apply an AC0 / RF / InbreedingCoeff filter? Pericentromeric 9p11-q12
     (CNTNAP3B) and fragile sites (PRKN/FRA6E) are classic call-artifact zones -- we expect
     several candidates to die here, exactly like LZTR1/22q11 did.
  2. FOUNDER?   FIN allele frequency vs non-Finnish-European (NFE). High FIN/NFE enrichment
     (or NFE=0 with FIN>0) corroborates a Finnish-founder origin independent of FinnGen.
  3. CONSTRAINT for the nearest gene (LOEUF/pLI/lof_z): is LoF in this gene even plausible?

A candidate worth carrying forward = present in gnomAD, NOT segdup/lcr/filtered, FIN-enriched,
in a constrained/plausible gene. Everything else is parked. Honest: even a survivor needs
gene-burden testing + a second bottlenecked cohort (Estonian Biobank) before it means anything.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

API = "https://gnomad.broadinstitute.org/api"
CAND = Path.home() / "finngen-triage" / "glioma_germline_candidates.csv"
CACHE = Path.home() / "finngen-triage" / "gnomad_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

TOP_N = 18  # top novel candidates by mlogp + any cancer-gene hit

VAR_Q = ("query($vid:String!){ variant(variantId:$vid, dataset:gnomad_r4){ variant_id flags "
         "exome{ ac an filters populations{ id ac an } } "
         "genome{ ac an filters populations{ id ac an } } } }")
GENE_Q = ("query($sym:String!){ gene(gene_symbol:$sym, reference_genome:GRCh38){ symbol "
          "gnomad_constraint{ oe_lof_upper pLI lof_z } } }")


def gql(query: str, variables: dict, key: str) -> dict:
    if key in cache:
        return cache[key]
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"content-type": "application/json", "user-agent": "finngen-triage/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            cache[key] = data
            CACHE.write_text(json.dumps(cache))
            time.sleep(0.4)
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"errors": [str(e)]}
            time.sleep(1.5)


def pop_af(block: dict | None):
    if not block:
        return {}, []
    out = {}
    for p in block.get("populations") or []:
        an = p["an"] or 0
        out[p["id"]] = (p["ac"], an, (p["ac"] / an if an else 0.0))
    an = block.get("an") or 0
    out["_overall"] = (block.get("ac"), an, (block.get("ac") / an if an else 0.0))
    return out, (block.get("filters") or [])


def load_candidates() -> list[dict]:
    rows = [r for r in csv.DictReader(CAND.open()) if not r["known_locus"]]
    # dedupe by variant, keep best (max mlogp) per variant
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["chrom"], r["pos"], r["ref"], r["alt"])
        if k not in best or float(r["mlogp"]) > float(best[k]["mlogp"]):
            best[k] = r
    uniq = list(best.values())
    by_ml = sorted(uniq, key=lambda r: -float(r["mlogp"]))
    top = by_ml[:TOP_N]
    # ensure any cancer-gene hit is included even if below TOP_N
    for r in uniq:
        if r["cancer_gene_hit"] and r not in top:
            top.append(r)
    return top


def constraint_for(sym: str) -> str:
    if not sym or sym == "NA":
        return ""
    gd = gql(GENE_Q, {"sym": sym}, f"gene:{sym}")
    g = (gd.get("data") or {}).get("gene") or {}
    c = g.get("gnomad_constraint") or {}
    if not c:
        return "no-constraint-data"
    lo = c.get("oe_lof_upper")
    return f"LOEUF={lo and round(lo,2)} pLI={c.get('pLI') and round(c['pLI'],2)} lof_z={c.get('lof_z') and round(c['lof_z'],1)}"


def verdict(present, flags, filters, fin_af, nfe_af) -> str:
    art = [f for f in (flags or []) if f in ("segdup", "lcr", "lc_lof", "mnv")]
    bad_filters = [f for f in (filters or []) if f in ("AC0", "RF", "InbreedingCoeff", "AS_VQSR")]
    if not present:
        return "ABSENT in gnomAD (ultra-rare or artifact) -- ambiguous"
    if art or bad_filters:
        tags = ",".join(art + bad_filters)
        return f"ARTIFACT-PRONE ({tags}) -- DROP"
    if fin_af > 0 and (nfe_af == 0 or fin_af / max(nfe_af, 1e-12) >= 5):
        enr = "NFE=0 founder-private" if nfe_af == 0 else f"{fin_af/nfe_af:.0f}x FIN/NFE"
        return f"FOUNDER-CONSISTENT ({enr}) -- carry forward"
    if fin_af > 0:
        return f"present but NOT FIN-enriched ({fin_af/max(nfe_af,1e-12):.1f}x) -- not a founder signal"
    return "present, FIN=0 -- not Finnish-enriched"


def main():
    print("=" * 110)
    print("GBM germline founder candidates -> gnomAD v4 (ARTIFACT triage + founder enrichment + gene constraint)")
    print("Honest: this is the FIRST filter on hypothesis-level hits, not validation of a finding.")
    print("=" * 110)
    cands = load_candidates()
    print(f"testing {len(cands)} unique novel candidates (top {TOP_N} by mlogp + cancer-gene hits)\n")
    keep, drop, ambiguous = [], [], []
    for r in cands:
        chrom = "X" if r["chrom"] == "23" else r["chrom"]
        vid = f"{chrom}-{r['pos']}-{r['ref']}-{r['alt']}"
        vd = gql(VAR_Q, {"vid": vid}, f"var:{vid}")
        v = (vd.get("data") or {}).get("variant") if not vd.get("errors") else None
        gene = r["nearest_genes"].split(",")[0].split(";")[0].strip()
        if v:
            ex, exf = pop_af(v.get("exome"))
            gen, genf = pop_af(v.get("genome"))
            table, filt = (ex, exf) if ("fin" in ex or "nfe" in ex) else (gen, genf)
            fin = table.get("fin", (0, 0, 0.0))
            nfe = table.get("nfe", (0, 0, 0.0))
            ovr = table.get("_overall", (0, 0, 0.0))
            flags = v.get("flags") or []
            vd_txt = verdict(True, flags, filt, fin[2], nfe[2])
            afinfo = (f"FIN {fin[2]:.3%} NFE {nfe[2]:.4%} global {ovr[2]:.4%}"
                      f"  flags={flags or '-'} filters={filt or '-'}")
        else:
            vd_txt = verdict(False, [], [], 0, 0)
            afinfo = "not in gnomAD v4"
        con = constraint_for(gene)
        tag = ("FOUNDER" in vd_txt and "carry" in vd_txt) and "KEEP" or \
              ("ARTIFACT" in vd_txt or "DROP" in vd_txt) and "DROP" or "WATCH"
        (keep if tag == "KEEP" else drop if tag == "DROP" else ambiguous).append(r)
        print(f"[{tag:<5}] {r['endpoint']:<18} {vid:<24} mlogp {float(r['mlogp']):>5.2f} "
              f"beta {float(r['beta']):>+6.2f}  gene={gene or 'NA':<12} "
              f"{('cancer:'+r['cancer_gene_hit']) if r['cancer_gene_hit'] else ''}")
        print(f"         {afinfo}")
        if con:
            print(f"         gene-constraint: {con}")
        print(f"         => {vd_txt}")
    print("\n" + "=" * 110)
    print(f"SUMMARY: {len(keep)} carry-forward (founder-consistent, non-artifact) | "
          f"{len(drop)} dropped (artifact/segdup/filtered) | {len(ambiguous)} watch (absent/not-enriched)")
    if keep:
        print("\nCARRY FORWARD (need gene-burden test + Estonian Biobank before they mean anything):")
        for r in keep:
            print(f"  {r['chrom']}:{r['pos']} {r['ref']}>{r['alt']}  {r['nearest_genes'][:30]}  "
                  f"mlogp {r['mlogp']} beta {r['beta']}  {r['endpoint']}")
    else:
        print("\nNo candidate survives as a clean founder-consistent, non-artifact signal at this pass.")
        print("That is the honest, expected outcome at 467-1987 GBM cases -- single rare variants")
        print("are underpowered. The real germline play is GENE-BURDEN (collapse rare LoF per gene),")
        print("not single-variant. Next: burden the glioma endpoints against DNA-repair/TSG gene sets.")


if __name__ == "__main__":
    main()
