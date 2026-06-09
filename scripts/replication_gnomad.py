#!/usr/bin/env python3
"""External evidence for the independence-tested Tier-1 LoF candidates via the gnomAD v4
GraphQL API (read-only, cached to gnomad_cache.json).

For each qualifying LoF/mask variant we pull gnomAD exome+genome AC/AN per genetic
ancestry group and compute the Finnish (fin) vs non-Finnish-European (nfe) allele-
frequency enrichment. This is the EXTERNAL test of the founder-enrichment claim,
independent of FinnGen. For each gene we pull the gnomAD LoF constraint (LOEUF/pLI/lof_z).

FOUNDER CAVEAT: rare Finnish-enriched alleles are by construction often absent or ultra-
rare in non-Finnish cohorts, so single-variant replication OUTSIDE Finland is frequently
infeasible. gnomAD-FIN is the right external corroboration of enrichment; the Estonian
Biobank (also bottlenecked) would be the ideal replication cohort. We therefore do NOT
treat 'absent in NFE' as a failure to replicate — it is the expected founder signature.

The INTERNAL verdict column echoes the FinnGen independence test (multi_independence.py):
which variants are genome-wide-significant + LD-independent vs burden-only vs could-tag.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

API = "https://gnomad.broadinstitute.org/api"
CACHE = Path.home() / "finngen-triage" / "gnomad_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

# gene -> list of (gnomad_variant_id, internal FinnGen independence verdict)
CANDIDATES = {
    "PLOD2": [
        ("3-146110290-G-C", "GW-sig lead in hip-OA (mlogp 7.7); indep of common OA SNP (r2max .033); PROTECTIVE stop_gained"),
    ],
    "NBEAL1": [
        ("2-203202685-A-G", "GW-SIG INDEP (mlogp 10-12; r2max .037-.049 vs common 2q33 lead)"),
        ("2-203130462-G-T", "burden-only (marginal null, AF<0.01%)"),
        ("2-203169768-C-T", "burden-only (marginal null)"),
    ],
    "ATG4C": [
        ("1-62834058-TTG-T", "locus sub-GW (lead mlogp 4.9); not a standalone signal"),
        ("1-62819215-C-CT", "could-tag; sub-GW"),
        ("1-62819152-CA-C", "burden-only; sub-GW"),
    ],
    "NRAP": [
        ("10-113595655-G-A", "sub-GW lead (mlogp 4.9); not GW-significant"),
        ("10-113597145-TG-T", "could-tag; sub-GW"),
        ("10-113640311-A-T", "could-tag; sub-GW"),
    ],
    "LZTR1": [
        ("22-20993977-G-A", "could-tag THAP7 lead (r2max .87); sub-GW (mlogp 5.6)"),
        ("22-20997287-T-TA", "could-tag; sub-GW"),
        ("22-20993660-A-G", "could-tag"),
        ("22-20982391-C-CG", "could-tag"),
    ],
}

VAR_Q = ("query($vid:String!){ variant(variantId:$vid, dataset:gnomad_r4){ variant_id "
         "exome{ ac an populations{ id ac an } } genome{ ac an populations{ id ac an } } } }")
GENE_Q = ("query($sym:String!){ gene(gene_symbol:$sym, reference_genome:GRCh38){ symbol gene_id "
          "gnomad_constraint{ oe_lof oe_lof_lower oe_lof_upper pLI lof_z mis_z } } }")


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


def pop_af(block: dict | None) -> dict:
    """ancestry id -> (ac, an, af); plus '_overall'. Empty if no block."""
    out: dict = {}
    if not block:
        return out
    for p in block.get("populations") or []:
        an = p["an"] or 0
        out[p["id"]] = (p["ac"], an, (p["ac"] / an if an else 0.0))
    an = block.get("an") or 0
    out["_overall"] = (block.get("ac"), an, (block.get("ac") / an if an else 0.0))
    return out


def fmt_enrichment(fin_af: float, nfe_af: float) -> str:
    if fin_af == 0:
        return "FIN=0"
    if nfe_af == 0:
        return "FIN-specific (NFE=0)"
    return f"{fin_af / nfe_af:,.0f}x"


def main():
    print("=" * 100)
    print("gnomAD v4 external evidence for FinnGen Tier-1 LoF candidates  "
          "(FIN vs NFE founder-enrichment + gene constraint)")
    print("=" * 100)
    for gene, variants in CANDIDATES.items():
        gd = gql(GENE_Q, {"sym": gene}, f"gene:{gene}")
        g = (gd.get("data") or {}).get("gene") or {}
        c = g.get("gnomad_constraint") or {}
        loeuf = c.get("oe_lof_upper")
        print(f"\n### {gene}  ({g.get('gene_id','?')})   "
              f"LOEUF={loeuf if loeuf is None else round(loeuf,3)}  "
              f"oe_lof={c.get('oe_lof') and round(c['oe_lof'],3)}  "
              f"pLI={c.get('pLI') and round(c['pLI'],3)}  lof_z={c.get('lof_z') and round(c['lof_z'],2)}")
        for vid, verdict in variants:
            vd = gql(VAR_Q, {"vid": vid}, f"var:{vid}")
            if vd.get("errors"):
                print(f"  {vid:<26} gnomAD error: {vd['errors'][0]!s:.60}")
                continue
            v = (vd.get("data") or {}).get("variant")
            if not v:
                print(f"  {vid:<26} NOT in gnomAD v4 (ultra-rare / different representation)")
                print(f"      internal: {verdict}")
                continue
            ex, gen = pop_af(v.get("exome")), pop_af(v.get("genome"))
            # prefer exome (coding callset, larger N) when it carries fin/nfe
            src, table = ("exome", ex) if ("fin" in ex or "nfe" in ex) else ("genome", gen)
            fin = table.get("fin", (0, 0, 0.0))
            nfe = table.get("nfe", (0, 0, 0.0))
            ovr = table.get("_overall", (0, 0, 0.0))
            print(f"  {vid:<26} [{src}]  "
                  f"FIN {fin[0]}/{fin[1]} (AF {fin[2]:.3%})   "
                  f"NFE {nfe[0]}/{nfe[1]} (AF {nfe[2]:.4%})   "
                  f"enrich {fmt_enrichment(fin[2], nfe[2])}   "
                  f"global AF {ovr[2]:.4%}")
            print(f"      internal: {verdict}")
    print("\n" + "=" * 100)
    print("Reading: high FIN/NFE enrichment in gnomAD = founder claim corroborated by an "
          "independent dataset. 'FIN-specific (NFE=0)' is the strongest founder signature, "
          "and also means non-Finnish single-variant replication is not possible.")


if __name__ == "__main__":
    main()
