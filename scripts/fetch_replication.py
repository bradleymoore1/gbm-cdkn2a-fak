#!/usr/bin/env python3
"""Locus-level external replication of the two surviving FinnGen Tier-1 founder hits.

We remote-slice (pysam over HTTPS, no full download) the GWAS Catalog HARMONISED
summary statistics for the best-powered NON-Finnish meta-analyses and test the two
loci that survived our internal independence test + gnomAD founder corroboration:

  PLOD2  3:146110290 G>C  protective stop_gained, hip/all-OA   (gnomAD FIN/NFE 315x)
  NBEAL1 2:203202685 A>G  rare independent CAD variant         (gnomAD FIN/NFE  58x)

External cohorts:
  OA  -> GCST90134284  all-OA, MVP + UK Biobank, MULTI-ANCESTRY (Eur/SAS/EAS/Afr/His),
         484k effective (140k cases / 344k controls per ancestry block). Multi-ancestry
         makes this a genuinely independent test, not a Finnish/European echo.
  CAD -> GCST90132314  Aragam 2022 CAD meta-analysis (CARDIoGRAMplusC4D + MVP + UKB ...).

HONEST SCOPE -- this is LOCUS-LEVEL, not single-variant, replication.
  Our edge is a RARE Finnish-founder allele (AF 0.2-0.5% in Finland). Such alleles are
  by construction absent / ultra-rare / MAF-filtered in non-Finnish meta-analyses, so a
  direct single-variant replication is usually IMPOSSIBLE here (that needs another
  bottlenecked cohort -- Estonian Biobank, gated). What these downloads DO test:
    (1) is the REGION OA/CAD-relevant outside Finland? (biological corroboration of locus)
    (2) what is the strongest signal in the window, and is it a known COMMON locus
        distinct from our rare independent variant?
    (3) is our exact variant present at all, and if so, same effect direction?
  A strong common signal in the window is EXPECTED at known loci (e.g. the WDR12/2q33
  CAD locus sits inside the NBEAL1 window) and does NOT by itself replicate our rare,
  LD-independent founder variant -- our internal r2_max test already showed independence.
"""
from __future__ import annotations

import gzip
import io
import urllib.request

import pysam

BASE = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
GWSIG = 5e-8
RARE = 0.01  # EAF threshold for "rare/low-frequency" in the external cohort

SOURCES = {
    "OA": f"{BASE}/GCST90134001-GCST90135000/GCST90134284/harmonised/GCST90134284.h.tsv.gz",
    "CAD": f"{BASE}/GCST90132001-GCST90133000/GCST90132314/harmonised/GCST90132314.h.tsv.gz",
}

LOCI = [
    {
        "name": "PLOD2  3:146110290 G>C (hip/all-OA protective stop_gained)",
        "source": "OA", "chrom": "3", "pos": 146110290, "ea": "C", "oa": "G",
        "window": (146_000_000, 146_200_000), "fin_af": 0.00227,
        "internal": "GW-sig mlogp 13.5 all-OA; beta -0.49 (PROTECTIVE); r2max .033 vs common OA lead -> INDEPENDENT",
    },
    {
        "name": "NBEAL1 2:203202685 A>G (rare independent CAD variant)",
        "source": "CAD", "chrom": "2", "pos": 203202685, "ea": "G", "oa": "A",
        "window": (202_800_000, 203_500_000), "fin_af": 0.00529,
        "internal": "GW-sig mlogp 12; r2max .037-.049 vs common 2q33/WDR12 CAD lead -> INDEPENDENT",
    },
]


def header_cols(url: str) -> dict[str, int]:
    """Read the column header from the first BGZF block via a small HTTP range request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    try:
        txt = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "replace")
    except Exception:
        import zlib
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        txt = d.decompress(raw).decode("utf-8", "replace")
    cols = txt.splitlines()[0].split("\t")
    return {name: i for i, name in enumerate(cols)}


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def scan_locus(loc: dict, url: str, col: dict[str, int]) -> dict:
    ci = col["chromosome"]; pi = col["base_pair_location"]
    eai = col["effect_allele"]; oai = col["other_allele"]
    bi = col["beta"]; pvi = col["p_value"]
    eafi = col.get("effect_allele_frequency")
    ori = col.get("odds_ratio")
    rsi = col.get("rsid", col.get("variant_id"))

    chrom, (w0, w1) = loc["chrom"], loc["window"]
    tbx = pysam.TabixFile(url)
    top = None            # strongest signal anywhere in the window
    top_rare = None       # strongest signal among EAF < RARE
    exact = None          # our exact variant (pos + allele match either orientation)
    n = 0
    for row in tbx.fetch(chrom, max(0, w0 - 1), w1):
        f = row.split("\t")
        pos = fnum(f[pi])
        if pos is None or not (w0 <= pos <= w1):
            continue
        n += 1
        p = fnum(f[pvi])
        if p is None:
            continue
        ea, oa = f[eai].upper(), f[oai].upper()
        eaf = fnum(f[eafi]) if eafi is not None else None
        rec = {
            "pos": int(pos), "ea": ea, "oa": oa, "beta": fnum(f[bi]),
            "or": fnum(f[ori]) if ori is not None else None,
            "eaf": eaf, "p": p, "rsid": f[rsi] if rsi is not None and rsi < len(f) else "?",
        }
        if top is None or p < top["p"]:
            top = rec
        if eaf is not None and min(eaf, 1 - eaf) < RARE and (top_rare is None or p < top_rare["p"]):
            top_rare = rec
        if int(pos) == loc["pos"] and {ea, oa} == {loc["ea"], loc["oa"]}:
            exact = rec
    tbx.close()
    return {"n": n, "top": top, "top_rare": top_rare, "exact": exact}


def direction_vs_finngen(loc, rec) -> str:
    """Align the external beta to OUR effect allele and state concordance.

    FinnGen: effect allele = loc['ea'] with PROTECTIVE (negative) beta for PLOD2.
    """
    if rec is None or rec["beta"] is None:
        return ""
    b = rec["beta"]
    if rec["ea"] == loc["ea"]:
        aligned = b
    elif rec["ea"] == loc["oa"]:
        aligned = -b
    else:
        return "  (alleles do not match our variant; cannot orient)"
    sign = "PROTECTIVE" if aligned < 0 else "RISK-increasing"
    return f"  aligned-to-our-EA({loc['ea']}) beta={aligned:+.3f} -> {sign} for the trait"


def fmt(rec) -> str:
    if rec is None:
        return "none"
    eaf = f"{rec['eaf']:.4%}" if rec["eaf"] is not None else "NA"
    orr = f" OR={rec['or']:.3f}" if rec["or"] is not None else ""
    return (f"{rec['rsid']} @ {rec['pos']:,} {rec['oa']}>{rec['ea']}  "
            f"p={rec['p']:.2e} beta={rec['beta']:+.3f}{orr} EAF={eaf}")


def main():
    print("=" * 100)
    print("LOCUS-LEVEL external replication (GWAS Catalog harmonised sumstats, remote-sliced)")
    print("Scope: tests the REGION in non-Finnish meta-analyses. Rare founder alleles are")
    print("usually absent/MAF-filtered here -> single-variant replication needs a bottlenecked")
    print("cohort (EstBB). A common window hit can be a KNOWN locus distinct from our rare variant.")
    print("=" * 100)
    headers = {}
    for loc in LOCI:
        src = loc["source"]; url = SOURCES[src]
        if src not in headers:
            headers[src] = header_cols(url)
        print(f"\n### {loc['name']}")
        print(f"    source: {src} = {url.rsplit('/',1)[1]}")
        print(f"    window: chr{loc['chrom']}:{loc['window'][0]:,}-{loc['window'][1]:,}   "
              f"FinnGen AF(FIN)={loc['fin_af']:.3%}")
        print(f"    internal verdict: {loc['internal']}")
        try:
            res = scan_locus(loc, url, headers[src])
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR slicing: {e!r}")
            continue
        print(f"    variants in window: {res['n']:,}")
        top = res["top"]
        gw = (top and top["p"] < GWSIG)
        print(f"    strongest in window: {fmt(top)}"
              f"   {'[GENOME-WIDE SIG]' if gw else '[not GW-sig]'}")
        if top:
            print(direction_vs_finngen(loc, top).lstrip() or "")
        print(f"    strongest RARE (EAF<{RARE:.0%}) in window: {fmt(res['top_rare'])}")
        ex = res["exact"]
        if ex:
            print(f"    >>> OUR EXACT VARIANT PRESENT: {fmt(ex)}")
            print("       " + direction_vs_finngen(loc, ex).strip())
            print(f"       {'REPLICATES (GW-sig)' if ex['p']<GWSIG else 'present but not GW-sig in this cohort'}")
        else:
            print(f"    >>> our exact variant {loc['chrom']}:{loc['pos']} {loc['oa']}>{loc['ea']} "
                  f"NOT in this cohort (expected for a rare Finnish-founder allele)")
    print("\n" + "=" * 100)
    print("Interpretation: a GW-sig COMMON hit in the window corroborates that the LOCUS is")
    print("trait-relevant outside Finland, but is a different (common) signal than our rare,")
    print("LD-independent founder variant (internal r2_max already proved independence).")
    print("True single-variant replication of the founder allele requires another bottlenecked")
    print("biobank (Estonian Biobank / FinnGen individual-level) -- both access-gated.")


if __name__ == "__main__":
    main()
