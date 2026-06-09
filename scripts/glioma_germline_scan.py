#!/usr/bin/env python3
"""Front A -- FinnGen glioma founder rare-variant predisposition hunt.

Scans the per-endpoint FinnGen sumstats (NOT the 30 GB annotation file) for rare,
Finnish-enriched, large-effect GERMLINE variants associated with glioma, focusing on
signal OUTSIDE the known glioma risk loci.

HONEST SCOPE: glioma case counts are small (C3_GBM 467, GBM_ASTROCYTOMA 618,
BRAIN_WIDE 1987). Single rare-variant associations at this N are underpowered and
mostly noise. This is HYPOTHESIS GENERATION. A candidate only becomes interesting if it
(a) sits in/near a biologically plausible gene (tumor-suppressor / DNA-repair),
(b) is Finnish-enriched in gnomAD (founder signature), and ideally
(c) recurs -- though the three endpoints are NESTED (GBM subset of GBM_ASTROCYTOMA
    subset of BRAIN_WIDE), so cross-endpoint recurrence is shared-sample, NOT independent.

Sumstats columns: chrom pos ref alt rsids nearest_genes pval mlogp beta sebeta
                  af_alt af_alt_cases af_alt_controls
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

SS = Path.home() / "finngen-r13" / "summary_stats"
OUT = Path.home() / "finngen-triage" / "glioma_germline_candidates.csv"
ENDPOINTS = ["C3_GBM", "C3_GBM_ASTROCYTOMA", "C3_BRAIN_WIDE"]

MAX_MAF = 0.02     # rare / low-frequency
MIN_MLOGP = 5.0    # suggestive (genome-wide = 7.3); relaxed because N is small

# Known glioma germline risk loci (GRCh38), flagged so novel signal stands out.
KNOWN = [
    ("5", 1_000_000, 1_400_000, "TERT 5p15.33"),
    ("7", 54_000_000, 56_000_000, "EGFR 7p11.2"),
    ("8", 128_000_000, 131_000_000, "CCDC26/MYC 8q24.21"),
    ("9", 21_000_000, 22_500_000, "CDKN2A/B 9p21.3"),
    ("11", 118_000_000, 119_200_000, "PHLDB1 11q23.3"),
    ("17", 7_000_000, 8_200_000, "TP53 17p13.1"),
    ("20", 62_000_000, 63_100_000, "RTEL1 20q13.33"),
    ("3", 169_000_000, 170_500_000, "TERC/MYNN 3q26"),
    ("2", 217_000_000, 218_500_000, "AKT? 2q33-q35 glioma"),
    ("15", 73_000_000, 74_500_000, "ETFA/15q24 glioma"),
    ("16", 9_800_000, 10_200_000, "16p13 glioma"),
]

# Cancer-relevant genes (predisposition / driver) for a proximity flag. Hitting one of
# these with a rare germline variant is what we actually care about.
CANCER_GENES = {
    "TP53", "PTEN", "NF1", "NF2", "RB1", "ATRX", "IDH1", "IDH2", "CIC", "FUBP1",
    "MUTYH", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "POLE", "POLD1",
    "BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "NBN", "RAD51C", "RAD51D", "BARD1",
    "TERT", "TERF1", "POT1", "TERF2IP", "STK11", "PTCH1", "SUFU", "APC", "VHL",
    "TSC1", "TSC2", "SMARCB1", "SMARCA4", "CDKN2A", "CDKN2B", "CDK4", "MDM2", "MDM4",
    "EGFR", "PDGFRA", "MET", "PIK3CA", "PIK3R1", "BRAF", "KIT", "FANCA", "FANCC",
    "FANCD2", "FANCM", "BLM", "WRN", "RECQL4", "ERCC2", "ERCC3", "XRCC2", "DICER1",
}


def known_locus(chrom: str, pos: int) -> str:
    for c, s, e, label in KNOWN:
        if chrom == c and s <= pos <= e:
            return label
    return ""


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def scan_endpoint(ep: str) -> list[dict]:
    path = SS / f"finngen_R13_{ep}.gz"
    if not path.exists():
        print(f"  {ep}: sumstats missing ({path})")
        return []
    # header-driven awk filter: ONE streaming zcat per file (sequential across endpoints).
    awk = (
        'NR==1{for(i=1;i<=NF;i++){h=$i;sub(/^#/,"",h);col[h]=i} next} '
        '{af=$col["af_alt"]+0; ml=$col["mlogp"]+0; maf=(af<=0.5)?af:1-af; '
        f'if(ml>={MIN_MLOGP} && maf<={MAX_MAF}) '
        'print $col["chrom"]"\\t"$col["pos"]"\\t"$col["ref"]"\\t"$col["alt"]"\\t"'
        '$col["rsids"]"\\t"$col["nearest_genes"]"\\t"ml"\\t"$col["beta"]"\\t"'
        '$col["af_alt"]"\\t"$col["af_alt_cases"]"\\t"$col["af_alt_controls"]}'
    )
    cmd = f"zcat {path} | awk -F'\\t' '{awk}'"
    print(f"  scanning {ep} (one streaming pass) ...", flush=True)
    res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    rows = []
    for line in res.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 11:
            continue
        chrom, pos = f[0], int(f[1])
        af_cases, af_ctrl = fnum(f[9]), fnum(f[10])
        ratio = (af_cases / af_ctrl) if (af_cases and af_ctrl) else None
        genes = f[5]
        cancer_hit = sorted({g for g in genes.replace(";", ",").split(",") if g.strip() in CANCER_GENES})
        rows.append({
            "endpoint": ep, "chrom": chrom, "pos": pos, "ref": f[2], "alt": f[3],
            "rsids": f[4], "nearest_genes": genes, "mlogp": round(float(f[6]), 2),
            "beta": round(float(f[7]), 3), "af_alt": float(f[8]),
            "af_cases": af_cases, "af_ctrl": af_ctrl,
            "case_ctrl_af_ratio": round(ratio, 2) if ratio else None,
            "direction": "RISK" if float(f[7]) > 0 else "PROTECTIVE",
            "known_locus": known_locus(chrom, pos),
            "cancer_gene_hit": ",".join(cancer_hit),
        })
    return rows


def main():
    print("=" * 100)
    print("FinnGen glioma founder rare-variant scan (germline predisposition; hypothesis-generating)")
    print(f"filters: MAF<={MAX_MAF:.0%}, mlogp>={MIN_MLOGP}  | endpoints (nested): {ENDPOINTS}")
    print("=" * 100)
    all_rows = []
    for ep in ENDPOINTS:
        rows = scan_endpoint(ep)
        print(f"    {ep}: {len(rows)} rare candidates pass filter "
              f"({sum(1 for r in rows if not r['known_locus'])} outside known loci)")
        all_rows.extend(rows)

    if not all_rows:
        print("no candidates"); return

    # write full table
    cols = ["endpoint", "chrom", "pos", "ref", "alt", "rsids", "nearest_genes", "mlogp",
            "beta", "direction", "af_alt", "af_cases", "af_ctrl", "case_ctrl_af_ratio",
            "known_locus", "cancer_gene_hit"]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: (-r["mlogp"])):
            w.writerow(r)
    print(f"\nwrote {OUT.name} ({len(all_rows)} rows)")

    # cross-endpoint recurrence (caveat: NESTED samples, not independent)
    by_var: dict[tuple, list] = {}
    for r in all_rows:
        by_var.setdefault((r["chrom"], r["pos"], r["ref"], r["alt"]), []).append(r)
    recurrent = {k: v for k, v in by_var.items() if len({r["endpoint"] for r in v}) >= 2}

    def show(rows, title, n=20):
        print(f"\n### {title}")
        hdr = f"{'endpoint':<20}{'variant':<22}{'mlogp':>6}{'beta':>7}{'dir':>10}{'MAF':>9}  {'gene(s)':<22}{'cancer':<14}{'known'}"
        print(hdr)
        for r in rows[:n]:
            var = f"{r['chrom']}:{r['pos']}:{r['ref']}>{r['alt']}"
            maf = min(r["af_alt"], 1 - r["af_alt"])
            print(f"{r['endpoint']:<20}{var:<22}{r['mlogp']:>6}{r['beta']:>7}{r['direction']:>10}"
                  f"{maf:>9.3%}  {r['nearest_genes'][:20]:<22}{r['cancer_gene_hit'][:12]:<14}{r['known_locus']}")

    novel = sorted([r for r in all_rows if not r["known_locus"]], key=lambda r: -r["mlogp"])
    show(novel, "Top rare candidates OUTSIDE known glioma loci (by mlogp)")

    cancer = sorted([r for r in all_rows if r["cancer_gene_hit"] and not r["known_locus"]],
                    key=lambda r: -r["mlogp"])
    show(cancer, "Rare candidates hitting a cancer-predisposition/driver gene (outside known loci)", n=30)

    print(f"\n### Variants recurring across >=2 (nested) glioma endpoints: {len(recurrent)}")
    for k, v in sorted(recurrent.items(), key=lambda kv: -max(r["mlogp"] for r in kv[1]))[:20]:
        eps = ",".join(sorted({r["endpoint"].replace('C3_', '') for r in v}))
        best = max(v, key=lambda r: r["mlogp"])
        var = f"{k[0]}:{k[1]}:{k[2]}>{k[3]}"
        print(f"  {var:<22} mlogp<={best['mlogp']:<6} {best['nearest_genes'][:24]:<26} "
              f"{best['cancer_gene_hit'] or '-':<10} known:{best['known_locus'] or '-':<18} in[{eps}]")

    print("\nReading: ignore known-locus rows (expected rediscoveries). A row outside known loci")
    print("that hits a cancer gene AND is Finnish-enriched in gnomAD is the only kind worth")
    print("chasing -- and even then it needs gene-burden testing + an external cohort.")


if __name__ == "__main__":
    main()
