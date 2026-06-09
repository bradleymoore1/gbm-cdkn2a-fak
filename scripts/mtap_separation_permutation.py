#!/usr/bin/env python3
"""
Tier-2 deconfounding: does the focal-adhesion/FAK -> TAZ/TEAD dependency in
CDKN2A/B-null GBM/cancer lines SURVIVE separation from the MTAP co-deletion?

The #1 limitation: CDKN2A/B (9p21.3) and MTAP are co-deleted ~always
(phi ~= 0.92). PRMT5/MAT2A dependency is *caused by* MTAP loss. If our
FAK/TEAD signal were the same kind of artifact, it would vanish once we hold
MTAP intact. So we build three genotype groups and run the decisive contrast:

    A = CDKN2A/B-null & MTAP-null    (the confounded majority; baseline signal)
    B = CDKN2A/B-null & MTAP-INTACT  (clean test: 9p21 CDK-locus loss WITHOUT MTAP loss)
    C = CDKN2A/B-intact & MTAP-intact (9p21-intact comparator)

Dependency scores: more NEGATIVE = more essential (CRISPR Chronos & RNAi DEMETER2).
effect = median(C) - median(B).  POSITIVE effect => group more dependent than C
         => dependency present.   For B-vs-C, a positive, significant effect means
         the dependency is NOT an MTAP-co-deletion artifact.

Controls (must behave correctly to trust the design):
  PRMT5, MAT2A  -- MTAP-synthetic-lethal. Strong in A-vs-C, should COLLAPSE in
                   B-vs-C (B is MTAP-intact). This is the calibration negative control.
  MTAP          -- neutral self-check.

Two orthogonal screens (CRISPR + RNAi) are pooled per gene/scope via Stouffer's Z
(weighted by sqrt(nB+nC)); agreement across silencing technologies rules out a
CRISPR copy-number-cutting artifact at the deleted locus.

Outputs: console report + mtap_separation_permutation.csv
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, norm
from concurrent.futures import ProcessPoolExecutor, as_completed

HOM       = 0.2        # CN < HOM  => homozygous deletion
# null definition: paper uses "either gene <0.2" (union); "both" is the stricter
# homozygous-9p21 call. Run both for a sensitivity comparison.  argv[1] in {either,both}
NULL_MODE = sys.argv[1] if len(sys.argv) > 1 else "either"
N_PERM    = 50000
N_BOOT    = 20000
BLOCK     = 5000       # chunk perms/boots for memory safety
MIN_B     = 4
MIN_C     = 4
MAX_WORK  = 12

MODULES = {
    "FAK / focal-adhesion": ["PTK2","ITGAV","TLN1","VCL","FERMT2","ITGB5","ILK","PXN","BCAR1"],
    "YAP/TAZ-TEAD output":  ["WWTR1","YAP1","TEAD1","TEAD2","TEAD3","TEAD4"],
    "CTRL MTAP-synth-leth": ["PRMT5","MAT2A"],
    "CTRL neutral":         ["MTAP"],
}
GENES = [g for v in MODULES.values() for g in v]
GENE2MOD = {g:m for m,v in MODULES.items() for g in v}

# ----------------------------------------------------------------------------
# genotype groups per cell line (ModelID)
# ----------------------------------------------------------------------------
def load_groups():
    cn_id = pd.read_csv("depmap/OmicsCNGene.csv", nrows=0).columns[0]
    cn = pd.read_csv("depmap/OmicsCNGene.csv", index_col=cn_id,
                     usecols=[cn_id,"CDKN2A (1029)","CDKN2B (1030)","MTAP (4507)"])
    cn.columns = ["CDKN2A","CDKN2B","MTAP"]
    cn = cn.dropna()
    if NULL_MODE == "both":
        cdkn_null = (cn.CDKN2A < HOM) & (cn.CDKN2B < HOM)
    else:                                   # "either" == paper's canonical definition
        cdkn_null = (cn.CDKN2A < HOM) | (cn.CDKN2B < HOM)
    mtap_null = cn.MTAP < HOM
    grp = pd.Series("excluded", index=cn.index)            # ~cdkn & mtap-null -> excluded
    grp[cdkn_null & mtap_null]   = "A"
    grp[cdkn_null & ~mtap_null]  = "B"
    grp[~cdkn_null & ~mtap_null] = "C"
    model = pd.read_csv("depmap/Model.csv",
                        usecols=["ModelID","CCLEName","OncotreeLineage"]).set_index("ModelID")
    g = pd.DataFrame({"group": grp}).join(model, how="left")
    return g  # index ModelID; cols group, CCLEName, OncotreeLineage

# ----------------------------------------------------------------------------
# screen matrices: lines (ModelID) x genes, with group + lineage attached
# ----------------------------------------------------------------------------
def load_crispr(groups):
    hdr = pd.read_csv("depmap/CRISPRGeneEffect.csv", nrows=0).columns
    sym2lab = {}
    for lab in hdr[1:]:
        sym2lab.setdefault(lab.split(" (")[0], lab)
    labs = {g: sym2lab[g] for g in GENES if g in sym2lab}
    df = pd.read_csv("depmap/CRISPRGeneEffect.csv", index_col=hdr[0],
                     usecols=[hdr[0]]+list(labs.values()))
    df.columns = [c.split(" (")[0] for c in df.columns]
    df = df.join(groups[["group","OncotreeLineage"]], how="inner")
    return df, list(labs.keys())

def load_rnai(groups):
    mat = pd.read_csv("rnai/D2_combined_gene_dep_scores.csv", index_col=0)
    sym2lab = {}
    for lab in mat.index:
        sym2lab.setdefault(lab.split(" (")[0], lab)
    labs = {g: sym2lab[g] for g in GENES if g in sym2lab}
    sub = mat.loc[list(labs.values())].T          # CCLEName x genes
    sub.columns = list(labs.keys())
    # attach group + lineage via CCLEName
    ccle = groups.dropna(subset=["CCLEName"]).set_index("CCLEName")[["group","OncotreeLineage"]]
    sub = sub.join(ccle, how="inner")
    return sub, list(labs.keys())

# ----------------------------------------------------------------------------
# per-gene/contrast statistics (worker)
# ----------------------------------------------------------------------------
def _perm_p(test, comp, obs, n_perm, rng):
    pool = np.concatenate([test, comp]); n = pool.size; nt = test.size
    ge = 0; done = 0
    while done < n_perm:
        k = min(BLOCK, n_perm - done)
        idx = np.argsort(rng.random((k, n)), axis=1)
        perm = pool[idx]
        eff = np.median(perm[:, nt:], axis=1) - np.median(perm[:, :nt], axis=1)
        ge += int(np.sum(eff >= obs - 1e-12))
        done += k
    return (1 + ge) / (n_perm + 1)

def _boot_ci(test, comp, n_boot, rng):
    nt, nc = test.size, comp.size
    out = np.empty(n_boot); done = 0
    while done < n_boot:
        k = min(BLOCK, n_boot - done)
        ti = test[rng.integers(0, nt, size=(k, nt))]
        ci = comp[rng.integers(0, nc, size=(k, nc))]
        out[done:done+k] = np.median(ci, axis=1) - np.median(ti, axis=1)
        done += k
    return tuple(np.percentile(out, [2.5, 97.5]))

def worker(task):
    gene, screen, scope, contrast, test, comp = task
    test = np.asarray(test, float); comp = np.asarray(comp, float)
    test = test[~np.isnan(test)]; comp = comp[~np.isnan(comp)]
    res = dict(gene=gene, module=GENE2MOD[gene], screen=screen, scope=scope,
               contrast=contrast, n_test=test.size, n_comp=comp.size,
               med_test=np.nan, med_comp=np.nan, effect=np.nan,
               mw_p=np.nan, perm_p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    if test.size < MIN_B or comp.size < MIN_C:
        return res
    obs = float(np.median(comp) - np.median(test))     # >0 => test more dependent
    rng = np.random.default_rng(abs(hash((gene,screen,scope,contrast))) % (2**32))
    try:
        _, mw = mannwhitneyu(test, comp, alternative="less")   # test < comp
    except ValueError:
        mw = np.nan
    res.update(med_test=float(np.median(test)), med_comp=float(np.median(comp)),
               effect=obs, mw_p=float(mw),
               perm_p=_perm_p(test, comp, obs, N_PERM, rng))
    lo, hi = _boot_ci(test, comp, N_BOOT, rng)
    res.update(ci_lo=float(lo), ci_hi=float(hi))
    return res

# ----------------------------------------------------------------------------
def build_tasks(frame, genes, screen):
    tasks = []
    scopes = {"PAN": frame, "CNS": frame[frame.OncotreeLineage == "CNS/Brain"]}
    for scope, fr in scopes.items():
        A = fr[fr.group == "A"]; B = fr[fr.group == "B"]; C = fr[fr.group == "C"]
        for g in genes:
            tasks.append((g, screen, scope, "A_vs_C", A[g].values, C[g].values))
            tasks.append((g, screen, scope, "B_vs_C", B[g].values, C[g].values))
    return tasks, scopes

def stouffer(rows):
    """Combine one-sided perm p across screens; weight sqrt(n_test+n_comp)."""
    zs, ws = [], []
    for r in rows:
        p = min(max(r["perm_p"], 1e-9), 1 - 1e-9)
        zs.append(norm.isf(p)); ws.append(np.sqrt(r["n_test"] + r["n_comp"]))
    zs = np.array(zs); ws = np.array(ws)
    Z = float(np.sum(ws*zs) / np.sqrt(np.sum(ws**2)))
    return Z, float(norm.sf(Z))

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"loading genotype groups + screens ...  NULL_MODE='{NULL_MODE}' "
          f"(either = paper's canonical CN<0.2 in either gene; both = strict homozygous)",
          flush=True)
    groups = load_groups()
    crispr, cg = load_crispr(groups)
    rnai,   rg = load_rnai(groups)

    print("\n=== genotype group sizes (lines) ===")
    print(f"{'':14s}{'A null&MTAPnull':>16s}{'B null&MTAPok':>15s}{'C intact':>10s}")
    for name, fr in [("CRISPR PAN", crispr),
                     ("CRISPR CNS", crispr[crispr.OncotreeLineage=='CNS/Brain']),
                     ("RNAi   PAN", rnai),
                     ("RNAi   CNS", rnai[rnai.OncotreeLineage=='CNS/Brain'])]:
        vc = fr.group.value_counts()
        print(f"{name:14s}{vc.get('A',0):>16d}{vc.get('B',0):>15d}{vc.get('C',0):>10d}")
    print(f"\nCRISPR genes found: {cg}\nRNAi genes found:   {rg}")

    tasks  = build_tasks(crispr, cg, "CRISPR")[0] + build_tasks(rnai, rg, "RNAi")[0]
    print(f"\nrunning {len(tasks)} (gene x screen x scope x contrast) jobs "
          f"x {N_PERM} perms + {N_BOOT} boots on {MAX_WORK} cores ...", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORK) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 20 == 0: print(f"  {i}/{len(tasks)} done", flush=True)

    df = pd.DataFrame(results)
    df["null_mode"] = NULL_MODE
    df.to_csv(f"mtap_separation_permutation_{NULL_MODE}.csv", index=False)

    def show(scope, contrast):
        sub = df[(df.scope==scope)&(df.contrast==contrast)]
        print(f"\n{'='*108}\n{contrast}   scope={scope}   "
              f"(effect>0 => test-group MORE dependent than 9p21-intact C)\n{'='*108}")
        hdr = f"{'gene':8s}{'screen':7s}{'nT':>4s}{'nC':>5s}{'medT':>8s}{'medC':>8s}{'effect':>9s}{'MW p':>10s}{'perm p':>10s}{'boot 95% CI':>20s}  flag"
        last_mod = None
        for mod in MODULES:
            block = sub[sub.module==mod]
            if block.empty: continue
            print(f"\n-- {mod} --"); print(hdr)
            for g in MODULES[mod]:
                for screen in ["CRISPR","RNAi"]:
                    r = block[(block.gene==g)&(block.screen==screen)]
                    if r.empty: continue
                    r = r.iloc[0]
                    if np.isnan(r.effect):
                        print(f"{g:8s}{screen:7s}  (too few: nT={int(r.n_test)}/nC={int(r.n_comp)})")
                        continue
                    sig = (r.perm_p < 0.05) and (r.effect > 0) and (r.ci_lo > 0)
                    flag = " <== survives" if sig else ""
                    print(f"{g:8s}{screen:7s}{int(r.n_test):>4d}{int(r.n_comp):>5d}"
                          f"{r.med_test:>8.3f}{r.med_comp:>8.3f}{r.effect:>9.3f}"
                          f"{r.mw_p:>10.3g}{r.perm_p:>10.3g}"
                          f"   [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]{flag}")

    show("PAN", "A_vs_C")   # confounded baseline (both 9p21 genes lost)
    show("PAN", "B_vs_C")   # DECONFOUNDED headline (MTAP intact)
    show("CNS", "B_vs_C")   # GBM/CNS-restricted deconfounded

    # pooled CRISPR+RNAi for the decisive B_vs_C contrast
    print(f"\n{'='*72}\nPOOLED CRISPR+RNAi  (Stouffer Z, weight sqrt(nT+nC))  B_vs_C\n{'='*72}")
    for scope in ["PAN","CNS"]:
        print(f"\n-- scope={scope} --")
        print(f"{'gene':8s}{'module':24s}{'#screens':>9s}{'Stouffer Z':>12s}{'pooled p':>12s}  flag")
        for g in GENES:
            rows = df[(df.scope==scope)&(df.contrast=='B_vs_C')&(df.gene==g)
                      &(~df.perm_p.isna())].to_dict("records")
            if not rows: continue
            Z, p = stouffer(rows)
            flag = " <== robust" if (p < 0.05 and Z > 0) else ""
            print(f"{g:8s}{GENE2MOD[g]:24s}{len(rows):>9d}{Z:>12.2f}{p:>12.3g}{flag}")

    print(f"\nwrote mtap_separation_permutation_{NULL_MODE}.csv")
