#!/usr/bin/env python3
"""
Push A1: GDSC FAK-inhibitor dose-response test.

Question: do CDKN2A/B-homozygously-deleted lines show SELECTIVE sensitivity to
a POTENT FAK inhibitor (PF-562271), using proper dose-response (LN_IC50 / AUC)?
This directly addresses PRISM's weak-tool gap (PRISM only had GSK2256098 at 2.5uM,
essentially inert across CNS lines).

FAK compounds in GDSC8.5:
  DRUG_ID 158  = PF-562271  (potent FAK/FAK2 inhibitor)  -- the proper tool
  DRUG_ID 1776 = GSK2256098C (FAK1)                       -- same class as PRISM hit

CDKN2A/B status comes from our DepMap OmicsCNGene calls (HOMDEL_THRESH=0.2),
joined to GDSC by SangerModelID / COSMICID.

Lower AUC  = more drug killing = MORE sensitive.
Lower LN_IC50 = MORE sensitive.
Hypothesis: null lines are MORE sensitive  ->  null AUC < intact AUC.
"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

HOMDEL = 0.2
FAK_DRUGS = {158: "PF-562271", 1776: "GSK2256098C"}

# ---- 1. CDKN2A/B status from DepMap CN ----
id_col = pd.read_csv("depmap/OmicsCNGene.csv", nrows=0).columns[0]
cn = pd.read_csv("depmap/OmicsCNGene.csv",
                 usecols=[id_col, "CDKN2A (1029)", "CDKN2B (1030)"],
                 index_col=id_col)
cn.columns = ["CDKN2A", "CDKN2B"]
null_mask = (cn["CDKN2A"] < HOMDEL) & (cn["CDKN2B"] < HOMDEL)
status = pd.Series(np.where(null_mask, "null", "intact"), index=cn.index, name="status")

# ---- 2. Map ModelID -> Sanger/COSMIC + lineage ----
model = pd.read_csv("depmap/Model.csv",
                    usecols=["ModelID", "OncotreeLineage", "SangerModelID", "COSMICID"])
model = model.set_index("ModelID").join(status)
model = model.dropna(subset=["status"])
# keep mapping tables
by_sanger = model.dropna(subset=["SangerModelID"]).set_index("SangerModelID")[["status", "OncotreeLineage"]]
by_cosmic = model.dropna(subset=["COSMICID"]).copy()
by_cosmic["COSMICID"] = by_cosmic["COSMICID"].astype(float).astype("Int64").astype(str)
by_cosmic = by_cosmic.set_index("COSMICID")[["status", "OncotreeLineage"]]

print(f"DepMap lines classified: {(status=='null').sum()} null / {(status=='intact').sum()} intact")

# ---- 3. Load GDSC fitted dose-response, both datasets ----
frames = []
for fn in ["gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx",
           "gdsc/GDSC1_fitted_dose_response_27Oct23.xlsx"]:
    df = pd.read_excel(fn, usecols=["DATASET","COSMIC_ID","CELL_LINE_NAME","SANGER_MODEL_ID",
                                     "TCGA_DESC","DRUG_ID","DRUG_NAME","LN_IC50","AUC","Z_SCORE"])
    frames.append(df)
gdsc = pd.concat(frames, ignore_index=True)
fak = gdsc[gdsc["DRUG_ID"].isin(FAK_DRUGS)].copy()
print(f"\nGDSC FAK-drug measurements: {len(fak)} rows")
print(fak.groupby(["DRUG_ID","DRUG_NAME","DATASET"]).size())

# ---- 4. Attach CDKN2A/B status (Sanger first, COSMIC fallback) ----
def attach(row):
    sm = row["SANGER_MODEL_ID"]
    if pd.notna(sm) and sm in by_sanger.index:
        r = by_sanger.loc[sm]
        return pd.Series([r["status"], r["OncotreeLineage"]])
    cid = row["COSMIC_ID"]
    if pd.notna(cid):
        cid = str(int(cid))
        if cid in by_cosmic.index:
            r = by_cosmic.loc[cid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            return pd.Series([r["status"], r["OncotreeLineage"]])
    return pd.Series([np.nan, np.nan])

fak[["status","lineage"]] = fak.apply(attach, axis=1)
fak = fak.dropna(subset=["status"])
print(f"\nFAK measurements with CDKN2A/B status: {len(fak)}")

# ---- 5. Compare null vs intact, pan-cancer and CNS-only ----
def compare(sub, label, metric):
    n = sub[sub.status=="null"][metric].dropna()
    i = sub[sub.status=="intact"][metric].dropna()
    if len(n) < 3 or len(i) < 3:
        print(f"  [{label}] {metric}: too few (null={len(n)}, intact={len(i)})")
        return None
    u,p = mannwhitneyu(n, i, alternative="two-sided")
    # one-sided: null MORE sensitive => null metric LOWER
    _,p_lt = mannwhitneyu(n, i, alternative="less")
    print(f"  [{label}] {metric}: null n={len(n)} med={n.median():.3f} | "
          f"intact n={len(i)} med={i.median():.3f} | "
          f"two-sided p={p:.3g} | null<intact p={p_lt:.3g}")
    return p

for drug_id, dname in FAK_DRUGS.items():
    d = fak[fak["DRUG_ID"]==drug_id]
    if d.empty: continue
    print(f"\n=== {dname} (DRUG_ID {drug_id}) ===")
    for metric in ["AUC","LN_IC50","Z_SCORE"]:
        print(f" PAN-CANCER:")
        compare(d, "pan", metric)
        cns = d[d["lineage"]=="CNS/Brain"]
        print(f" CNS/Brain only (n_lines={cns['CELL_LINE_NAME'].nunique()}):")
        compare(cns, "CNS", metric)

# lineage label check
print("\nLineage values among FAK rows:", sorted(fak["lineage"].dropna().unique())[:20])

fak.to_csv("gdsc_fak_sensitivity.csv", index=False)
print("\nwrote gdsc_fak_sensitivity.csv")
