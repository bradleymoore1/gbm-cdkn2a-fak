# Computational triage of therapeutic and germline vulnerabilities in IDH-wildtype glioblastoma

Analysis code and derived results for the preprint:

> **Computational triage of therapeutic and germline vulnerabilities in IDH-wildtype glioblastoma: a focal-adhesion dependency in CDKN2A/B-deleted tumors**
> Brad Moore. *bioRxiv* (2026). DOI: _to be added on posting_

This is a hypothesis-generating, computational re-analysis of publicly available functional-genomic, transcriptomic, pharmacologic, clinical, and population-genetic datasets. The repository holds the analysis scripts and the **aggregate, derived** result tables that underlie the figures and reported numbers. Raw third-party data and access-controlled data are **not** redistributed here (see *Data availability* below).

## Findings, in brief

1. **A focal-adhesion / FAK dependency in CDKN2A/B-null GBM** — supported by five complementary lines of evidence: selective CRISPR essentiality, orthogonal RNAi, selective sensitivity to a potent FAK inhibitor (PF-562271), tumor mRNA upregulation, and elevated FAK abundance + Y397 autophosphorylation in tumor proteomics. *Major caveat:* CDKN2A/B co-deletes with the adjacent 9p21 gene MTAP in ~92% of null lines and the two are statistically inseparable in DepMap, so the dependency is most defensibly described as **9p21-codeletion-associated** and awaits isogenic wet-lab confirmation.
2. **A CDK4/6-inhibitor-eligible subset with an explicit resistance map** — a large, molecularly-definable eligible subset (IDH-wt, RB1-intact, CDK4/6-axis-activated) that shows no survival advantage and frequently carries PTEN/PI3K-axis alterations that plausibly explain prior null monotherapy trials.
3. **A Finnish-founder MUTYH germline lead** (p.Gly155Asp, rs587781864) — a suggestive, replication-grade association with GBM in FinnGen R13 that is internally validated by expected colorectal signals but does not survive multiple-testing correction.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Analysis and figure-generation scripts (Python). |
| `results/` | Derived, aggregate result tables (CSV/TXT) computed from public data. |
| `figures/` | Main and supplementary figures (PDF). |
| `data/` | Pointers for obtaining the raw input datasets (not redistributed here). |
| `GBM_preprint.pdf`, `GBM_preprint.md` | The manuscript. |

## Data availability

This repository does **not** redistribute raw third-party data. All primary datasets are publicly available from their original sources — see [`data/README.md`](data/README.md) for download pointers:

- **DepMap** (CRISPR Chronos, DEMETER2 RNAi, copy number, PRISM) — <https://depmap.org>
- **GDSC** drug response — <https://www.cancerrxgene.org>
- **TCGA-GBM** and **CPTAC-GBM** — via the GDC (<https://portal.gdc.cancer.gov>) and cBioPortal (<https://www.cbioportal.org>)
- **MSK-IMPACT** — via cBioPortal
- **Neftel et al. 2019** single-cell GBM atlas — GEO accession **GSE131928**
- **GWAS Catalog** — <https://www.ebi.ac.uk/gwas>
- **gnomAD** — <https://gnomad.broadinstitute.org>

**FinnGen note.** Finding 3 uses **FinnGen R13**, which is access-controlled. FinnGen individual-level and derived data are **not** included and cannot be redistributed under FinnGen's data-access terms. The scripts that operate on FinnGen data (`finngen_gbm_burden.py`, `mutyh_phenome_wide.py`, `glioma_germline_scan.py`) are provided for methodological transparency and require approved FinnGen access to run.

## Reproducing

Scripts are standalone Python (pandas / numpy / scipy / matplotlib; survival analyses use `lifelines`; single-cell uses `scanpy`/`anndata`). They expect the raw inputs described in `data/README.md`. **File paths in the scripts reflect the original analysis environment** (`/home/brad/finngen-triage`) and must be adjusted to your local data layout.

## Citing

If you use this code or the derived tables, please cite the preprint (DOI above).

## License

- **Code** (`scripts/`): MIT License — see [`LICENSE`](LICENSE).
- **Manuscript text and figures**: CC-BY 4.0, consistent with the bioRxiv deposit.

---
*Dedicated to James E. Moore (1961–2025).*
