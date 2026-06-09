# Obtaining the raw input data

This project re-analyzes publicly available datasets. The raw data are **not** redistributed in this repository. Download them from the original sources below and adjust the paths in the scripts accordingly.

| Dataset | Used for | Source / accession |
|---|---|---|
| DepMap CRISPR (Chronos), gene copy number, Model metadata | FAK dependency; MTAP confound | <https://depmap.org/portal/download> — `CRISPRGeneEffect.csv`, `OmicsCNGene.csv`, `Model.csv` |
| DepMap DEMETER2 RNAi | Orthogonal RNAi validation | <https://depmap.org> — `D2_combined_gene_dep_scores.csv` |
| PRISM repurposing (LFC) | FAK inhibitor GSK2256098 | <https://depmap.org> — PRISM `LFC_COLLAPSED.csv` + metadata |
| GDSC1 / GDSC2 fitted dose-response | FAK inhibitor PF-562271 | <https://www.cancerrxgene.org/downloads> |
| TCGA-GBM (expression, somatic, clinical/survival) | Eligibility, resistance, survival | <https://portal.gdc.cancer.gov> ; <https://www.cbioportal.org> |
| CPTAC-GBM proteomics / phosphoproteomics | FAK abundance + Y397 phospho | <https://proteomics.cancer.gov/programs/cptac> ; cBioPortal |
| MSK-IMPACT (GBM) | Eligibility replication | <https://www.cbioportal.org> |
| Neftel et al. 2019 single-cell GBM atlas | Mesenchymal-state deconfound | GEO **GSE131928** (Smart-seq2 + 10x TPM matrices, tumor metadata) |
| GWAS Catalog (glioma/glioblastoma) | Germline association context | <https://www.ebi.ac.uk/gwas> |
| gnomAD v4 | Variant frequency / constraint | <https://gnomad.broadinstitute.org> |
| **FinnGen R13** | MUTYH germline lead (Finding 3) | **Access-controlled** — <https://www.finngen.fi> (not redistributable) |

Scripts in `../scripts/` reference an analysis root of `/home/brad/finngen-triage`; edit those paths to point at wherever you place the downloaded files.
