# Data Availability

## EM-DAT Disaster Events

**Source:** EM-DAT — The International Disaster Database, Centre for Research on the Epidemiology of Disasters (CRED), Université catholique de Louvain, Brussels, Belgium.  
**URL:** https://www.emdat.be  
**Coverage:** Sri Lanka physical disasters, 2000–2025.  
**Access:** Freely available after registration at emdat.be.  
**File:** `pipeline/data/raw/Disaster data.xlsx` (not included in this repository due to EM-DAT terms of service).

## Tourist Arrivals

**Source:** Sri Lanka Tourism Development Authority (SLTDA), Annual Statistical Reports.  
**URL:** https://www.sltda.gov.lk/en/statistical-reports  
**Coverage:** Monthly international tourist arrivals to Sri Lanka, January 2000 – December 2025.  
**Note:** 2023–2025 figures are provisional and subject to revision.  
**File:** `pipeline/data/raw/tourist_arrivals.csv` (not included; obtain from SLTDA).

## Macro-Economic Indicators

**Source:** World Bank Open Data (GDP growth rate), via `feature_engineering.py`.  
**URL:** https://data.worldbank.org  
**Coverage:** Sri Lanka annual GDP growth, 2000–2024.  
**Access:** Publicly available under Creative Commons Attribution 4.0 (CC-BY 4.0).

## Reproducibility

With the two data files above placed in `pipeline/data/raw/`, running:

```bash
pip install -r pipeline/requirements.txt
cd pipeline && python run_experiments.py
```

will reproduce all results in `pipeline/outputs/` to 4 decimal places (seed=42, pinned environment — see `pipeline/docs/VERSION_SENSITIVITY.md`).
