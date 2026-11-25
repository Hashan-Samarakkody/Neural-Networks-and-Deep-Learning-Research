# Data Provenance

All input files for the tourism recovery duration pipeline.

## EM-DAT Disaster Events

| Field | Value |
|-------|-------|
| File | `pipeline/data/raw/Disaster data.xlsx` |
| Source | EM-DAT: The International Disaster Database, Centre for Research on the Epidemiology of Disasters (CRED), Université catholique de Louvain, Brussels, Belgium |
| URL | https://www.emdat.be |
| Coverage | Sri Lanka (LKA), 2000–2025, all disaster types |
| Accessed | 2026-10 |
| License | EM-DAT data may not be redistributed. Provide download instructions for replication: register at https://www.emdat.be/data and filter: Country = Sri Lanka, Start year = 2000. |
| Notes | Raw extract: 86 records. After taxonomy filter (physical hazards only, excluding epidemics): 80 records. After compound event clustering (30-day window): 65 events (52 single, 11 pairs of 2, 2 triples). |

## SLTDA Monthly Tourist Arrivals

| Field | Value |
|-------|-------|
| File | `pipeline/data/raw/tourist_arrivals.csv` |
| Source | Sri Lanka Tourism Development Authority (SLTDA), Monthly Statistical Bulletin |
| URL | https://www.sltda.gov.lk/en/research-and-publications |
| Coverage | Sri Lanka, 2000–2025, monthly international tourist arrivals |
| Accessed | 2026-10 |
| License | Public statistical release. Cite as: SLTDA (2026). Monthly Tourist Arrivals, Colombo. |

## SLTDA District Room Shares (DTII)

| Field | Value |
|-------|-------|
| Source | SLTDA Annual Statistical Report, registered accommodation rooms by district |
| Edition | SLTDA Annual Statistical Report 2023, Table 4.1 (Registered Accommodation Establishments) |
| URL | https://www.sltda.gov.lk/en/research-and-publications |
| Notes | 23 districts listed; shares normalised to sum = 1.0. Configuration in `pipeline/configs/model_config.yaml` under `dtii:`. The sum of the listed shares is 0.99 due to rounding; normalised in code. |

## Macroeconomic Indicators

| Field | Value |
|-------|-------|
| File | `pipeline/data/raw/macro_indicators.csv` |
| Coverage | 2000–2025 |

| Indicator | WDI Code | Source | Notes |
|-----------|----------|--------|-------|
| gdp_growth | NY.GDP.MKTP.KD.ZG | World Bank WDI | Sri Lanka GDP growth (annual %) |
| inflation | FP.CPI.TOTL.ZG | World Bank WDI | CPI inflation (annual %) |
| exr_lkr_usd | PA.NUS.FCRF | World Bank WDI | Official exchange rate LKR/USD (annual average) |
| global_tourism_idx | ST.INT.ARVL | UNWTO | World international tourist arrivals, rebased to 2000 = 100 |

**Vintage note:** 2024 and 2025 values are IMF/World Bank projections as of November 2026 (marked `vintage=projection` in the CSV). Values for 2000–2023 are actuals. The annual value for the event year is used as a feature; it is published after the event ends, introducing a minor ex-ante violation for events in Q1 of the event year. A sensitivity with `macro_lag_years=1` (previous year's value) is available via config.

**World Bank access:** https://data.worldbank.org/indicator/{CODE}?locations=LK — accessed 2026-11-01.
