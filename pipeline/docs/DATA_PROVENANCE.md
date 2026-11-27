# Data Provenance

## Tourist Arrivals

| Year range | Status | Source |
|------------|--------|--------|
| 2000–2022 | Final | SLTDA Annual Statistical Reports |
| 2023–2024 | Provisional | SLTDA Monthly Bulletins (subject to revision) |
| 2025 | Provisional | SLTDA press releases / estimates |

**Note:** 2025 arrivals data is provisional and based on preliminary SLTDA figures.
Revised figures may alter recovery targets for events starting in 2024–2025.

## EM-DAT Disaster Events

| Year range | Status | Source |
|------------|--------|--------|
| 2000–2024 | Final | EM-DAT download (accessed November 2026) |
| 2025 | Provisional | EM-DAT; some events may be pending validation |

## Macro-Economic Indicators

| Indicator | Lag | Years | Source |
|-----------|-----|-------|--------|
| GDP growth rate | 1 year | 2000–2024 | World Bank WDI |
| Global tourism index | 0 years | 2000–2024 | UNWTO / World Bank |

### Macro Lag Sensitivity

GDP growth is used with a 1-year lag (disaster year t uses GDP growth for year t-1)
because full-year GDP figures are only available with a 12-month lag.

Sensitivity check: using lag=0 (same-year GDP) vs lag=1 (default) vs lag=2.
Results logged in `feature_engineering.py` comments and in `feature_diagnostics.json`.

## Right-Censoring

Events where tourism did not recover to 90% of baseline within 12 months are
right-censored at T_rec = 12. These are **not** treated as 12-month recoveries
in regression; the survival models (CoxPH, CoxnetAFT) handle censoring explicitly.

## Clustering

EM-DAT events starting within 30 days of each other (same disaster type) are
merged into a single composite event. See `data_loader.py:_cluster_compound_events`.
