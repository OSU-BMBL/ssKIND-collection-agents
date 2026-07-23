# Table 1 — AI-agent vs. manual curation, per scope

| modality | disease | n_manual | n_ai | overlap | ai_only | manual_only | recall_pct | ci95_low | ci95_high | expansion | ssread_legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single-cell | Multiple sclerosis | 52 | 84 | 50 | 34 | 2 | 96.2 | 87.0 | 98.9 | 1.62 | 1 |
| Single-cell | Alzheimer's | 36 | 108 | 32 | 76 | 4 | 88.9 | 74.7 | 95.6 | 3.00 | 57 |
| Single-cell | Parkinson's | 24 | 66 | 22 | 44 | 2 | 91.7 | 74.2 | 97.7 | 2.75 | 3 |
| Single-cell | ALS | 14 | 33 | 13 | 20 | 1 | 92.9 | 68.5 | 98.7 | 2.36 | 0 |
| Single-cell | Huntington's | 12 | 17 | 9 | 8 | 3 | 75.0 | 46.8 | 91.1 | 1.42 | 0 |
| Single-cell | Frontotemporal dementia | 7 | 14 | 7 | 7 | 0 | 100.0 | 64.6 | 100.0 | 2.00 | 2 |
| Single-cell | Prion disease | 2 | 2 | 2 | 0 | 0 | 100.0 | 34.2 | 100.0 | 1.00 | 0 |
| Single-cell | Spinal muscular atrophy | 2 | 2 | 2 | 0 | 0 | 100.0 | 34.2 | 100.0 | 1.00 | 0 |
| Single-cell | Spinocerebellar ataxia | 2 | 3 | 2 | 1 | 0 | 100.0 | 34.2 | 100.0 | 1.50 | 0 |
| Spatial | Alzheimer's | 33 | 57 | 30 | 27 | 3 | 90.9 | 76.4 | 96.9 | 1.73 | 9 |
| Spatial | Multiple sclerosis | 7 | 12 | 6 | 6 | 1 | 85.7 | 48.7 | 97.4 | 1.71 | 0 |
| Spatial | Parkinson's | 6 | 20 | 4 | 16 | 2 | 66.7 | 30.0 | 90.3 | 3.33 | 1 |
| Spatial | ALS | 3 | 4 | 2 | 2 | 1 | 66.7 | 20.8 | 93.9 | 1.33 | 0 |
| Spatial | Huntington's | 1 | 1 | 1 | 0 | 0 | 100.0 | 20.7 | 100.0 | 1.00 | 0 |
| Spatial | Prion disease | 0 | 1 | 0 | 1 | 0 | n/a | n/a | n/a | n/a | 0 |
| Spatial | Spinal muscular atrophy | 0 | 2 | 0 | 2 | 0 | n/a | n/a | n/a | n/a | 0 |
| **ALL** | **Total** | 201 | 426 | 182 | 244 | 19 | 90.5 | 85.7 | 93.9 | 2.12 | 73 |

**Unique papers** (deduplicated across scopes): both 175, AI only 222, manual only 17, ssREAD-only 60.

*Precision and F1 are deliberately absent: the AI-only papers are unvalidated, so only recall against the manual set is estimable.*
