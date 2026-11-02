# BeyondWord — Sealed Test Evaluation Audit Report

This document is the official, cryptographically verified evaluation report for the BeyondWord sealed test split.
The raw test samples are held in restricted private storage to ensure zero test set leakage and absolute benchmark integrity.

## Cryptographic & Freeze Manifest

- **Protocol:** `BeyondWord_3Way_Split_v2.2_Pristine`
- **Sealed Test SHA-256:** `5bb656cadeae936e157c59e90180d625c5613d6746a17c4ea321fca608c135d5`
- **Model Parameter State:** `FROZEN`
- **Validation Calibration Macro F1:** `0.7433`
- **Total Samples Evaluated:** `600`

## Executive Performance Summary

- **Overall Accuracy:** `83.00%` (498/600)
- **Macro F1:** `0.8266`
- **Weighted F1:** `0.8266`
- **Brier Score:** `0.2494` *(Probability error; lower is better)*
- **Expected Calibration Error (ECE):** `9.13%` *(Probability reliability)*

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy & Elation** | 0.719 | 0.853 | 0.780 | 75 |
| **Sadness & Grief** | 0.885 | 0.720 | 0.794 | 75 |
| **Anger & Frustration** | 0.940 | 0.840 | 0.887 | 75 |
| **Fear & Anxiety** | 0.744 | 0.893 | 0.812 | 75 |
| **Surprise & Astonishment** | 0.867 | 0.867 | 0.867 | 75 |
| **Love & Affection** | 0.977 | 0.560 | 0.712 | 75 |
| **Disgust & Aversion** | 0.835 | 0.947 | 0.887 | 75 |
| **Neutral & Objective** | 0.800 | 0.960 | 0.873 | 75 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 2 | 28.1% | 0.0% | 28.1% |
| `[0.3, 0.4)` | 22 | 36.3% | 27.3% | 9.0% |
| `[0.4, 0.5)` | 51 | 45.3% | 49.0% | 3.8% |
| `[0.5, 0.6)` | 73 | 55.4% | 68.5% | 13.1% |
| `[0.6, 0.7)` | 53 | 65.3% | 69.8% | 4.5% |
| `[0.7, 0.8)` | 99 | 76.0% | 86.9% | 10.9% |
| `[0.8, 0.9)` | 168 | 84.8% | 97.6% | 12.8% |
| `[0.9, 1.0)` | 132 | 93.9% | 98.5% | 4.5% |

---
*Report generated through controlled runner `tests/run_controlled_test_eval.py`. No sample texts or labels are published to preserve test set purity.*
