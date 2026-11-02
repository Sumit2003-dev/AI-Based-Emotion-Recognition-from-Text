# BeyondWord — Sealed Test Evaluation Audit Report

This document is the official, cryptographically verified evaluation report for the BeyondWord sealed test split.
The raw test samples are held in restricted private storage to ensure zero test set leakage and absolute benchmark integrity.

## Cryptographic & Freeze Manifest

- **Protocol:** `BeyondWord_3Way_Split_v2.2`
- **Sealed Test SHA-256:** `295e7e2bd8e55465fe8147741c596c46ba8f090ca4c97669a4477feaaf5234b8`
- **Model Parameter State:** `FROZEN`
- **Validation Calibration Macro F1:** `0.7433`
- **Total Samples Evaluated:** `600`

## Executive Performance Summary

- **Overall Accuracy:** `64.67%` (388/600)
- **Macro F1:** `0.6444`
- **Weighted F1:** `0.6444`
- **Brier Score:** `0.3994` *(Probability error; lower is better)*
- **Expected Calibration Error (ECE):** `12.93%` *(Probability reliability)*

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy & Elation** | 0.812 | 0.693 | 0.748 | 75 |
| **Sadness & Grief** | 0.667 | 0.640 | 0.653 | 75 |
| **Anger & Frustration** | 0.549 | 0.373 | 0.444 | 75 |
| **Fear & Anxiety** | 0.838 | 0.827 | 0.832 | 75 |
| **Surprise & Astonishment** | 0.569 | 0.493 | 0.529 | 75 |
| **Love & Affection** | 1.000 | 0.507 | 0.673 | 75 |
| **Disgust & Aversion** | 0.595 | 0.920 | 0.723 | 75 |
| **Neutral & Objective** | 0.450 | 0.720 | 0.554 | 75 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 7 | 27.9% | 57.1% | 29.3% |
| `[0.3, 0.4)` | 38 | 35.7% | 34.2% | 1.5% |
| `[0.4, 0.5)` | 67 | 45.3% | 44.8% | 0.6% |
| `[0.5, 0.6)` | 76 | 54.8% | 32.9% | 21.9% |
| `[0.6, 0.7)` | 89 | 64.4% | 42.7% | 21.7% |
| `[0.7, 0.8)` | 72 | 74.2% | 43.1% | 31.1% |
| `[0.8, 0.9)` | 80 | 86.1% | 96.2% | 10.1% |
| `[0.9, 1.0)` | 171 | 94.7% | 99.4% | 4.7% |

---
*Report generated through controlled runner `tests/run_controlled_test_eval.py`. No sample texts or labels are published to preserve test set purity.*
