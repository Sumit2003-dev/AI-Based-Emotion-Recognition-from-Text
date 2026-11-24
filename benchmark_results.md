# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report

**Date:** 2026-11-02  
**Dataset:** `data/hinglish_eval_benchmark.json` (200 curated samples)  

## Executive Summary

- **Overall Accuracy:** `95.00%` (190/200)
- **Macro F1:** `0.9450`
- **Weighted F1:** `0.9500`
- **Brier Score:** `0.1974` *(Measures probability accuracy; lower is better)*
- **Expected Calibration Error (ECE):** `21.41%` *(Quantifies probability reliability)*

## Category Breakdown

| Category | Tested | Correct | Accuracy | Description |
|---|---|---|---|---|
| `contrast` | 35 | 35 | **100.0%** | Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome |
| `locative_par` | 20 | 20 | **100.0%** | Sentences using 'par' as locative postposition (on/at) rather than contrast |
| `mixed_emotions` | 30 | 23 | **76.7%** | Ambivalent emotions, bittersweet moments, dual primary/secondary states |
| `negation` | 30 | 28 | **93.3%** | Negation of negative states ('daant nahi padi') vs negated positive states |
| `pure_emotions` | 50 | 50 | **100.0%** | Canonical Hinglish & English emotional expressions |
| `sarcasm` | 35 | 34 | **97.1%** | Sarcastic praise, ironic frustration, over-enthusiastic cynicism |

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy** | 0.941 | 0.980 | 0.960 | 49 |
| **Sadness** | 0.976 | 0.872 | 0.921 | 47 |
| **Anger** | 0.947 | 0.982 | 0.964 | 55 |
| **Fear** | 1.000 | 1.000 | 1.000 | 8 |
| **Surprise** | 0.714 | 1.000 | 0.833 | 5 |
| **Love** | 1.000 | 0.857 | 0.923 | 7 |
| **Disgust** | 1.000 | 1.000 | 1.000 | 5 |
| **Neutral** | 0.958 | 0.958 | 0.958 | 24 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 1 | 28.6% | 100.0% | 71.4% |
| `[0.3, 0.4)` | 8 | 34.7% | 75.0% | 40.3% |
| `[0.4, 0.5)` | 14 | 44.6% | 92.9% | 48.3% |
| `[0.5, 0.6)` | 24 | 54.9% | 95.8% | 40.9% |
| `[0.6, 0.7)` | 22 | 65.4% | 95.5% | 30.0% |
| `[0.7, 0.8)` | 44 | 74.3% | 95.5% | 21.2% |
| `[0.8, 0.9)` | 34 | 84.5% | 100.0% | 15.5% |
| `[0.9, 1.0)` | 53 | 96.4% | 94.3% | 2.1% |
