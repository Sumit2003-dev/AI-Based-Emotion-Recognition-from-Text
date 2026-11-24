# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report

**Date:** 2026-11-02  
**Dataset:** `data/hinglish_eval_benchmark.json` (200 curated samples)  

## Executive Summary

- **Overall Accuracy:** `68.50%` (137/200)
- **Macro F1:** `0.6362`
- **Weighted F1:** `0.7005`
- **Brier Score:** `0.4543` *(Measures probability accuracy; lower is better)*
- **Expected Calibration Error (ECE):** `6.65%` *(Quantifies probability reliability)*

## Category Breakdown

| Category | Tested | Correct | Accuracy | Description |
|---|---|---|---|---|
| `contrast` | 35 | 25 | **71.4%** | Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome |
| `locative_par` | 20 | 15 | **75.0%** | Sentences using 'par' as locative postposition (on/at) rather than contrast |
| `mixed_emotions` | 30 | 12 | **40.0%** | Ambivalent emotions, bittersweet moments, dual primary/secondary states |
| `negation` | 30 | 18 | **60.0%** | Negation of negative states ('daant nahi padi') vs negated positive states |
| `pure_emotions` | 50 | 40 | **80.0%** | Canonical Hinglish & English emotional expressions |
| `sarcasm` | 35 | 27 | **77.1%** | Sarcastic praise, ironic frustration, over-enthusiastic cynicism |

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy** | 0.800 | 0.735 | 0.766 | 49 |
| **Sadness** | 0.750 | 0.574 | 0.651 | 47 |
| **Anger** | 0.800 | 0.727 | 0.762 | 55 |
| **Fear** | 0.583 | 0.875 | 0.700 | 8 |
| **Surprise** | 0.200 | 0.800 | 0.320 | 5 |
| **Love** | 0.800 | 0.571 | 0.667 | 7 |
| **Disgust** | 0.417 | 1.000 | 0.588 | 5 |
| **Neutral** | 0.700 | 0.583 | 0.636 | 24 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 5 | 26.0% | 20.0% | 6.0% |
| `[0.3, 0.4)` | 19 | 34.7% | 36.8% | 2.1% |
| `[0.4, 0.5)` | 29 | 44.4% | 37.9% | 6.5% |
| `[0.5, 0.6)` | 32 | 55.2% | 65.6% | 10.4% |
| `[0.6, 0.7)` | 22 | 65.6% | 72.7% | 7.1% |
| `[0.7, 0.8)` | 39 | 74.3% | 84.6% | 10.4% |
| `[0.8, 0.9)` | 31 | 84.4% | 87.1% | 2.7% |
| `[0.9, 1.0)` | 23 | 95.3% | 91.3% | 4.0% |
