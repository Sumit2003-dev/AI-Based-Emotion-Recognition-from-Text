# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report

**Date:** 2026-11-02  
**Dataset:** `data/hinglish_eval_benchmark.json` (200 curated samples)  

## Executive Summary

- **Overall Accuracy:** `77.50%` (155/200)
- **Macro F1:** `0.7032`
- **Weighted F1:** `0.7731`
- **Brier Score:** `0.3643` *(Measures probability accuracy; lower is better)*
- **Expected Calibration Error (ECE):** `4.31%` *(Quantifies probability reliability)*

## Category Breakdown

| Category | Tested | Correct | Accuracy | Description |
|---|---|---|---|---|
| `contrast` | 35 | 28 | **80.0%** | Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome |
| `locative_par` | 20 | 19 | **95.0%** | Sentences using 'par' as locative postposition (on/at) rather than contrast |
| `mixed_emotions` | 30 | 16 | **53.3%** | Ambivalent emotions, bittersweet moments, dual primary/secondary states |
| `negation` | 30 | 22 | **73.3%** | Negation of negative states ('daant nahi padi') vs negated positive states |
| `pure_emotions` | 50 | 43 | **86.0%** | Canonical Hinglish & English emotional expressions |
| `sarcasm` | 35 | 27 | **77.1%** | Sarcastic praise, ironic frustration, over-enthusiastic cynicism |

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy** | 0.840 | 0.857 | 0.848 | 49 |
| **Sadness** | 0.711 | 0.681 | 0.696 | 47 |
| **Anger** | 0.875 | 0.764 | 0.816 | 55 |
| **Fear** | 0.636 | 0.875 | 0.737 | 8 |
| **Surprise** | 0.444 | 0.800 | 0.571 | 5 |
| **Love** | 1.000 | 0.286 | 0.444 | 7 |
| **Disgust** | 0.571 | 0.800 | 0.667 | 5 |
| **Neutral** | 0.786 | 0.917 | 0.846 | 24 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 1 | 28.0% | 0.0% | 28.0% |
| `[0.3, 0.4)` | 4 | 32.8% | 25.0% | 7.8% |
| `[0.4, 0.5)` | 22 | 45.0% | 63.6% | 18.7% |
| `[0.5, 0.6)` | 22 | 54.9% | 54.5% | 0.3% |
| `[0.6, 0.7)` | 29 | 65.6% | 75.9% | 10.2% |
| `[0.7, 0.8)` | 32 | 74.6% | 75.0% | 0.4% |
| `[0.8, 0.9)` | 40 | 85.3% | 85.0% | 0.3% |
| `[0.9, 1.0)` | 50 | 94.7% | 96.0% | 1.3% |
