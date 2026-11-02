# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report

**Date:** 2026-11-02  
**Dataset:** `data/hinglish_eval_benchmark.json` (200 curated samples)  

## Executive Summary

- **Overall Accuracy:** `79.50%` (159/200)
- **Macro F1:** `0.7202`
- **Weighted F1:** `0.7933`
- **Brier Score:** `0.3391` *(Measures probability accuracy; lower is better)*
- **Expected Calibration Error (ECE):** `5.38%` *(Quantifies probability reliability)*

## Category Breakdown

| Category | Tested | Correct | Accuracy | Description |
|---|---|---|---|---|
| `contrast` | 35 | 28 | **80.0%** | Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome |
| `locative_par` | 20 | 20 | **100.0%** | Sentences using 'par' as locative postposition (on/at) rather than contrast |
| `mixed_emotions` | 30 | 18 | **60.0%** | Ambivalent emotions, bittersweet moments, dual primary/secondary states |
| `negation` | 30 | 22 | **73.3%** | Negation of negative states ('daant nahi padi') vs negated positive states |
| `pure_emotions` | 50 | 43 | **86.0%** | Canonical Hinglish & English emotional expressions |
| `sarcasm` | 35 | 28 | **80.0%** | Sarcastic praise, ironic frustration, over-enthusiastic cynicism |

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy** | 0.875 | 0.857 | 0.866 | 49 |
| **Sadness** | 0.723 | 0.723 | 0.723 | 47 |
| **Anger** | 0.878 | 0.782 | 0.827 | 55 |
| **Fear** | 0.700 | 0.875 | 0.778 | 8 |
| **Surprise** | 0.444 | 0.800 | 0.571 | 5 |
| **Love** | 1.000 | 0.286 | 0.444 | 7 |
| **Disgust** | 0.571 | 0.800 | 0.667 | 5 |
| **Neutral** | 0.821 | 0.958 | 0.885 | 24 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 1 | 28.0% | 0.0% | 28.0% |
| `[0.3, 0.4)` | 3 | 33.3% | 33.3% | 0.1% |
| `[0.4, 0.5)` | 19 | 44.3% | 52.6% | 8.3% |
| `[0.5, 0.6)` | 24 | 54.8% | 62.5% | 7.7% |
| `[0.6, 0.7)` | 27 | 66.0% | 81.5% | 15.5% |
| `[0.7, 0.8)` | 31 | 74.5% | 77.4% | 3.0% |
| `[0.8, 0.9)` | 44 | 85.5% | 84.1% | 1.4% |
| `[0.9, 1.0)` | 51 | 95.5% | 98.0% | 2.6% |
