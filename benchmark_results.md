# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report

**Date:** 2026-11-02  
**Dataset:** `data/hinglish_eval_benchmark.json` (200 curated samples)  

## Executive Summary

- **Overall Accuracy:** `56.50%` (113/200)
- **Macro F1:** `0.5456`
- **Weighted F1:** `0.5626`
- **Brier Score:** `0.5973` *(Measures probability accuracy; lower is better)*
- **Expected Calibration Error (ECE):** `6.99%` *(Quantifies probability reliability)*

## Category Breakdown

| Category | Tested | Correct | Accuracy | Description |
|---|---|---|---|---|
| `contrast` | 35 | 17 | **48.6%** | Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome |
| `locative_par` | 20 | 15 | **75.0%** | Sentences using 'par' as locative postposition (on/at) rather than contrast |
| `mixed_emotions` | 30 | 15 | **50.0%** | Ambivalent emotions, bittersweet moments, dual primary/secondary states |
| `negation` | 30 | 14 | **46.7%** | Negation of negative states ('daant nahi padi') vs negated positive states |
| `pure_emotions` | 50 | 40 | **80.0%** | Canonical Hinglish & English emotional expressions |
| `sarcasm` | 35 | 12 | **34.3%** | Sarcastic praise, ironic frustration, over-enthusiastic cynicism |

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy** | 0.600 | 0.551 | 0.574 | 49 |
| **Sadness** | 0.580 | 0.617 | 0.598 | 47 |
| **Anger** | 0.759 | 0.400 | 0.524 | 55 |
| **Fear** | 0.467 | 0.875 | 0.609 | 8 |
| **Surprise** | 0.267 | 0.800 | 0.400 | 5 |
| **Love** | 1.000 | 0.286 | 0.444 | 7 |
| **Disgust** | 0.500 | 0.800 | 0.615 | 5 |
| **Neutral** | 0.500 | 0.750 | 0.600 | 24 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 3 | 25.7% | 0.0% | 25.7% |
| `[0.3, 0.4)` | 24 | 36.2% | 29.2% | 7.1% |
| `[0.4, 0.5)` | 36 | 45.6% | 36.1% | 9.5% |
| `[0.5, 0.6)` | 34 | 54.6% | 58.8% | 4.2% |
| `[0.6, 0.7)` | 31 | 65.2% | 58.1% | 7.1% |
| `[0.7, 0.8)` | 28 | 74.3% | 67.9% | 6.5% |
| `[0.8, 0.9)` | 29 | 84.8% | 79.3% | 5.5% |
| `[0.9, 1.0)` | 15 | 93.7% | 86.7% | 7.0% |
