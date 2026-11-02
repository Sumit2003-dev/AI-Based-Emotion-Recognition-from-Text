# BeyondWord — Unseen 500-Sample Generalization Benchmark Report

Out-of-sample evaluation across 500 completely untouched, stratified samples from all 8 emotion classes.

## Executive Summary

- **Overall Accuracy:** `66.60%` (333/500)
- **Macro F1:** `0.6679`
- **Weighted F1:** `0.6707`
- **Brier Score:** `0.4013` *(Probability error; lower is better)*
- **Expected Calibration Error (ECE):** `11.17%` *(Quantifies probability reliability)*

## Emotion Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Joy & Elation** | 0.881 | 0.800 | 0.839 | 65 |
| **Sadness & Grief** | 0.696 | 0.600 | 0.645 | 65 |
| **Anger & Frustration** | 0.692 | 0.554 | 0.615 | 65 |
| **Fear & Anxiety** | 0.871 | 0.831 | 0.850 | 65 |
| **Surprise & Astonishment** | 0.500 | 0.367 | 0.423 | 60 |
| **Love & Affection** | 1.000 | 0.650 | 0.788 | 60 |
| **Disgust & Aversion** | 0.559 | 0.867 | 0.680 | 60 |
| **Neutral & Objective** | 0.411 | 0.650 | 0.503 | 60 |

## Confidence Calibration (Reliability Diagram Data)

| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|
| `[0.2, 0.3)` | 6 | 29.0% | 50.0% | 21.0% |
| `[0.3, 0.4)` | 35 | 35.8% | 45.7% | 9.9% |
| `[0.4, 0.5)` | 65 | 45.0% | 52.3% | 7.3% |
| `[0.5, 0.6)` | 73 | 55.2% | 46.6% | 8.7% |
| `[0.6, 0.7)` | 73 | 64.4% | 42.5% | 21.9% |
| `[0.7, 0.8)` | 58 | 74.6% | 51.7% | 22.9% |
| `[0.8, 0.9)` | 62 | 85.3% | 91.9% | 6.6% |
| `[0.9, 1.0)` | 128 | 94.8% | 100.0% | 5.2% |
