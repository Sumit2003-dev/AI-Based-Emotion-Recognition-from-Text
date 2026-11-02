"""
Unseen 500-Sample Generalization Evaluation Suite
Measures out-of-sample generalization accuracy, Macro F1, Brier Score, and Expected Calibration Error (ECE)
over 500 stratified, untouched real-world samples across all 8 emotion classes.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ["OFFLINE_MODE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_FILE = PROJECT_ROOT / "data" / "unseen_test_benchmark_500.json"
OUTPUT_REPORT = PROJECT_ROOT / "unseen_500_benchmark_results.md"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model import EmotionClassifier, EMOTION_META


def compute_metrics(eval_data, classifier):
    classes = list(EMOTION_META.keys())
    total = len(eval_data)
    correct = 0

    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in classes}

    brier_sum = 0.0
    predictions = []

    for item in eval_data:
        text = item["text"]
        expected = item["expected"]
        category = item.get("category", "general")

        pred_res = classifier.predict(text)
        predicted = pred_res["dominant_emotion"]
        conf = pred_res["confidence"]
        prob_list = pred_res["probabilities"]
        dist = {p["emotion"]: p.get("ratio", p.get("probability", 0.0) / 100.0) for p in prob_list}

        category_stats[category]["total"] += 1
        if expected in class_stats:
            class_stats[expected]["support"] += 1

        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
            category_stats[category]["correct"] += 1
            if predicted in class_stats:
                class_stats[predicted]["tp"] += 1
        else:
            if expected in class_stats:
                class_stats[expected]["fn"] += 1
            if predicted in class_stats:
                class_stats[predicted]["fp"] += 1

        sample_brier = 0.0
        for c in classes:
            p_k = dist.get(c, 0.0)
            y_k = 1.0 if c == expected else 0.0
            sample_brier += (p_k - y_k) ** 2
        brier_sum += sample_brier

        predictions.append({
            "text": text,
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "confidence": conf,
            "correct": is_correct,
            "distribution": dist,
        })

    accuracy = correct / total if total > 0 else 0.0
    brier_score = brier_sum / total if total > 0 else 0.0

    f1_list = []
    weighted_f1_sum = 0.0
    total_support = 0
    per_class_metrics = {}

    for c in classes:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        sup = class_stats[c]["support"]
        total_support += sup

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        f1_list.append(f1)
        weighted_f1_sum += f1 * sup
        per_class_metrics[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0

    num_bins = 10
    bins = [[] for _ in range(num_bins)]
    for p in predictions:
        conf_val = p["confidence"] / 100.0 if p["confidence"] > 1.0 else p["confidence"]
        bin_idx = min(int(conf_val * num_bins), num_bins - 1)
        bins[bin_idx].append({**p, "conf_norm": conf_val})

    ece = 0.0
    calibration_bins = []
    for idx, b in enumerate(bins):
        bin_size = len(b)
        if bin_size == 0:
            continue
        bin_acc = sum(1 for x in b if x["correct"]) / bin_size
        bin_conf = sum(x["conf_norm"] for x in b) / bin_size
        bin_weight = bin_size / total
        ece += bin_weight * abs(bin_acc - bin_conf)
        calibration_bins.append({
            "bin": f"[{idx*0.1:.1f}, {(idx+1)*0.1:.1f})",
            "samples": bin_size,
            "avg_conf": bin_conf,
            "acc": bin_acc,
            "gap": abs(bin_acc - bin_conf),
        })

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier_score,
        "ece": ece,
        "category_stats": dict(category_stats),
        "per_class_metrics": per_class_metrics,
        "calibration_bins": calibration_bins,
        "predictions": predictions,
    }


def main():
    print(f"Loading unseen 500-sample benchmark from: {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    classifier = EmotionClassifier()
    print("Evaluating 500 out-of-sample samples against EmotionClassifier...")
    metrics = compute_metrics(eval_data, classifier)

    print("\n" + "=" * 50)
    print("      UNSEEN 500 GENERALIZATION BENCHMARK REPORT      ")
    print("=" * 50)
    print(f"Total Samples Tested : {metrics['total']}")
    print(f"Overall Accuracy     : {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"Macro F1 Score       : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score    : {metrics['weighted_f1']:.4f}")
    print(f"Brier Score          : {metrics['brier_score']:.4f} (lower is better, 0 = perfect)")
    print(f"Expected Calib Error : {metrics['ece']*100:.2f}% (ECE < 10% is well-calibrated)")
    print("=" * 50)

    print("\nClass Performance Breakdown:")
    for c, stats in metrics["per_class_metrics"].items():
        print(f" - {c:<10}: F1={stats['f1']:.3f} | Prec={stats['precision']:.3f} | Rec={stats['recall']:.3f} | Support={stats['support']}")

    md = []
    md.append("# BeyondWord — Unseen 500-Sample Generalization Benchmark Report\n")
    md.append("Out-of-sample evaluation across 500 completely untouched, stratified samples from all 8 emotion classes.\n")
    md.append("## Executive Summary\n")
    md.append(f"- **Overall Accuracy:** `{metrics['accuracy']*100:.2f}%` ({metrics['correct']}/{metrics['total']})")
    md.append(f"- **Macro F1:** `{metrics['macro_f1']:.4f}`")
    md.append(f"- **Weighted F1:** `{metrics['weighted_f1']:.4f}`")
    md.append(f"- **Brier Score:** `{metrics['brier_score']:.4f}` *(Probability error; lower is better)*")
    md.append(f"- **Expected Calibration Error (ECE):** `{metrics['ece']*100:.2f}%` *(Quantifies probability reliability)*\n")

    md.append("## Emotion Class Performance\n")
    md.append("| Emotion | Precision | Recall | F1-Score | Support |")
    md.append("|---|---|---|---|---|")
    for c, stats in metrics["per_class_metrics"].items():
        md.append(f"| **{EMOTION_META[c]['label']}** | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} |")

    md.append("\n## Confidence Calibration (Reliability Diagram Data)\n")
    md.append("| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |")
    md.append("|---|---|---|---|---|")
    for b in metrics["calibration_bins"]:
        md.append(f"| `{b['bin']}` | {b['samples']} | {b['avg_conf']*100:.1f}% | {b['acc']*100:.1f}% | {b['gap']*100:.1f}% |")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

