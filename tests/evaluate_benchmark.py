"""
Hinglish Benchmark Evaluation Suite
Evaluates Accuracy, Category Accuracy, Macro F1, Brier Score, and Expected Calibration Error (ECE)
over 200 curated Hinglish sentences across contrast, sarcasm, negation, locative_par, and mixed emotions.
"""

import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

# Path setup & deterministic offline benchmarking
os.environ["OFFLINE_MODE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_FILE = PROJECT_ROOT / "data" / "hinglish_eval_benchmark.json"
OUTPUT_REPORT = PROJECT_ROOT / "benchmark_results.md"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model import EmotionClassifier, EMOTION_META


def compute_metrics(eval_data, classifier):
    classes = list(EMOTION_META.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    total = len(eval_data)
    correct = 0

    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in classes}

    brier_sum = 0.0
    predictions = []

    for item in eval_data:
        text = item["text"]
        expected = item.get("expected") or item.get("expected_emotion")
        co_occ = item.get("co_occurring")
        category = item.get("category", "general")

        pred_res = classifier.predict(text)
        predicted = pred_res["dominant_emotion"]
        conf = pred_res["confidence"]
        prob_list = pred_res["probabilities"]
        dist = {p["emotion"]: p.get("ratio", p.get("probability", 0.0) / 100.0) for p in prob_list}

        is_correct = (predicted == expected)
        if is_correct:
            correct += 1

        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        if expected in class_stats:
            class_stats[expected]["support"] += 1
            if is_correct:
                class_stats[expected]["tp"] += 1
            else:
                class_stats[expected]["fn"] += 1
                if predicted in class_stats:
                    class_stats[predicted]["fp"] += 1

        # Brier score calculation: sum_k (p_k - y_k)^2
        sample_brier = 0.0
        for c in classes:
            p_k = dist.get(c, 0.0)
            y_k = 1.0 if c == expected else 0.0
            sample_brier += (p_k - y_k) ** 2
        brier_sum += sample_brier

        predictions.append({
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "confidence": conf,
            "correct": is_correct,
            "category": category,
            "co_occurring": co_occ
        })

    # Overall accuracy
    accuracy = correct / total if total > 0 else 0.0
    brier_score = brier_sum / total if total > 0 else 0.0

    # Macro & Weighted F1
    f1_list = []
    weighted_f1_sum = 0.0
    total_support = 0

    per_class_metrics = {}
    for c in classes:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        supp = class_stats[c]["support"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_metrics[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp
        }

        if supp > 0:
            f1_list.append(f1)
            weighted_f1_sum += f1 * supp
            total_support += supp

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0

    # Expected Calibration Error (ECE) with 10 bins
    num_bins = 10
    bins = [[] for _ in range(num_bins)]
    for p in predictions:
        conf = p["confidence"] / 100.0 if p["confidence"] > 1.0 else p["confidence"]
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append({**p, "conf_norm": conf})

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
            "avg_accuracy": bin_acc,
            "avg_confidence": bin_conf,
            "abs_error": abs(bin_acc - bin_conf)
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
        "predictions": predictions
    }


def main():
    if not DATA_FILE.exists():
        print(f"Error: Dataset {DATA_FILE} does not exist.")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Evaluating {len(eval_data)} benchmark samples against EmotionClassifier...")
    classifier = EmotionClassifier()
    metrics = compute_metrics(eval_data, classifier)

    print("\n" + "="*50)
    print("          BEYONDWORD BENCHMARK REPORT          ")
    print("="*50)
    print(f"Total Samples Tested : {metrics['total']}")
    print(f"Overall Accuracy     : {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"Macro F1 Score       : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score    : {metrics['weighted_f1']:.4f}")
    print(f"Brier Score          : {metrics['brier_score']:.4f} (lower is better, 0 = perfect)")
    print(f"Expected Calib Error : {metrics['ece']*100:.2f}% (ECE < 10% is well-calibrated)")
    print("="*50)

    print("\nCategory Accuracy Breakdown:")
    for cat, stats in sorted(metrics["category_stats"].items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f" - {cat:<18}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    failed = [p for p in metrics["predictions"] if not p["correct"]]
    print(f"\nSample Misclassifications ({len(failed)} total):")
    for f_item in failed[:10]:
        print(f" - [{f_item['category']}] '{f_item['text']}' -> Pred: {f_item['predicted']} ({f_item['confidence']}%), Expected: {f_item['expected']}")

    # Write Markdown Report
    md = []
    md.append("# BeyondWord 2.0 Hinglish & Complex Context Benchmark Report\n")
    md.append(f"**Date:** 2026-11-02  \n**Dataset:** `data/hinglish_eval_benchmark.json` ({metrics['total']} curated samples)  \n")
    md.append("## Executive Summary\n")
    md.append(f"- **Overall Accuracy:** `{metrics['accuracy']*100:.2f}%` ({metrics['correct']}/{metrics['total']})")
    md.append(f"- **Macro F1:** `{metrics['macro_f1']:.4f}`")
    md.append(f"- **Weighted F1:** `{metrics['weighted_f1']:.4f}`")
    md.append(f"- **Brier Score:** `{metrics['brier_score']:.4f}` *(Measures probability accuracy; lower is better)*")
    md.append(f"- **Expected Calibration Error (ECE):** `{metrics['ece']*100:.2f}%` *(Quantifies probability reliability)*\n")

    md.append("## Category Breakdown\n")
    md.append("| Category | Tested | Correct | Accuracy | Description |")
    md.append("|---|---|---|---|---|")
    cat_desc = {
        "contrast": "Clauses joined by 'par/lekin/but' resolving to second clause or positive outcome",
        "locative_par": "Sentences using 'par' as locative postposition (on/at) rather than contrast",
        "sarcasm": "Sarcastic praise, ironic frustration, over-enthusiastic cynicism",
        "negation": "Negation of negative states ('daant nahi padi') vs negated positive states",
        "mixed_emotions": "Ambivalent emotions, bittersweet moments, dual primary/secondary states",
        "pure_emotions": "Canonical Hinglish & English emotional expressions"
    }
    for cat, stats in sorted(metrics["category_stats"].items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        desc = cat_desc.get(cat, "")
        md.append(f"| `{cat}` | {stats['total']} | {stats['correct']} | **{acc:.1f}%** | {desc} |")

    md.append("\n## Emotion Class Performance\n")
    md.append("| Emotion | Precision | Recall | F1-Score | Support |")
    md.append("|---|---|---|---|---|")
    for emo, m in metrics["per_class_metrics"].items():
        if m["support"] > 0:
            md.append(f"| **{emo.capitalize()}** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")

    md.append("\n## Confidence Calibration (Reliability Diagram Data)\n")
    md.append("| Confidence Bin | Samples | Avg Confidence | Observed Accuracy | Calibration Gap |")
    md.append("|---|---|---|---|---|")
    for b in metrics["calibration_bins"]:
        md.append(f"| `{b['bin']}` | {b['samples']} | {b['avg_confidence']*100:.1f}% | {b['avg_accuracy']*100:.1f}% | {b['abs_error']*100:.1f}% |")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved benchmark markdown report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
