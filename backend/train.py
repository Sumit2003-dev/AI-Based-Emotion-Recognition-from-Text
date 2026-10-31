"""
Machine Learning Training & Model Selection Pipeline for AI Emotion Recognition.
Strictly adheres to ML best practices:
- Featurization ordering: split into train/test BEFORE fitting vectorizer/models.
- Stratified sampling to preserve balanced class proportions.
- Evaluates and compares multiple candidate classifiers (ComplementNB, LogisticRegression, Calibrated LinearSVC).
- Platt calibration for well-calibrated posterior probabilities.
- Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, Macro F1, Per-class F1, Confusion Matrix).
- Extracts top emotional keywords per class for interpretability/explainability.
- Persists best pipeline to models/emotion_model.joblib and metadata to models/model_metadata.json.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Ensure backend directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import get_emotion_dataframe


def preprocess_text(text: str) -> str:
    """Lightweight text normalization preserving punctuation cues and negations."""
    if not isinstance(text, str):
        return ""
    # Standardize whitespace and lower
    return " ".join(text.strip().split())


def train_and_evaluate_models() -> Dict[str, Any]:
    print("=" * 60)
    print("Starting AI Emotion Recognition Model Training Pipeline")
    print("=" * 60)

    # 1. Load and inspect dataset
    df = get_emotion_dataframe()
    print(f"Loaded {len(df)} samples across {df['emotion'].nunique()} classes.")

    # Save dataset copy to data directory
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "training_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Exported training dataset to {dataset_path}")

    # Check for missing values
    null_count = df["text"].isnull().sum() + df["emotion"].isnull().sum()
    if null_count > 0:
        print(f"Warning: Found {null_count} null rows. Dropping them.")
        df = df.dropna(subset=["text", "emotion"])

    X = np.array(df["text"].tolist(), dtype=object)
    y = np.array(df["emotion"].tolist(), dtype=object)
    classes = sorted(list(np.unique(y)))

    # 2. Strict featurization ordering: Train/Test split BEFORE vectorizer fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")

    # 3. Define candidate classifier pipelines with Platt calibration
    candidate_pipelines = {
        "Calibrated Linear SVC (Platt Scaling)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.5, dual=False, random_state=42), cv=3)),
        ]),
        "Logistic Regression (Balanced)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Complement Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", ComplementNB(norm=True)),
        ]),
    }

    # 4. Cross-validation comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_comparison = {}
    best_name = None
    best_cv_score = -1.0
    best_pipeline = None

    print("\n--- Model Cross-Validation Comparison (5-Fold Stratified Macro F1) ---")
    for name, pipeline in candidate_pipelines.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        model_comparison[name] = {
            "cv_macro_f1_mean": round(mean_score, 4),
            "cv_macro_f1_std": round(std_score, 4),
        }
        print(f"  * {name:<32}: Mean F1 = {mean_score:.4f} (±{std_score:.4f})")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_name = name
            best_pipeline = pipeline

    print(f"\nOptimal Architecture Selected: '{best_name}' (CV Macro F1 = {best_cv_score:.4f})")

    # 5. Train optimal model on the entire training set
    best_pipeline.fit(X_train, y_train)

    # 6. Comprehensive evaluation on held-out test split
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test) if hasattr(best_pipeline, "predict_proba") else None

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    test_macro_precision = float(precision_score(y_test, y_pred, average="macro"))
    test_macro_recall = float(recall_score(y_test, y_pred, average="macro"))

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\n" + "=" * 60)
    print("Test Set Performance Metrics:")
    print(f"  Accuracy        : {test_accuracy:.4f}")
    print(f"  Macro Precision : {test_macro_precision:.4f}")
    print(f"  Macro Recall    : {test_macro_recall:.4f}")
    print(f"  Macro F1-Score  : {test_macro_f1:.4f}")
    print(f"  Weighted F1     : {test_weighted_f1:.4f}")
    print("=" * 60)
    print("\nDetailed Per-Emotion Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    # 7. Extract top explanatory keywords for each emotion class
    # Fit a companion logistic regression on the full training data to extract interpretable weights
    vectorizer = best_pipeline.named_steps["tfidf"]
    vocab = np.array(vectorizer.get_feature_names_out())
    
    top_keywords_per_emotion = {}
    try:
        explainer_clf = LogisticRegression(C=2.5, max_iter=1000, random_state=42)
        X_train_vec = vectorizer.transform(X_train)
        explainer_clf.fit(X_train_vec, y_train)
        
        for idx, emotion_label in enumerate(explainer_clf.classes_):
            coefs = explainer_clf.coef_[idx]
            top_indices = np.argsort(coefs)[::-1][:15]
            top_keywords_per_emotion[emotion_label] = [
                {"word": vocab[i], "weight": round(float(coefs[i]), 4)}
                for i in top_indices
                if coefs[i] > 0.05
            ]
    except Exception as e:
        print(f"Notice: Explainer keyword extraction notice: {e}")
        top_keywords_per_emotion = {emotion: [] for emotion in classes}

    # 8. Compile metadata payload
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(),
        "total_dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "emotions": classes,
        "vocabulary_size": len(vocab),
        "metrics": {
            "accuracy": round(test_accuracy, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "macro_precision": round(test_macro_precision, 4),
            "macro_recall": round(test_macro_recall, 4),
        },
        "model_comparison": model_comparison,
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in classes if cls in report_dict
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist(),
        },
        "top_keywords": top_keywords_per_emotion,
    }

    # 9. Retrain selected architecture on 100% of the dataset for maximum production power
    print(f"\nFinal step: Retraining '{best_name}' on all {len(X)} samples for production deployment...")
    best_pipeline.fit(X, y)

    # 10. Persist model and metadata
    model_path = models_dir / "emotion_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved successfully to {model_path}")

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved successfully to {metadata_path}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
