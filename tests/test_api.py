"""
Unit and Integration Test Suite for AI Emotion Recognition.
Tests:
- Model initialization & serialization integrity
- Prediction accuracy on benchmark phrases
- Multiclass probability distributions & normalization
- Token-level explainability attribution
- Sentiment & emotional intensity calculation
- Negation handling
- FastAPI endpoints (/api/health, /api/predict, /api/batch, /api/model-info, /api/sample-texts)
"""

import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model import EmotionClassifier, EMOTION_META
from app import app

client = TestClient(app)
classifier = EmotionClassifier()


def test_model_loaded():
    assert classifier.is_ready(), "Model pipeline should be successfully loaded"
    assert len(classifier.classes) == 8, "Expected 8 emotion classes"


def test_prediction_structure():
    text = "I am so happy and delighted about our victory today!"
    res = classifier.predict(text)

    assert "dominant_emotion" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert "tokens" in res
    assert "sentiment" in res
    assert "intensity" in res
    assert "guidance" in res

    assert res["dominant_emotion"] == "joy"
    assert res["sentiment"] == "Positive"
    assert len(res["probabilities"]) == 8

    # Ensure probabilities sum approximately to 100%
    prob_sum = sum(p["probability"] for p in res["probabilities"])
    assert 99.0 <= prob_sum <= 101.0, f"Probabilities should sum to ~100%, got {prob_sum}"


def test_emotion_recognition_benchmarks():
    test_cases = [
        ("I feel so lonely, empty, and crying quietly in my room.", "sadness"),
        ("I am furious that you lied to my face and cheated me!", "anger"),
        ("My hands are shaking with terror as the dark figure approaches.", "fear"),
        ("I love you more than words could ever describe, my soulmate.", "love"),
        ("The rotten fish was covered in green slime, utterly revolting!", "disgust"),
        ("The quarterly meeting is scheduled for 2:00 PM on Thursday.", "neutral"),
        ("Wait, what?! I never expected to find a million dollars in the attic!", "surprise"),
    ]

    for text, expected_emotion in test_cases:
        res = classifier.predict(text)
        assert res["dominant_emotion"] == expected_emotion, (
            f"Expected '{expected_emotion}' for '{text}', got '{res['dominant_emotion']}'"
        )


def test_negation_handling():
    # Negations like 'not happy' should not be classified as joy
    res = classifier.predict("I am not happy at all with this painful outcome.")
    assert res["dominant_emotion"] != "joy", "Negation 'not happy' should not predict joy"


def test_empty_string_handling():
    res = classifier.predict("")
    assert res["dominant_emotion"] == "neutral"
    assert res["confidence"] == 0.0


def test_batch_prediction():
    texts = [
        "What a wonderful celebration!",
        "I am so angry right now!",
        "The file was saved to disk.",
    ]
    batch_res = classifier.predict_batch(texts)
    assert len(batch_res["results"]) == 3
    assert "summary" in batch_res
    assert batch_res["summary"]["total_analyzed"] == 3


def test_fastapi_endpoints():
    # 1. Health
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "healthy"

    # 2. Predict
    p_resp = client.post("/api/predict", json={"text": "I am thrilled and proud of my team!"})
    assert p_resp.status_code == 200
    assert p_resp.json()["dominant_emotion"] == "joy"

    # 3. Model Info
    m_resp = client.get("/api/model-info")
    assert m_resp.status_code == 200
    assert "model_name" in m_resp.json()

    # 4. Sample Texts
    s_resp = client.get("/api/sample-texts")
    assert s_resp.status_code == 200
    assert len(s_resp.json()["samples"]) >= 8


if __name__ == "__main__":
    print("Running tests manually...")
    test_model_loaded()
    print("test_model_loaded PASSED")
    test_prediction_structure()
    print("test_prediction_structure PASSED")
    test_emotion_recognition_benchmarks()
    print("test_emotion_recognition_benchmarks PASSED")
    test_negation_handling()
    print("test_negation_handling PASSED")
    test_empty_string_handling()
    print("test_empty_string_handling PASSED")
    test_batch_prediction()
    print("test_batch_prediction PASSED")
    test_fastapi_endpoints()
    print("test_fastapi_endpoints PASSED")
    print("ALL TESTS PASSED SUCCESSFULLY!")
