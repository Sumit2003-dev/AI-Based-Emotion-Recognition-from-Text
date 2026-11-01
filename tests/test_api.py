"""
Unit and Integration Test Suite for AI Emotion Recognition.
Created & Maintained by Sumit Kumar (https://github.com/Sumit2003-dev)

Tests:
- Model initialization & serialization integrity
- Prediction accuracy on benchmark phrases
- Multiclass probability distributions & normalization
- Token-level explainability attribution
- Sentiment & emotional intensity calculation
- Negation handling
- FastAPI endpoints (/api/health, /api/predict, /api/batch, /api/model-info, /api/sample-texts)
"""

import sys
from pathlib import Path

# Ensure backend directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model import EMOTION_META
from app import (
    app,
    classifier,
    health_check,
    predict_emotion,
    batch_predict,
    get_model_info,
    get_sample_texts,
    submit_feedback,
    PredictRequest,
    BatchPredictRequest,
    FeedbackRequest,
)


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

    # Dislike negations like 'i dont like you' should be negative, never joy or love
    res_dislike = classifier.predict("i dont like you")
    assert res_dislike["dominant_emotion"] in ("anger", "sadness", "disgust"), f"Dislike negation should be negative, got {res_dislike['dominant_emotion']}"
    assert res_dislike["dominant_emotion"] not in ("joy", "love"), "Dislike negation 'i dont like you' should not predict joy or love"

    # Phrase-level negated positive traits like 'you are not that great', 'huh you are not that great'
    res_critique = classifier.predict("huh you are not that great")
    assert res_critique["dominant_emotion"] in ("anger", "disgust"), f"Critique should predict anger/disgust, got {res_critique['dominant_emotion']}"
    assert res_critique["dominant_emotion"] not in ("joy", "love"), "Critique 'huh you are not that great' should not predict joy or love"
    assert res_critique["sentiment"] == "Negative"


def test_sarcasm_and_irony_handling():
    # Sarcastic text with positive words + negative situation should predict anger/frustration, never joy
    sarcastic_text = "Oh fantastic, my flight was delayed for 5 hours, what a joy"
    res = classifier.predict(sarcastic_text)
    assert res["dominant_emotion"] in ("anger", "disgust"), f"Expected anger/disgust for sarcasm, got {res['dominant_emotion']}"
    assert res["dominant_emotion"] != "joy", "Sarcasm should not predict joy"
    assert res["sentiment"] == "Negative"

    # Litotes 'not bad' should not predict negative/disgust
    res_litotes = classifier.predict("The service was not bad at all.")
    assert res_litotes["dominant_emotion"] in ("neutral", "joy"), f"Expected neutral/joy for 'not bad', got {res_litotes['dominant_emotion']}"


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
    h_resp = health_check()
    assert h_resp["status"] == "healthy"
    assert "model_loaded" in h_resp

    # 2. Predict
    p_resp = predict_emotion(PredictRequest(text="I am thrilled and proud of my team!"))
    assert p_resp["dominant_emotion"] == "joy"

    # 3. Batch Predict
    b_resp = batch_predict(BatchPredictRequest(texts=["I am thrilled!", "I am so angry right now!"]))
    assert len(b_resp["results"]) == 2
    assert "summary" in b_resp

    # 4. Model Info
    m_resp = get_model_info()
    assert "model_name" in m_resp
    assert "emotions" in m_resp

    # 5. Sample Texts
    s_resp = get_sample_texts()
    assert len(s_resp["samples"]) >= 8

    # 6. Feedback
    fb_resp = submit_feedback(FeedbackRequest(text="Happy day", predicted_emotion="joy", is_correct=True))
    assert fb_resp["status"] == "success"


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
    test_sarcasm_and_irony_handling()
    print("test_sarcasm_and_irony_handling PASSED")
    test_empty_string_handling()
    print("test_empty_string_handling PASSED")
    test_batch_prediction()
    print("test_batch_prediction PASSED")
    test_fastapi_endpoints()
    print("test_fastapi_endpoints PASSED")
    print("ALL TESTS PASSED SUCCESSFULLY!")
