"""
FastAPI Backend Application for AI-Based Emotion Recognition.
Created by: Sumit Kumar (https://github.com/Sumit2003-dev)

Provides:
- Real-time text emotion analysis with probabilities and explainability
- Batch CSV and text array processing
- Model metadata, metrics, and confusion matrix API
- Static file serving for the interactive Web UI
"""

import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure backend directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import EmotionClassifier, EMOTION_META

# Initialize classifier
classifier = EmotionClassifier()

app = FastAPI(
    title="AI-Based Emotion Recognition API",
    description="High-performance Multi-Class Emotion Recognition Intelligence Suite created by Sumit Kumar.",
    version="2.1.0",
    contact={
        "name": "Sumit Kumar",
        "url": "https://github.com/Sumit2003-dev/AI-Based-Emotion-Recognition-from-Text",
    },
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500, description="List of texts to analyze")


class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Long-form text to segment and analyze")

class FeedbackRequest(BaseModel):
    text: str
    predicted_emotion: str
    corrected_emotion: Optional[str] = None
    is_correct: bool = True
    user_comment: Optional[str] = None


# Global in-memory feedback log
feedback_store = []


# API Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier.is_ready(),
        "model_name": classifier.metadata.get("model_name", "EmotionClassifier"),
    }


@app.post("/api/predict")
def predict_emotion(request: PredictRequest):
    """Predicts dominant emotion, probability distribution, sentiment, intensity, and token highlights."""
    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Emotion classifier model is not ready.")
    result = classifier.predict(request.text)
    return result


@app.post("/api/batch")
def batch_predict(request: BatchPredictRequest):
    """Processes a batch array of textual inputs."""
    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Emotion classifier model is not ready.")
    batch_result = classifier.predict_batch(request.texts)
    return batch_result


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Accepts a CSV file, detects the text column, and returns batch predictions."""
    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Emotion classifier model is not ready.")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")
    
    contents = await file.read()
    try:
        decoded = contents.decode("utf-8")
    except UnicodeDecodeError:
        decoded = contents.decode("latin-1")

    reader = csv.reader(io.StringIO(decoded))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    # Find candidate text column
    header = [h.strip().lower() for h in rows[0]]
    text_col_idx = 0
    candidate_names = ["text", "sentence", "content", "message", "tweet", "review", "comment", "utterance"]
    for idx, col in enumerate(header):
        if col in candidate_names:
            text_col_idx = idx
            break

    # Extract sentences (skip header if detected)
    start_row = 1 if any(c in header for c in candidate_names) else 0
    extracted_texts = []
    for r in rows[start_row:]:
        if len(r) > text_col_idx and r[text_col_idx].strip():
            extracted_texts.append(r[text_col_idx].strip())

    if not extracted_texts:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded CSV.")

    batch_result = classifier.predict_batch(extracted_texts[:300]) # Cap at 300 for responsiveness
    return batch_result


@app.get("/api/model-info")
def get_model_info():
    """Returns model architecture, accuracy, macro F1, confusion matrix, and training details."""
    metadata = dict(classifier.metadata) if classifier.metadata else {}
    metadata["active_engine"] = classifier.active_engine
    if not metadata.get("model_name"):
        metadata["model_name"] = classifier.active_engine
        metadata["emotions"] = list(EMOTION_META.keys())
    return metadata


@app.get("/api/sample-texts")
def get_sample_texts():
    """Returns curated presets for quick 1-click testing across all emotion categories."""
    return {
        "samples": [
            {
                "category": "Joy",
                "emoji": "😊",
                "text": "We just closed our series A funding round and I couldn't be more thrilled and proud of our entire team!",
            },
            {
                "category": "Sadness",
                "emoji": "😢",
                "text": "Saying goodbye to my childhood home today broke my heart, I feel so empty and alone inside.",
            },
            {
                "category": "Anger",
                "emoji": "😡",
                "text": "This is completely unacceptable! You charged my card three times and lied straight to my face!",
            },
            {
                "category": "Fear",
                "emoji": "😨",
                "text": "My hands are trembling and my chest feels tight, I am terrified of what the surgery results might be.",
            },
            {
                "category": "Surprise",
                "emoji": "😲",
                "text": "Wait, are you serious?! I opened the envelope and was stunned to find out I won the grand prize!",
            },
            {
                "category": "Love",
                "emoji": "❤️",
                "text": "I love you with all my heart and soul, thank you for being my rock and bringing so much warmth to my life.",
            },
            {
                "category": "Disgust",
                "emoji": "🤢",
                "text": "The public bathroom was covered in filthy black mold and reeked of rotting sewage, absolutely revolting!",
            },
            {
                "category": "Neutral",
                "emoji": "😐",
                "text": "The quarterly financial balance sheet will be presented at the scheduled 2:00 PM boardroom meeting.",
            },
        ]
    }


@app.post("/api/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """Allows users to submit corrections or positive feedback for continuous learning with persistence."""
    data = feedback.model_dump()
    data["id"] = f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(feedback_store) + 1}"
    data["timestamp"] = datetime.now().isoformat()
    data["status"] = "pending_human_review"
    feedback_store.append(data)

    # Persist to disk for human-in-the-loop active learning
    feedback_file = PROJECT_ROOT / "data" / "feedback_store.jsonl"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return {
        "status": "success",
        "feedback_id": data["id"],
        "message": "Thank you for helping improve the emotion recognition model!",
        "total_feedbacks": len(feedback_store),
    }

@app.post("/api/analyze-document")
def analyze_document(request: DocumentRequest):
    """Performs sliding-window chunking and generates an emotion timeline for long-form texts."""
    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Emotion classification engine is not initialized.")
    return classifier.analyze_long_text(request.text)


@app.get("/api/feedback/pending")
def get_pending_feedback():
    """Returns active feedback items queued for human review."""
    feedback_file = PROJECT_ROOT / "data" / "feedback_store.jsonl"
    items = []
    if feedback_file.exists():
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line.strip()))
                    except:
                        pass
    return {"total": len(items), "feedback_queue": items[-50:]}



@app.get("/")
def serve_api_root():
    """Provides API gateway root metadata and available endpoints."""
    return {
        "name": "AI-Based Emotion Recognition API",
        "status": "online",
        "version": "2.2.0",
        "documentation": "/docs",
        "health_check": "/api/health",
        "frontend_dev_url": "http://localhost:3000",
        "endpoints": [
            "/api/health",
            "/api/predict",
            "/api/batch",
            "/api/upload-csv",
            "/api/model-info",
            "/api/sample-texts",
            "/api/feedback",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
