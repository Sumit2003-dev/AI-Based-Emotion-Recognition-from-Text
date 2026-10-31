"""
FastAPI Backend Application for AI-Based Emotion Recognition.
Provides:
- Real-time text emotion analysis with probabilities and explainability
- Batch CSV and text array processing
- Model metadata, metrics, and confusion matrix API
- Static file serving for the interactive Web UI
"""

import csv
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
    title="AI Emotion Recognition API",
    description="Multiclass emotion recognition from text using calibrated NLP models, explainability, and empathy tips.",
    version="1.0.0",
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
    """Allows users to submit corrections or positive feedback for continuous learning."""
    feedback_store.append(feedback.model_dump())
    return {
        "status": "success",
        "message": "Thank you for helping improve the emotion recognition model!",
        "total_feedbacks": len(feedback_store),
    }


# Mount frontend static directory or provide API gateway overview
frontend_dist = PROJECT_ROOT / "frontend" / "dist" / "client"
frontend_legacy = PROJECT_ROOT / "frontend"

if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(frontend_dist / "index.html")
elif (frontend_legacy / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_legacy)), name="static")

    @app.get("/")
    def serve_legacy_index():
        return FileResponse(frontend_legacy / "index.html")
else:
    @app.get("/")
    def serve_api_root():
        return {
            "name": "AI-Based Emotion Recognition API",
            "status": "online",
            "version": "2.0.0",
            "documentation": "/docs",
            "health_check": "/api/health",
            "frontend_dev_url": "http://localhost:3000",
            "endpoints": [
                "/api/health",
                "/api/predict",
                "/api/batch",
                "/api/model-info",
                "/api/sample-texts",
                "/api/feedback"
            ]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
