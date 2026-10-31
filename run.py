"""
AI-Based Emotion Recognition from Text - Launcher.
Created by: Sumit Kumar (https://github.com/Sumit2003-dev)

Validates environment, ensures model is trained and calibrated, and starts the FastAPI server.
"""

import os
import sys
import webbrowser
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "emotion_model.joblib"

# Add backend to path
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def check_and_prepare_model():
    """Verifies that the trained model pipeline exists; if not, triggers training automatically."""
    if not MODEL_PATH.exists():
        print("\n[!] Trained emotion model not found. Automatically triggering training pipeline...")
        from backend.train import train_and_evaluate_models
        train_and_evaluate_models()
        print("[✓] Model training completed successfully!\n")
    else:
        print(f"[✓] Verified existing trained model: {MODEL_PATH}")


def main():
    print("=" * 65)
    print("🧠 AI-Based Emotion Recognition from Text")
    print("Created by Sumit Kumar (https://github.com/Sumit2003-dev)")
    print("=" * 65)

    check_and_prepare_model()

    import uvicorn
    from backend.app import app

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print(f"\n🚀 Starting High-Performance NLP Emotion Server at:")
    print(f"   👉 Web Application UI : {url}")
    print(f"   👉 Interactive Swagger : {url}/docs")
    print(f"   👉 Health Endpoint    : {url}/api/health\n")
    print("Press CTRL+C anytime to stop the server.\n")

    # Launch server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
