# ==============================================================================
# Production Dockerfile for AI Emotion Recognition — FastAPI Backend
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements with CPU-optimized PyTorch to minimize image size
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application logic, calibrated models, and datasets
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/

WORKDIR /app/backend

EXPOSE 8000

# Health check monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start high-performance Uvicorn ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

