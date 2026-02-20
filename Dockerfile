# AgriFlow Backend — Hugging Face Spaces Docker deployment
# 16 GB RAM free tier handles XGBoost + SHAP comfortably
#
# HF Spaces requires port 7860.
# All API keys are injected as Repository Secrets (never in this file).

FROM python:3.11-slim

WORKDIR /app

# System deps needed to compile numpy/scipy/scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (separate layer → cached unless requirements change)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy project source (db, src, archia prompts, etc.)
COPY . .

# HF Spaces injects PORT=7860; fallback to 7860 for local Docker testing
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# Run the FastAPI backend
CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1
