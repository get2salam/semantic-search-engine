# ---- Semantic Search Engine ----
# Multi-stage build for a lean production image

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---------- dependencies ----------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM deps AS runtime

# Non-root user for security
RUN groupadd -r sse && useradd -r -g sse -d /app -s /sbin/nologin sse

COPY . .

# Pre-download the default model so first run is fast
ENV TRANSFORMERS_CACHE=/app/.cache \
    SSE_HOST=0.0.0.0 \
    SSE_PORT=8000 \
    SSE_LOG_LEVEL=INFO

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    chown -R sse:sse /app

USER sse

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "api:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
