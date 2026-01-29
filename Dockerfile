# Inference container for COVID MLOps Lab
FROM python:3.11-slim

# Prevent Python from writing pyc + ensure logs flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.lock.txt

# Copy source code
COPY src /app/src
COPY configs /app/configs

# Expose API port
EXPOSE 8000

# By default, run the API.
# Model artifacts are expected at /app/artifacts via volume mount.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

