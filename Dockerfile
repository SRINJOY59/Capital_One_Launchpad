# Use Python 3.11 as base image (more stable than 3.13 for ML packages)
FROM python:3.11-slim-bookworm

# Set workdir
WORKDIR /app

# Install system dependencies for OpenCV, YOLO, and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libx11-6 \
    curl \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xgboost

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/cache /app/models /app/logs && \
    chmod -R 755 /app

# Expose port
EXPOSE 8080

# Health check (make sure /health exists in your app)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1



# Run the app (no reload for prod) - use PORT env var for Google Cloud
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
