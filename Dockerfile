# File: Dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "omniverse_ai.api.server:app", "--host", "0.0.0.0", "--port", "8000"]