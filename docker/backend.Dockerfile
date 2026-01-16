# USF BIOS Backend - GPU-enabled Docker Image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY web/backend/requirements.txt /app/requirements.txt

# Create requirements.txt if it doesn't exist
RUN if [ ! -s /app/requirements.txt ]; then \
    echo "fastapi>=0.104.0" > /app/requirements.txt && \
    echo "uvicorn[standard]>=0.24.0" >> /app/requirements.txt && \
    echo "pydantic>=2.5.0" >> /app/requirements.txt && \
    echo "python-multipart>=0.0.6" >> /app/requirements.txt && \
    echo "aiofiles>=23.2.1" >> /app/requirements.txt && \
    echo "websockets>=12.0" >> /app/requirements.txt && \
    echo "pynvml>=11.5.0" >> /app/requirements.txt && \
    echo "psutil>=5.9.0" >> /app/requirements.txt && \
    echo "torch>=2.1.0" >> /app/requirements.txt && \
    echo "transformers>=4.36.0" >> /app/requirements.txt && \
    echo "datasets>=2.15.0" >> /app/requirements.txt && \
    echo "peft>=0.7.0" >> /app/requirements.txt && \
    echo "trl>=0.7.0" >> /app/requirements.txt && \
    echo "accelerate>=0.25.0" >> /app/requirements.txt && \
    echo "bitsandbytes>=0.41.0" >> /app/requirements.txt && \
    echo "sentencepiece>=0.1.99" >> /app/requirements.txt; \
    fi

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY web/backend /app

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/datasets /app/output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
