# USF BIOS - Complete Docker Image for RunPod GPU Server
# This image includes backend + frontend with auto-start
# Push to Docker Hub: docker build -t your-username/usf-bios:latest . && docker push your-username/usf-bios:latest

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL maintainer="US Inc"
LABEL description="USF BIOS AI Fine-tuning Platform"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Application directories
ENV APP_HOME=/app
ENV BACKEND_DIR=/app/backend
ENV FRONTEND_DIR=/app/frontend
ENV LOG_DIR=/var/log/usf-bios
ENV DATA_DIR=/app/data

# Create directories
RUN mkdir -p $APP_HOME $BACKEND_DIR $FRONTEND_DIR $LOG_DIR $DATA_DIR/uploads $DATA_DIR/datasets $DATA_DIR/output

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    wget \
    git \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy backend requirements and install
COPY web/backend/requirements.txt $BACKEND_DIR/requirements.txt
RUN pip3 install --no-cache-dir -r $BACKEND_DIR/requirements.txt

# Install additional ML dependencies that might not be in requirements
RUN pip3 install --no-cache-dir \
    torch>=2.1.0 \
    transformers>=4.36.0 \
    datasets>=2.15.0 \
    peft>=0.7.0 \
    trl>=0.7.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    sentencepiece>=0.1.99 \
    pynvml>=11.5.0 \
    psutil>=5.9.0

# Copy backend code
COPY web/backend $BACKEND_DIR

# Copy frontend and build
COPY web/frontend $FRONTEND_DIR
WORKDIR $FRONTEND_DIR
RUN npm ci --only=production 2>/dev/null || npm install
RUN npm run build

# Create startup script
RUN cat > /app/start.sh << 'STARTSCRIPT'
#!/bin/bash
set -e

LOG_DIR="/var/log/usf-bios"
mkdir -p $LOG_DIR

echo "=================================================="
echo "USF BIOS - Starting Services"
echo "=================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting USF BIOS" | tee $LOG_DIR/startup.log

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_DIR/startup.log
}

# Check GPU availability
log "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> $LOG_DIR/startup.log 2>&1 && log "GPU detected" || log "WARNING: nvidia-smi failed"
else
    log "WARNING: nvidia-smi not found - GPU may not be available"
fi

# Check Python packages
log "Checking BIOS installation..."
python3 << 'PYCHECK' 2>&1 | tee -a $LOG_DIR/startup.log
import sys
errors = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        errors.append("CUDA not available")
except ImportError as e:
    errors.append(f"PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"Transformers: {e}")

try:
    import peft
    print(f"✓ PEFT {peft.__version__}")
except ImportError as e:
    errors.append(f"PEFT: {e}")

try:
    import pynvml
    pynvml.nvmlInit()
    print("✓ pynvml initialized")
    pynvml.nvmlShutdown()
except Exception as e:
    print(f"⚠ pynvml: {e}")

if errors:
    print("\n❌ INSTALLATION ERRORS:")
    for err in errors:
        print(f"  - {err}")
    print("\nSystem may not function properly!")
else:
    print("\n✓ All packages installed correctly")
PYCHECK

# Start backend
log "Starting Backend server on port 8000..."
cd /app/backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > $LOG_DIR/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > /var/run/usf-backend.pid
log "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
log "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log "Backend is ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        log "ERROR: Backend failed to start within 30 seconds"
        cat $LOG_DIR/backend.log | tail -50 >> $LOG_DIR/startup.log
    fi
done

# Start frontend
log "Starting Frontend server on port 3000..."
cd /app/frontend
npm start > $LOG_DIR/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > /var/run/usf-frontend.pid
log "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
log "Waiting for frontend to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        log "Frontend is ready"
        break
    fi
    sleep 1
    if [ $i -eq 60 ]; then
        log "WARNING: Frontend may not be ready yet"
    fi
done

echo ""
echo "=================================================="
echo "USF BIOS - Services Started"
echo "=================================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo "Status:   http://localhost:8000/api/system/status"
echo ""
echo "Logs:"
echo "  Startup:  $LOG_DIR/startup.log"
echo "  Backend:  $LOG_DIR/backend.log"
echo "  Frontend: $LOG_DIR/frontend.log"
echo ""
echo "=================================================="

# Keep container running and show logs
log "Tailing logs (Ctrl+C to exit)..."
tail -f $LOG_DIR/backend.log $LOG_DIR/frontend.log
STARTSCRIPT

RUN chmod +x /app/start.sh

# Create health check script
RUN cat > /app/healthcheck.sh << 'HEALTHCHECK'
#!/bin/bash
curl -sf http://localhost:8000/health > /dev/null && \
curl -sf http://localhost:3000 > /dev/null
HEALTHCHECK
RUN chmod +x /app/healthcheck.sh

# Expose ports
EXPOSE 8000 3000

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command - start all services
CMD ["/app/start.sh"]
