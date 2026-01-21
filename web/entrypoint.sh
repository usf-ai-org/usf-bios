#!/bin/sh
set -e

# ============================================================================
# USF BIOS - Production Entrypoint
# Copyright (c) US Inc. All rights reserved.
# ============================================================================

# Security: Block CLI access
export USF_DISABLE_CLI=1
export USF_UI_ONLY=1

# Database configuration
export DATABASE_URL="${DATABASE_URL:-sqlite:////app/data/db/usf_bios.db}"

# ============================================================================
# API Proxy Architecture (No Runtime Config Needed)
# Browser → Next.js (port 3000) /api/* → Python Backend (localhost:8000)
# Frontend uses relative URLs, Next.js server proxies to backend internally
# No public URL detection needed - backend stays internal
# ============================================================================

# Create required directories (data + all cache directories for ML training)
mkdir -p /app/data/db /app/data/uploads /app/data/datasets \
         /app/data/output /app/data/outputs /app/data/checkpoints /app/data/logs \
         /app/data/logs/tensorboard /app/data/terminal_logs /app/data/encrypted_logs \
         /app/data/models /app/web/backend/data \
         /app/.cache/huggingface /app/.cache/huggingface/hub \
         /app/.cache/datasets /app/.cache/modelscope \
         /app/.cache/torch /app/.cache/torch/hub /app/.cache/torch_extensions \
         /app/.cache/triton /app/.triton /app/.triton/autotune /app/.triton/cache \
         /app/.cache/deepspeed /app/.deepspeed \
         /app/.cache/numba /app/.cache/xformers /app/.cache/matplotlib \
         /app/.cache/fontconfig /app/.cache/gradio /app/.cache/wandb \
         /app/.cache/ray /app/.cache/ccache /app/.cache/pip \
         /app/.cache/vllm /app/.cache/sglang /app/.cache/lmdeploy /app/.cache/cuda \
         /app/.local /app/.local/share /app/tmp 2>/dev/null || true

echo "=============================================="
echo "  USF BIOS - AI Fine-tuning Platform"
echo "  Copyright (c) US Inc. All rights reserved."
echo "=============================================="

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local name=$3
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "http://${host}:${port}/health" > /dev/null 2>&1; then
            echo "$name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    echo "ERROR: $name failed to start after ${max_attempts}s"
    return 1
}

# Start backend API (internal only - bound to 127.0.0.1)
echo "[1/2] Starting Backend API server..."
cd /app/web/backend
python -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --log-level warning \
    --no-access-log &
BACKEND_PID=$!

# Wait for backend health check
sleep 3
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend process died immediately!"
    echo "Checking logs..."
    exit 1
fi

# Start frontend
echo "[2/2] Starting Frontend server..."
cd /app/web/frontend
NODE_ENV=production node server.js &
FRONTEND_PID=$!

# Wait for services
sleep 2

echo ""
echo "=============================================="
echo "  Services Started Successfully!"
echo "=============================================="
echo ""
echo "  Access: http://0.0.0.0:3000"
echo ""
echo "=============================================="

# Graceful shutdown handler
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Shutdown complete."
    exit 0
}
trap cleanup TERM INT QUIT

# Keep container running - wait for any process to exit
wait -n $BACKEND_PID $FRONTEND_PID 2>/dev/null || wait $BACKEND_PID $FRONTEND_PID
