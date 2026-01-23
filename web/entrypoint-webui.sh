#!/bin/sh
set -e

# ============================================================================
# USF BIOS Web UI - Entrypoint
# Copyright (c) US Inc. All rights reserved.
# ============================================================================

# Database configuration
export DATABASE_URL="${DATABASE_URL:-sqlite:////app/data/db/usf_bios.db}"

# Create required directories
mkdir -p /app/data/db /app/data/uploads /app/data/datasets \
         /app/data/output /app/data/checkpoints /app/data/logs \
         /app/data/terminal_logs /app/data/models /app/backend/data 2>/dev/null || true

echo "=============================================="
echo "  USF BIOS Web UI"
echo "  Copyright (c) US Inc. All rights reserved."
echo "=============================================="

# Start backend API (internal only - bound to 127.0.0.1)
echo "[1/2] Starting Backend API server..."
cd /app/backend
python -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --log-level warning &
BACKEND_PID=$!

# Wait for backend health check
echo "Waiting for backend to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ Backend is ready"
        break
    fi
    sleep 1
done

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend process died!"
    exit 1
fi

# Start frontend
echo "[2/2] Starting Frontend server..."
cd /app/frontend
NODE_ENV=production node server.js &
FRONTEND_PID=$!

# Wait for services
sleep 3

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

# Keep container running
wait -n $BACKEND_PID $FRONTEND_PID 2>/dev/null || wait $BACKEND_PID $FRONTEND_PID
