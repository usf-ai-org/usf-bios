#!/bin/sh

# Block any CLI attempts
export USF_DISABLE_CLI=1
export USF_UI_ONLY=1

# Set database path to writable directory
export DATABASE_URL="${DATABASE_URL:-sqlite:////app/data/db/usf_bios.db}"

# Create database directory if needed
mkdir -p /app/data/db 2>/dev/null || true

echo "=============================================="
echo "  USF BIOS - AI Fine-tuning Platform"
echo "  Copyright (c) US Inc. All rights reserved."
echo "=============================================="
echo ""
echo "  Mode: WebUI Only (CLI Disabled)"
echo ""

# Start backend API
echo "Starting backend API..."
cd /app/web/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend failed to start!"
    exit 1
fi
echo "Backend started successfully (PID: $BACKEND_PID)"

# Start frontend
cd /app/web/frontend
node server.js &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "  Services Ready"
echo "=============================================="
echo ""
echo "  Web UI:  http://localhost:3000"
echo "  API:     http://localhost:8000"
echo ""
echo "=============================================="

# Handle shutdown (POSIX compatible)
cleanup() {
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup TERM INT

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
