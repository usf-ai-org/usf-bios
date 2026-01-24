# USF BIOS - Production Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Frontend Builder
# ============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY frontend/ ./

# Build the frontend
RUN npm run build

# ============================================================================
# Stage 2: Production Image
# ============================================================================
FROM python:3.11-slim

LABEL maintainer="US Inc <support@us.inc>"
LABEL description="USF BIOS - Enterprise AI Fine-tuning Platform"
LABEL version="1.0.10"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USF_UI_ONLY=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy USF BIOS core
COPY requirements/ /app/requirements/
COPY usf_bios/ /app/usf_bios/
COPY setup.py /app/
COPY setup.cfg /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart websockets

# Copy backend
COPY web/backend/ /app/web/backend/

# Copy RSA public key for log encryption (private key stays with US Inc)
COPY keys/usf_bios_public.pem /app/keys/usf_bios_public.pem

# Install cryptography for RSA encryption
RUN pip install --no-cache-dir cryptography

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/.next/standalone /app/web/frontend/
COPY --from=frontend-builder /app/frontend/.next/static /app/web/frontend/.next/static
COPY --from=frontend-builder /app/frontend/public /app/web/frontend/public

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "  USF BIOS - AI Fine-tuning Platform"
echo "  Powered by US Inc"
echo "=============================================="
echo ""
echo "Starting services..."

# Start backend API
cd /app/web/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 3

# Start frontend
cd /app/web/frontend
node server.js &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "  Services Started Successfully!"
echo "=============================================="
echo ""
echo "  Frontend UI:  http://localhost:3000"
echo "  Backend API:  http://localhost:8000"
echo ""
echo "=============================================="

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
EOF
chmod +x /app/start.sh

# Expose ports
EXPOSE 3000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/start.sh"]
