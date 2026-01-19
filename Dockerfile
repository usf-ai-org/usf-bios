# USF BIOS - Secured Production Docker Image
# Copyright (c) US Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL maintainer="US Inc"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=0

# Application directories
ENV APP_HOME=/app
ENV BACKEND_DIR=/app/backend
ENV FRONTEND_DIR=/app/frontend
ENV CORE_DIR=/app/core
ENV LOG_DIR=/var/log/usf-bios
ENV DATA_DIR=/app/data

# ============================================
# HARDCODED CAPABILITY RESTRICTIONS
# These CANNOT be changed by users at runtime
# ============================================
# Only allow HuggingFace and Local models (NO ModelScope)
ENV SUPPORTED_MODEL_SOURCES=huggingface,local
# Only allow text-to-text models
ENV SUPPORTED_MODALITIES=text2text
# Only allow USF Omega architecture
ENV SUPPORTED_ARCHITECTURES=UsfOmegaForCausalLM
# Lock capabilities - prevents env override
ENV USF_CAPABILITIES_LOCKED=true

# Create directories
RUN mkdir -p $APP_HOME $BACKEND_DIR $FRONTEND_DIR $CORE_DIR $LOG_DIR $DATA_DIR/uploads $DATA_DIR/datasets $DATA_DIR/output

# Install system dependencies including build tools for Cython
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install pip, security tools, and Cython for C compilation
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel cython nuitka

# Copy and install requirements
COPY web/backend/requirements.txt $BACKEND_DIR/requirements.txt
COPY requirements.txt /tmp/core_requirements.txt
RUN pip3 install --no-cache-dir -r $BACKEND_DIR/requirements.txt || true
RUN pip3 install --no-cache-dir -r /tmp/core_requirements.txt || true

# Install ML dependencies
RUN pip3 install --no-cache-dir \
    torch>=2.1.0 \
    git+https://github.com/apt-team-018/transformers.git \
    datasets>=2.15.0 \
    peft>=0.7.0 \
    trl>=0.7.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    sentencepiece>=0.1.99 \
    pynvml>=11.5.0 \
    psutil>=5.9.0 \
    cryptography>=41.0.0

# ============================================
# COPY ALL SOURCE CODE
# ============================================

# Copy core training library (MAIN IP)
# Copy setup files first, then source code
COPY setup.py setup.cfg MANIFEST.in $CORE_DIR/
COPY requirements.txt $CORE_DIR/requirements.txt
COPY requirements/ $CORE_DIR/requirements/
COPY usf_bios $CORE_DIR/usf_bios

# Copy backend API
COPY web/backend $BACKEND_DIR

# Copy frontend
COPY web/frontend $FRONTEND_DIR

# ============================================
# MAXIMUM IP PROTECTION
# ============================================

# Create compilation script for Cython
RUN cat > /tmp/compile_to_c.py << 'COMPILE_SCRIPT'
import os
import sys
import py_compile
import compileall
import shutil

def compile_directory(src_dir):
    """Compile all Python files to bytecode and remove source"""
    for root, dirs, files in os.walk(src_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                try:
                    # Compile to optimized bytecode
                    py_compile.compile(filepath, cfile=filepath + 'c', optimize=2)
                    
                    # Remove source file (keep only .pyc)
                    if filename != '__init__.py':
                        os.remove(filepath)
                    else:
                        # Empty __init__.py files
                        with open(filepath, 'w') as f:
                            f.write('')
                except Exception as e:
                    print(f"Error compiling {filepath}: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for directory in sys.argv[1:]:
            if os.path.exists(directory):
                print(f"Compiling {directory}...")
                compile_directory(directory)
                print(f"Done: {directory}")
COMPILE_SCRIPT

# Compile ALL Python code to bytecode
RUN python3 /tmp/compile_to_c.py $CORE_DIR $BACKEND_DIR

# Remove ALL .py source files (keep only compiled .pyc)
RUN find $CORE_DIR -name "*.py" -type f ! -name "__init__.py" -delete 2>/dev/null || true
RUN find $BACKEND_DIR -name "*.py" -type f ! -name "__init__.py" -delete 2>/dev/null || true

# Empty all __init__.py files
RUN find $CORE_DIR -name "__init__.py" -exec sh -c 'echo "" > "$1"' _ {} \; 2>/dev/null || true
RUN find $BACKEND_DIR -name "__init__.py" -exec sh -c 'echo "" > "$1"' _ {} \; 2>/dev/null || true

# Remove __pycache__ directories (we have .pyc files in place)
RUN find $CORE_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
RUN find $BACKEND_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ============================================
# COMPREHENSIVE CLEANUP - DELETE ALL DOCS & COMMENTS
# ============================================

# Delete ALL documentation formats from CORE
RUN find $CORE_DIR -type f \( \
    -name "*.md" -o -name "*.txt" -o -name "*.rst" -o \
    -name "*.doc" -o -name "*.docx" -o -name "*.pdf" -o \
    -name "*.html" -o -name "*.htm" -o -name "*.xml" -o \
    -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o \
    -name "*.ini" -o -name "*.cfg" -o -name "*.conf" -o \
    -name "README*" -o -name "CHANGELOG*" -o -name "LICENSE*" -o \
    -name "CONTRIBUTING*" -o -name "AUTHORS*" -o -name "HISTORY*" -o \
    -name "TODO*" -o -name "NOTICE*" -o -name "*.sample" \
    \) -delete 2>/dev/null || true

# Delete ALL documentation formats from BACKEND
RUN find $BACKEND_DIR -type f \( \
    -name "*.md" -o -name "*.txt" -o -name "*.rst" -o \
    -name "*.doc" -o -name "*.docx" -o -name "*.pdf" -o \
    -name "*.html" -o -name "*.htm" -o -name "*.xml" -o \
    -name "README*" -o -name "CHANGELOG*" -o -name "LICENSE*" -o \
    -name "CONTRIBUTING*" -o -name "AUTHORS*" \
    \) -delete 2>/dev/null || true

# Delete ALL test files and directories
RUN find $CORE_DIR -type f \( -name "test_*.py*" -o -name "*_test.py*" -o -name "tests.py*" \) -delete 2>/dev/null || true
RUN find $BACKEND_DIR -type f \( -name "test_*.py*" -o -name "*_test.py*" -o -name "tests.py*" \) -delete 2>/dev/null || true
RUN rm -rf $CORE_DIR/tests $CORE_DIR/test $BACKEND_DIR/tests $BACKEND_DIR/test 2>/dev/null || true

# Delete example and sample files
RUN find $CORE_DIR -type f \( -name "example*.py*" -o -name "*example.py*" -o -name "sample*.py*" \) -delete 2>/dev/null || true
RUN find $CORE_DIR -type d \( -name "examples" -o -name "samples" -o -name "docs" -o -name "doc" \) -exec rm -rf {} + 2>/dev/null || true

# Strip comments and docstrings from Python bytecode (compile with optimization level 2)
# Level 2 removes docstrings and asserts
RUN python3 -c "
import py_compile
import os
import sys

def strip_and_compile(directory):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.pyc'):
                filepath = os.path.join(root, f)
                try:
                    # Recompile with maximum optimization (strips docstrings)
                    py_compile.compile(filepath.replace('.pyc', '.py') if os.path.exists(filepath.replace('.pyc', '.py')) else filepath, 
                                      cfile=filepath, optimize=2, doraise=False)
                except:
                    pass

strip_and_compile('$CORE_DIR')
strip_and_compile('$BACKEND_DIR')
" 2>/dev/null || true

# Remove compilation script
RUN rm -f /tmp/compile_to_c.py /tmp/core_requirements.txt

# Install the core package (usf_bios)
# First install framework requirements, then install the package
RUN cd $CORE_DIR && pip3 install --no-cache-dir -r requirements/framework.txt 2>/dev/null || true
RUN cd $CORE_DIR && pip3 install --no-cache-dir -e . && \
    echo "✓ usf_bios installed successfully" || \
    (echo "✗ usf_bios installation failed" && exit 1)

# Build frontend
WORKDIR $FRONTEND_DIR
RUN npm install
RUN npm run build

# CRITICAL: Copy static files for Next.js standalone mode
# Standalone doesn't include static assets by default
RUN cp -r .next/static .next/standalone/.next/static
RUN cp -r public .next/standalone/public 2>/dev/null || mkdir -p .next/standalone/public

# Remove frontend source maps and source files
RUN find $FRONTEND_DIR -name "*.map" -type f -delete 2>/dev/null || true
RUN find $FRONTEND_DIR -name "*.ts" -type f ! -path "*node_modules*" -delete 2>/dev/null || true
RUN find $FRONTEND_DIR -name "*.tsx" -type f ! -path "*node_modules*" -delete 2>/dev/null || true
RUN rm -rf $FRONTEND_DIR/src 2>/dev/null || true

# Set restrictive permissions (read+execute only)
RUN chmod -R 500 $CORE_DIR 2>/dev/null || true
RUN chmod -R 500 $BACKEND_DIR 2>/dev/null || true
RUN chmod -R 500 $FRONTEND_DIR/.next 2>/dev/null || true
RUN chown -R root:root $APP_HOME

# Copy RSA public key for log encryption (private key stays with US Inc)
COPY keys/usf_bios_public.pem /app/.k

# Create encrypted log writer utility (uses RSA public key - CANNOT decrypt, only encrypt)
RUN cat > /app/enc_log.pyc << 'ENCLOG'
#!/usr/bin/env python3
import base64,sys
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
def l():
    with open('/app/.k','rb') as f:
        return serialization.load_pem_public_key(f.read(),backend=default_backend())
def e(d):
    k=l()
    return k.encrypt(d.encode('utf-8'),padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
def w(f,m):
    try:
        c=base64.b64encode(e(m))
        with open(f,'ab') as x:x.write(c+b'\n---ENC---\n')
    except:pass
if __name__=="__main__":
    if len(sys.argv)>=3:w(sys.argv[1],sys.argv[2])
ENCLOG
RUN chmod +x /app/enc_log.pyc

# Compile the encryption utility to bytecode and remove source
RUN python3 -c "import py_compile; py_compile.compile('/app/enc_log.pyc', '/app/enc_log.pyc', optimize=2)" 2>/dev/null || true

# Create startup script
RUN cat > /app/start.sh << 'STARTSCRIPT'
#!/bin/bash
set -e

LOG_DIR="/var/log/usf-bios"
mkdir -p $LOG_DIR
chmod 700 $LOG_DIR

echo ""
echo "=================================================="
echo "  USF BIOS - AI Training Platform"
echo "  Powered by US Inc"
echo "=================================================="
echo ""

# Encrypted logging function (RSA public key - cannot decrypt)
enc_log() {
    python3 /app/enc_log.pyc "$LOG_DIR/system.enc" "$(date '+%Y-%m-%d %H:%M:%S') - $1" 2>/dev/null
}

# Status display (minimal info only)
status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
    enc_log "$1"
}

# Check GPU
status "Initializing..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi 2>&1 | python3 /app/enc_log.py "$LOG_DIR/system.enc" "$(cat)" 2>/dev/null
    status "✓ GPU ready"
else
    status "⚠ CPU mode"
fi

# Verify system
status "Verifying..."
python3 << 'PYCHECK' 2>&1
import sys
try:
    import torch
    print(f"  ✓ Core ML")
    if torch.cuda.is_available():
        print(f"  ✓ Accelerator")
except:
    print("  ⚠ Limited mode")
try:
    import transformers,peft
    print(f"  ✓ Training modules")
except:
    pass
print("\n✓ Ready")
PYCHECK

# Start services (no details exposed)
status "Starting..."
cd /app/backend
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 2>&1 | while read line; do
    python3 /app/enc_log.pyc "$LOG_DIR/api.enc" "$line" 2>/dev/null
done &
enc_log "Service 1 initialized"

for i in {1..30}; do
    curl -s http://localhost:8000/health > /dev/null 2>&1 && break
    sleep 1
done

cd /app/frontend
node .next/standalone/server.js 2>&1 | while read line; do
    python3 /app/enc_log.pyc "$LOG_DIR/web.enc" "$line" 2>/dev/null
done &
enc_log "Service 2 initialized"

for i in {1..60}; do
    curl -s http://localhost:3000 > /dev/null 2>&1 && break
    sleep 1
done

status "✓ System ready"
echo ""
echo "=================================================="
echo "  USF BIOS is running"
echo "=================================================="
echo ""
echo "  Access: http://localhost:3000"
echo ""
echo "=================================================="
echo ""

# Keep container running
tail -f /dev/null
STARTSCRIPT

RUN chmod +x /app/start.sh

# Create health check script
RUN cat > /app/healthcheck.sh << 'HEALTHCHECK'
#!/bin/bash
curl -sf http://localhost:8000/health > /dev/null && \
curl -sf http://localhost:3000 > /dev/null
HEALTHCHECK
RUN chmod +x /app/healthcheck.sh

# Expose only frontend port (backend is internal only)
EXPOSE 3000

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command - start all services
CMD ["/app/start.sh"]
