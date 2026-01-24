#!/bin/bash
# ============================================================================
# USF BIOS - GPU Docker Build Script
# Run this on H200/H100 GPU server for fast CUDA compilation
# ============================================================================
#
# USAGE:
#   ./scripts/build-docker-gpu.sh [VERSION]
#
# EXAMPLE:
#   ./scripts/build-docker-gpu.sh 2.0.0
#
# REQUIREMENTS:
#   - NVIDIA GPU (H200, H100, A100, etc.)
#   - Docker with nvidia-container-toolkit
#   - ~50GB disk space
#
# BUILD TIME ESTIMATES:
#   - H200: ~15-20 minutes (first build), ~2-5 min (cached)
#   - H100: ~20-25 minutes (first build), ~2-5 min (cached)
#   - A100: ~25-30 minutes (first build), ~3-5 min (cached)
#   - GitHub Actions (no GPU): 6+ hours (timeout)
#
# CACHING:
#   Uses BuildKit cache for fast rebuilds. Cache is stored locally.
#   Only changed layers are rebuilt.
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project root first (needed for version extraction)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Extract version dynamically from usf_bios/version.py
DYNAMIC_VERSION=$(python3 -c "exec(open('usf_bios/version.py').read()); print(__version__)" 2>/dev/null || echo "2.0.08")
VERSION="${1:-$DYNAMIC_VERSION}"

# Docker Hub image name
IMAGE_NAME="arpitsh018/usf-bios"
DOCKERFILE="web/Dockerfile.gpu"

echo -e "${GREEN}"
echo "============================================================================"
echo "  USF BIOS - GPU Docker Build"
echo "  Version: ${VERSION}"
echo "  Image: ${IMAGE_NAME}:${VERSION}"
echo "============================================================================"
echo -e "${NC}"

# Check for NVIDIA GPU
echo -e "${YELLOW}[1/5] Checking GPU availability...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. This script requires NVIDIA GPU.${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo -e "${GREEN}  ✓ GPU detected: ${GPU_NAME}${NC}"

# Check Docker
echo -e "${YELLOW}[2/5] Checking Docker...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker not running.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker is running${NC}"

# Check nvidia-container-toolkit
echo -e "${YELLOW}[3/5] Checking nvidia-container-toolkit...${NC}"
if ! docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-container-toolkit not configured.${NC}"
    echo "Install with: sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi
echo -e "${GREEN}  ✓ nvidia-container-toolkit working${NC}"

# Already at project root
echo -e "${YELLOW}[4/5] Building from: $(pwd)${NC}"

# Build with GPU support and BuildKit
echo -e "${YELLOW}[5/5] Starting Docker build (BuildKit enabled)...${NC}"
echo ""

# Enable BuildKit for better caching and parallel builds
export DOCKER_BUILDKIT=1

# Use regular docker build (has access to host GPU for compilation)
# Note: buildx with docker-container driver does NOT have GPU access
docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --build-arg DS_BUILD_OPS=1 \
    --build-arg DS_BUILD_AIO=1 \
    --build-arg DS_BUILD_SPARSE_ATTN=1 \
    --build-arg DS_BUILD_TRANSFORMER=1 \
    --build-arg DS_BUILD_TRANSFORMER_INFERENCE=1 \
    --build-arg DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    --build-arg DS_BUILD_FUSED_ADAM=1 \
    --build-arg DS_BUILD_FUSED_LAMB=1 \
    --build-arg DS_BUILD_CPU_ADAM=1 \
    --build-arg DS_BUILD_CPU_LION=1 \
    --build-arg DS_BUILD_UTILS=1 \
    --build-arg DS_BUILD_EVOFORMER_ATTN=1 \
    --build-arg DS_BUILD_RANDOM_LTD=1 \
    --build-arg DS_BUILD_INFERENCE_CORE_OPS=1 \
    --build-arg DS_BUILD_CUTLASS_OPS=1 \
    --build-arg DS_BUILD_RAGGED_DEVICE_OPS=1 \
    --progress=plain \
    .

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  ✓ BUILD COMPLETE${NC}"
echo -e "${GREEN}  Image: ${IMAGE_NAME}:${VERSION}${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

# Show image info
docker images "${IMAGE_NAME}:${VERSION}"

# ============================================================================
# EXTRACT ALL VERSIONS FROM BUILT IMAGE
# ============================================================================
echo ""
echo -e "${YELLOW}[6/8] Extracting complete version report from image...${NC}"
echo ""

VERSION_DIR="${PROJECT_ROOT}/docs/versions"
mkdir -p "${VERSION_DIR}"

# Extract version reports from the built image
CONTAINER_ID=$(docker create ${IMAGE_NAME}:${VERSION})
docker cp ${CONTAINER_ID}:/app/data/version_report.json "${VERSION_DIR}/version_report_v${VERSION}.json" 2>/dev/null || echo "JSON report not found"
docker cp ${CONTAINER_ID}:/app/data/version_report.txt "${VERSION_DIR}/version_report_v${VERSION}.txt" 2>/dev/null || echo "TXT report not found"
docker rm ${CONTAINER_ID} > /dev/null

# Also run pip freeze directly to get exact versions (use --entrypoint to override default)
echo -e "${YELLOW}[7/8] Running pip freeze to capture exact versions...${NC}"
timeout 60 docker run --rm --entrypoint pip ${IMAGE_NAME}:${VERSION} freeze > "${VERSION_DIR}/pip_freeze_v${VERSION}.txt" 2>/dev/null || echo "pip freeze timed out or failed"

# Get dpkg list for Linux packages
timeout 30 docker run --rm --entrypoint dpkg-query ${IMAGE_NAME}:${VERSION} -W -f='${Package}=${Version}\n' > "${VERSION_DIR}/dpkg_list_v${VERSION}.txt" 2>/dev/null || echo "dpkg query timed out or failed"

# Get detailed system info (quick version - no hanging imports)
timeout 60 docker run --rm --entrypoint bash ${IMAGE_NAME}:${VERSION} -c "
echo '# USF BIOS v${VERSION} - Complete System Information'
echo '# Generated: \$(date)'
echo ''
echo '================== SYSTEM =================='
uname -a
cat /etc/os-release
echo ''
echo '================== PYTHON =================='
python --version
pip --version
echo ''
echo '================== NODE.JS =================='
node --version
npm --version
echo ''
echo '================== KEY ML PACKAGES =================='
pip show torch transformers peft trl accelerate deepspeed flash-attn xformers bitsandbytes datasets numpy 2>/dev/null | grep -E '^(Name|Version):'
echo ''
echo '================== USF BIOS =================='
python -c 'import usf_bios; print(f\"usf_bios=={usf_bios.__version__}\")' 2>/dev/null || echo 'usf_bios not found'
echo ''
echo '================== PIP PACKAGE COUNT =================='
pip list 2>/dev/null | wc -l
" > "${VERSION_DIR}/system_info_v${VERSION}.txt" 2>/dev/null || echo "system info extraction timed out"

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  ✓ VERSION REPORTS EXTRACTED${NC}"
echo -e "${GREEN}  Location: ${VERSION_DIR}/${NC}"
echo -e "${GREEN}  Files:${NC}"
ls -la "${VERSION_DIR}"/*v${VERSION}* 2>/dev/null | awk '{print "    " $NF}'
echo -e "${GREEN}============================================================================${NC}"
echo ""

echo ""
echo -e "${YELLOW}[8/8] Pushing to Docker Hub...${NC}"
echo ""

# Push to Docker Hub
docker push ${IMAGE_NAME}:${VERSION}
docker push ${IMAGE_NAME}:latest

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  ✓ PUSH COMPLETE${NC}"
echo -e "${GREEN}  Image: ${IMAGE_NAME}:${VERSION}${NC}"
echo -e "${GREEN}  Image: ${IMAGE_NAME}:latest${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "${GREEN}  Version reports saved to:${NC}"
echo "    ${VERSION_DIR}/version_report_v${VERSION}.json"
echo "    ${VERSION_DIR}/version_report_v${VERSION}.txt"
echo "    ${VERSION_DIR}/pip_freeze_v${VERSION}.txt"
echo "    ${VERSION_DIR}/dpkg_list_v${VERSION}.txt"
echo "    ${VERSION_DIR}/system_info_v${VERSION}.txt"
echo ""
echo -e "${YELLOW}To run:${NC}"
echo "  docker run --gpus all -p 3000:3000 ${IMAGE_NAME}:${VERSION}"
