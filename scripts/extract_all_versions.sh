#!/bin/bash
# ============================================================================
# USF BIOS - Complete Version Extraction Script
# Run this AFTER Docker build to capture ALL package versions
# This script extracts every single package, library, and tool version
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="${1:-arpitsh018/usf-bios:latest}"
OUTPUT_DIR="${2:-$(pwd)/docs/versions}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  USF BIOS - Complete Version Extraction${NC}"
echo -e "${BLUE}  Image: ${IMAGE_NAME}${NC}"
echo -e "${BLUE}  Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Get version from image (use --entrypoint to prevent default entrypoint from running)
VERSION=$(docker run --rm --entrypoint python ${IMAGE_NAME} -c "import usf_bios; print(usf_bios.__version__)" 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ USF BIOS Version: ${VERSION}${NC}"

OUTPUT_PREFIX="${OUTPUT_DIR}/usf_bios_v${VERSION}_${TIMESTAMP}"

# ============================================================================
# 1. COMPLETE PIP FREEZE - All Python packages with exact versions
# ============================================================================
echo ""
echo -e "${YELLOW}[1/10] Extracting ALL pip packages (pip freeze)...${NC}"
docker run --rm --entrypoint pip ${IMAGE_NAME} freeze > "${OUTPUT_PREFIX}_pip_freeze.txt"
PIP_COUNT=$(wc -l < "${OUTPUT_PREFIX}_pip_freeze.txt")
echo -e "${GREEN}  ✓ ${PIP_COUNT} pip packages captured${NC}"

# ============================================================================
# 2. PIP LIST with versions - Alternative format
# ============================================================================
echo -e "${YELLOW}[2/10] Extracting pip list (formatted)...${NC}"
docker run --rm --entrypoint pip ${IMAGE_NAME} list --format=freeze > "${OUTPUT_PREFIX}_pip_list.txt"
echo -e "${GREEN}  ✓ pip list captured${NC}"

# ============================================================================
# 3. ALL LINUX/APT PACKAGES
# ============================================================================
echo -e "${YELLOW}[3/10] Extracting ALL Linux/apt packages...${NC}"
docker run --rm --entrypoint dpkg-query ${IMAGE_NAME} -W -f='${Package}=${Version}\n' > "${OUTPUT_PREFIX}_apt_packages.txt"
APT_COUNT=$(wc -l < "${OUTPUT_PREFIX}_apt_packages.txt")
echo -e "${GREEN}  ✓ ${APT_COUNT} apt packages captured${NC}"

# ============================================================================
# 4. KEY ML FRAMEWORK VERSIONS (Detailed)
# ============================================================================
echo -e "${YELLOW}[4/10] Extracting ML framework versions...${NC}"
docker run --rm --entrypoint python ${IMAGE_NAME} << 'PYTHON_SCRIPT' > "${OUTPUT_PREFIX}_ml_frameworks.txt"
import sys
print("=" * 80)
print("USF BIOS - ML Framework Versions (Detailed)")
print("=" * 80)
print()

# Helper function
def get_version(module_name, attr='__version__'):
    try:
        mod = __import__(module_name)
        return getattr(mod, attr, 'installed (version unknown)')
    except ImportError:
        return 'NOT INSTALLED'
    except Exception as e:
        return f'ERROR: {e}'

# PyTorch ecosystem
print("=" * 40)
print("PYTORCH ECOSYSTEM")
print("=" * 40)
try:
    import torch
    print(f"torch=={torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch: ERROR - {e}")

print(f"torchvision=={get_version('torchvision')}")
print(f"torchaudio=={get_version('torchaudio')}")
print(f"triton=={get_version('triton')}")
print()

# Transformers ecosystem
print("=" * 40)
print("TRANSFORMERS ECOSYSTEM")
print("=" * 40)
print(f"transformers=={get_version('transformers')}")
print(f"peft=={get_version('peft')}")
print(f"trl=={get_version('trl')}")
print(f"accelerate=={get_version('accelerate')}")
print(f"datasets=={get_version('datasets')}")
print(f"huggingface_hub=={get_version('huggingface_hub')}")
print(f"tokenizers=={get_version('tokenizers')}")
print(f"safetensors=={get_version('safetensors')}")
print(f"sentencepiece=={get_version('sentencepiece')}")
print(f"tiktoken=={get_version('tiktoken')}")
print()

# Attention & Optimization
print("=" * 40)
print("ATTENTION & OPTIMIZATION")
print("=" * 40)
print(f"flash_attn=={get_version('flash_attn')}")
print(f"xformers=={get_version('xformers')}")
print(f"bitsandbytes=={get_version('bitsandbytes')}")
print(f"deepspeed=={get_version('deepspeed')}")
print(f"auto_gptq=={get_version('auto_gptq')}")
print(f"optimum=={get_version('optimum')}")
print(f"liger_kernel=={get_version('liger_kernel')}")
print()

# Inference Engines (optional)
print("=" * 40)
print("INFERENCE ENGINES (Optional)")
print("=" * 40)
print(f"vllm=={get_version('vllm')}")
print(f"sglang=={get_version('sglang')}")
print(f"lmdeploy=={get_version('lmdeploy')}")
print()

# Data Processing
print("=" * 40)
print("DATA PROCESSING")
print("=" * 40)
print(f"numpy=={get_version('numpy')}")
print(f"pandas=={get_version('pandas')}")
print(f"pillow=={get_version('PIL', 'VERSION')}")
try:
    import PIL
    print(f"  (PIL.__version__={PIL.__version__})")
except:
    pass
print(f"scipy=={get_version('scipy')}")
print(f"scikit-learn=={get_version('sklearn', '__version__')}")
print(f"fsspec=={get_version('fsspec')}")
print(f"numba=={get_version('numba')}")
print()

# Training & Monitoring
print("=" * 40)
print("TRAINING & MONITORING")
print("=" * 40)
print(f"tensorboard=={get_version('tensorboard')}")
print(f"wandb=={get_version('wandb')}")
print(f"swanlab=={get_version('swanlab')}")
print(f"mlflow=={get_version('mlflow')}")
print(f"ray=={get_version('ray')}")
print(f"optuna=={get_version('optuna')}")
print(f"evaluate=={get_version('evaluate')}")
print()

# Multimodal
print("=" * 40)
print("MULTIMODAL SUPPORT")
print("=" * 40)
print(f"timm=={get_version('timm')}")
print(f"qwen_vl_utils=={get_version('qwen_vl_utils')}")
print(f"qwen_omni_utils=={get_version('qwen_omni_utils')}")
print(f"decord=={get_version('decord')}")
print(f"librosa=={get_version('librosa')}")
print(f"soundfile=={get_version('soundfile')}")
print(f"openai_whisper=={get_version('whisper')}")
print()

# Web Framework
print("=" * 40)
print("WEB FRAMEWORK")
print("=" * 40)
print(f"fastapi=={get_version('fastapi')}")
print(f"uvicorn=={get_version('uvicorn')}")
print(f"gradio=={get_version('gradio')}")
print(f"pydantic=={get_version('pydantic')}")
print(f"websockets=={get_version('websockets')}")
print(f"aiohttp=={get_version('aiohttp')}")
print()

# Utilities
print("=" * 40)
print("UTILITIES")
print("=" * 40)
print(f"einops=={get_version('einops')}")
print(f"rich=={get_version('rich')}")
print(f"tqdm=={get_version('tqdm')}")
print(f"pyyaml=={get_version('yaml')}")
print(f"omegaconf=={get_version('omegaconf')}")
print(f"cryptography=={get_version('cryptography')}")
print()

# USF BIOS
print("=" * 40)
print("USF BIOS")
print("=" * 40)
print(f"usf_bios=={get_version('usf_bios')}")

print()
print("=" * 80)
print("END OF ML FRAMEWORK VERSIONS")
print("=" * 80)
PYTHON_SCRIPT
echo -e "${GREEN}  ✓ ML framework versions captured${NC}"

# ============================================================================
# 5. SYSTEM INFORMATION
# ============================================================================
echo -e "${YELLOW}[5/10] Extracting system information...${NC}"
docker run --rm --entrypoint bash ${IMAGE_NAME} << 'BASH_SCRIPT' > "${OUTPUT_PREFIX}_system_info.txt"
echo "================================================================================"
echo "USF BIOS - Complete System Information"
echo "Generated: $(date)"
echo "================================================================================"
echo ""

echo "==================== OPERATING SYSTEM ===================="
uname -a
echo ""
cat /etc/os-release
echo ""

echo "==================== CPU INFO ===================="
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/cpuinfo | grep "cpu cores" | head -1
echo ""

echo "==================== MEMORY ===================="
free -h
echo ""

echo "==================== DISK ===================="
df -h / 2>/dev/null || echo "N/A"
echo ""

echo "==================== PYTHON ===================="
python --version
python3 --version 2>/dev/null || true
pip --version
which python
echo ""

echo "==================== CUDA TOOLKIT ===================="
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
echo ""

echo "==================== NVIDIA DRIVER ===================="
cat /proc/driver/nvidia/version 2>/dev/null || echo "No NVIDIA driver info"
echo ""

echo "==================== CUDA LIBRARIES ===================="
ls -la /usr/local/cuda/lib64/*.so* 2>/dev/null | head -20 || echo "No CUDA libs found"
echo ""

echo "==================== cuDNN ===================="
cat /usr/include/cudnn_version.h 2>/dev/null | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" | head -3 || echo "cuDNN header not found"
echo ""

echo "==================== NODE.JS ===================="
node --version 2>/dev/null || echo "Node.js not installed"
npm --version 2>/dev/null || echo "npm not installed"
echo ""

echo "==================== BUILD TOOLS ===================="
gcc --version 2>/dev/null | head -1 || echo "gcc not installed"
g++ --version 2>/dev/null | head -1 || echo "g++ not installed"
cmake --version 2>/dev/null | head -1 || echo "cmake not installed"
ninja --version 2>/dev/null || echo "ninja not installed"
make --version 2>/dev/null | head -1 || echo "make not installed"
git --version 2>/dev/null || echo "git not installed"
echo ""

echo "==================== ENVIRONMENT VARIABLES ===================="
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "CUDA_PATH=${CUDA_PATH:-not set}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-not set}"
echo "HF_HOME=${HF_HOME:-not set}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-not set}"
echo ""

echo "================================================================================"
echo "END OF SYSTEM INFORMATION"
echo "================================================================================"
BASH_SCRIPT
echo -e "${GREEN}  ✓ System information captured${NC}"

# ============================================================================
# 6. CUDA DETAILED INFO
# ============================================================================
echo -e "${YELLOW}[6/10] Extracting CUDA detailed info...${NC}"
docker run --rm --gpus all --entrypoint bash ${IMAGE_NAME} << 'BASH_SCRIPT' > "${OUTPUT_PREFIX}_cuda_info.txt" 2>&1 || true
echo "================================================================================"
echo "USF BIOS - CUDA Detailed Information"
echo "================================================================================"
echo ""

echo "==================== nvidia-smi ===================="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (no GPU access in this container)"
echo ""

echo "==================== nvidia-smi -L ===================="
nvidia-smi -L 2>/dev/null || echo "N/A"
echo ""

echo "==================== CUDA Device Query ===================="
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA Available: True')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'Device Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'')
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Total Memory: {props.total_memory / 1024**3:.2f} GB')
        print(f'  Multi Processor Count: {props.multi_processor_count}')
        print(f'  Compute Capability: {props.major}.{props.minor}')
else:
    print('CUDA Available: False')
" 2>/dev/null || echo "Could not query CUDA devices"
echo ""

echo "================================================================================"
echo "END OF CUDA INFORMATION"
echo "================================================================================"
BASH_SCRIPT
echo -e "${GREEN}  ✓ CUDA info captured${NC}"

# ============================================================================
# 7. INSTALLED SHARED LIBRARIES (.so files)
# ============================================================================
echo -e "${YELLOW}[7/10] Extracting shared libraries info...${NC}"
docker run --rm --entrypoint bash ${IMAGE_NAME} << 'BASH_SCRIPT' > "${OUTPUT_PREFIX}_shared_libs.txt"
echo "================================================================================"
echo "USF BIOS - Installed Shared Libraries"
echo "================================================================================"
echo ""

echo "==================== Python Site-Packages .so Files ===================="
find /usr/local/lib/python3.11/dist-packages -name "*.so" 2>/dev/null | head -100
echo ""
echo "Total .so files in site-packages:"
find /usr/local/lib/python3.11/dist-packages -name "*.so" 2>/dev/null | wc -l
echo ""

echo "==================== CUDA Libraries ===================="
ls -la /usr/local/cuda/lib64/*.so* 2>/dev/null | head -50 || echo "No CUDA libs"
echo ""

echo "================================================================================"
BASH_SCRIPT
echo -e "${GREEN}  ✓ Shared libraries captured${NC}"

# ============================================================================
# 8. EXTERNAL REPOS (Pre-cloned)
# ============================================================================
echo -e "${YELLOW}[8/10] Extracting external repos info...${NC}"
docker run --rm --entrypoint bash ${IMAGE_NAME} << 'BASH_SCRIPT' > "${OUTPUT_PREFIX}_external_repos.txt"
echo "================================================================================"
echo "USF BIOS - Pre-cloned External Repositories"
echo "================================================================================"
echo ""

if [ -d /app/external_repos ]; then
    for repo in /app/external_repos/*/; do
        if [ -d "$repo" ]; then
            repo_name=$(basename "$repo")
            echo "Repository: $repo_name"
            if [ -d "$repo/.git" ]; then
                cd "$repo"
                echo "  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
                echo "  Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
                echo "  Date: $(git log -1 --format=%cd 2>/dev/null || echo 'unknown')"
            fi
            echo ""
        fi
    done
else
    echo "No external repos directory found"
fi

echo "================================================================================"
BASH_SCRIPT
echo -e "${GREEN}  ✓ External repos captured${NC}"

# ============================================================================
# 9. CREATE COMBINED SUMMARY
# ============================================================================
echo -e "${YELLOW}[9/10] Creating combined summary...${NC}"
cat > "${OUTPUT_PREFIX}_SUMMARY.txt" << EOF
================================================================================
USF BIOS v${VERSION} - Complete Version Summary
Generated: $(date)
Image: ${IMAGE_NAME}
================================================================================

This archive contains complete version information for reproducing the build:

FILES INCLUDED:
- _pip_freeze.txt       : All pip packages with exact versions (${PIP_COUNT} packages)
- _pip_list.txt         : Alternative pip list format
- _apt_packages.txt     : All Linux/apt packages (${APT_COUNT} packages)
- _ml_frameworks.txt    : Detailed ML framework versions
- _system_info.txt      : System and build tool information
- _cuda_info.txt        : CUDA and GPU information
- _shared_libs.txt      : Installed shared libraries
- _external_repos.txt   : Pre-cloned external repositories
- _SUMMARY.txt          : This file

QUICK REFERENCE (Key Packages):
$(docker run --rm --entrypoint python ${IMAGE_NAME} << 'PYEOF'
def get_ver(m, a='__version__'):
    try:
        mod = __import__(m)
        return getattr(mod, a, '?')
    except:
        return 'N/A'

print(f"  torch=={get_ver('torch')}")
print(f"  transformers=={get_ver('transformers')}")
print(f"  peft=={get_ver('peft')}")
print(f"  trl=={get_ver('trl')}")
print(f"  accelerate=={get_ver('accelerate')}")
print(f"  deepspeed=={get_ver('deepspeed')}")
print(f"  flash_attn=={get_ver('flash_attn')}")
print(f"  xformers=={get_ver('xformers')}")
print(f"  bitsandbytes=={get_ver('bitsandbytes')}")
print(f"  vllm=={get_ver('vllm')}")
print(f"  sglang=={get_ver('sglang')}")
print(f"  lmdeploy=={get_ver('lmdeploy')}")
print(f"  numpy=={get_ver('numpy')}")
print(f"  datasets=={get_ver('datasets')}")
print(f"  usf_bios=={get_ver('usf_bios')}")
PYEOF
)

================================================================================
EOF
echo -e "${GREEN}  ✓ Summary created${NC}"

# ============================================================================
# 10. CREATE ZIP/TAR ARCHIVE (optional - skip if tools not available)
# ============================================================================
echo -e "${YELLOW}[10/10] Creating archive...${NC}"
cd "${OUTPUT_DIR}"
ZIP_NAME="usf_bios_v${VERSION}_${TIMESTAMP}_versions.zip"
TAR_NAME="usf_bios_v${VERSION}_${TIMESTAMP}_versions.tar.gz"
if command -v zip &> /dev/null; then
    zip -q "${ZIP_NAME}" usf_bios_v${VERSION}_${TIMESTAMP}_*.txt
    echo -e "${GREEN}  ✓ Archive created: ${ZIP_NAME}${NC}"
elif command -v tar &> /dev/null; then
    tar -czf "${TAR_NAME}" usf_bios_v${VERSION}_${TIMESTAMP}_*.txt
    echo -e "${GREEN}  ✓ Archive created: ${TAR_NAME}${NC}"
else
    echo -e "${YELLOW}  ⚠ zip/tar not available - skipping archive (files still saved)${NC}"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  ✓ VERSION EXTRACTION COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "Output files in: ${BLUE}${OUTPUT_DIR}${NC}"
echo ""
ls -lh "${OUTPUT_DIR}"/usf_bios_v${VERSION}_${TIMESTAMP}_* 2>/dev/null
echo ""
echo -e "${GREEN}Archive: ${OUTPUT_DIR}/${ZIP_NAME}${NC}"
echo ""
echo -e "${YELLOW}To view key versions:${NC}"
echo "  cat ${OUTPUT_PREFIX}_ml_frameworks.txt"
echo ""
echo -e "${YELLOW}To view all pip packages:${NC}"
echo "  cat ${OUTPUT_PREFIX}_pip_freeze.txt"
echo ""
