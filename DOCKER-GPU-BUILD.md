# USF BIOS - Docker GPU Build Guide

**Version: 2.0.11**

Complete guide for building and deploying USF BIOS Docker image on GPU servers.

---

## Prerequisites

### Hardware Requirements
- **GPU Server**: H200, H100, A100, or RTX 4090
- **RAM**: Minimum 32GB (64GB recommended)
- **Disk**: 50GB free space for Docker build
- **Network**: Good bandwidth for downloading packages

### Software Requirements
- Docker with NVIDIA Container Toolkit
- Git
- Python 3.11+ (for version extraction)

---

## Quick Start

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/apt-team-018/usf-bios.git
cd usf-bios

# Or if already cloned, pull latest
git pull origin main
```

### Step 2: Set GitHub Token

Required for installing the custom USF Transformers fork:

```bash
export GITHUB_TOKEN=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
```

### Step 3: Build Docker Image

```bash
# Build with automatic version detection (recommended)
./scripts/build-docker-gpu.sh

# Or specify version manually
./scripts/build-docker-gpu.sh 2.0.11
```

**Build Time Estimates:**
| GPU | First Build | Cached Build |
|-----|-------------|--------------|
| H200 | ~15-20 min | ~2-5 min |
| H100 | ~20-25 min | ~2-5 min |
| A100 | ~25-30 min | ~3-5 min |

### Step 4: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the image
docker push arpitsh018/usf-bios:2.0.11
```

### Step 5: Deploy on RunPod

1. Go to [RunPod Dashboard](https://runpod.io/console/pods)
2. Create a new Pod with GPU (H100/A100 recommended)
3. Set container image: `arpitsh018/usf-bios:2.0.11`
4. Expose port: **3000**
5. Deploy

---

## Package Versions

The Docker image includes these pre-compiled packages:

### Core ML Stack
| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.4.0 | CUDA 12.4 |
| Transformers | 4.57.6 | USF Custom Fork |
| PEFT | 0.11-0.18 | LoRA/QLoRA support |
| TRL | 0.15-0.24 | RLHF training |
| Accelerate | Latest | Multi-GPU support |
| DeepSpeed | Latest | ZeRO optimization |

### CUDA Acceleration
| Package | Version | Notes |
|---------|---------|-------|
| Flash Attention 2 | 2.6.3 | Ampere+ GPUs |
| Flash Attention 3 | Latest | Hopper GPUs only |
| xformers | 0.0.27.post2 | Memory efficient |
| bitsandbytes | Latest | 4/8-bit quantization |
| Triton | 3.0-3.4 | GPU kernels |

### Training Utilities
| Package | Version |
|---------|---------|
| datasets | 3.0+ |
| evaluate | Latest |
| tensorboard | Latest |
| wandb | Latest |
| liger-kernel | Latest |

---

## Get Package Versions from Running Container

After deployment, get all installed package versions:

```bash
# SSH into RunPod container, then:
pip list --format=json > /app/data/pip_packages.json

# Or use the built-in version report
cat /app/data/version_report.json
```

### Quick Version Check Script

```bash
python3 << 'EOF'
import torch
import transformers
import peft
import trl
import accelerate
import deepspeed

print("=" * 50)
print("USF BIOS - Package Versions")
print("=" * 50)
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA:          {torch.version.cuda}")
print(f"Transformers:  {transformers.__version__}")
print(f"PEFT:          {peft.__version__}")
print(f"TRL:           {trl.__version__}")
print(f"Accelerate:    {accelerate.__version__}")
print(f"DeepSpeed:     {deepspeed.__version__}")

try:
    import flash_attn
    print(f"Flash Attn:    {flash_attn.__version__}")
except:
    print("Flash Attn:    Not installed")

try:
    import bitsandbytes
    print(f"bitsandbytes:  {bitsandbytes.__version__}")
except:
    print("bitsandbytes:  Not installed")

print("=" * 50)
EOF
```

---

## Build Script Details

The `./scripts/build-docker-gpu.sh` script:

1. **Extracts version** from `usf_bios/version.py`
2. **Verifies NVIDIA GPU** is available
3. **Builds with BuildKit** for layer caching
4. **Tags image** as `arpitsh018/usf-bios:VERSION`
5. **Optionally pushes** to Docker Hub

### Script Options

```bash
# Default: Build with cache, push to Docker Hub
./scripts/build-docker-gpu.sh

# Build without cache (fresh build) - use when you need to force rebuild all layers
./scripts/build-docker-gpu.sh --no-cache

# Build only, do not push to Docker Hub
./scripts/build-docker-gpu.sh --no-push

# Build specific version
./scripts/build-docker-gpu.sh 2.0.11

# Fresh build, no push, specific version
./scripts/build-docker-gpu.sh --no-cache --no-push 2.0.11

# Show help
./scripts/build-docker-gpu.sh --help
```

### When to Use `--no-cache`

Use `--no-cache` when:
- Package versions need updating (e.g., new transformers release)
- Build is failing due to corrupted cache
- You've modified Dockerfile and cache isn't invalidating
- Testing a completely fresh build

---

## Dockerfile.gpu Build Stages

| Stage | Purpose | Base Image |
|-------|---------|------------|
| 0 | USF BIOS verification | python:3.11-slim |
| 1 | Frontend build | node:20-alpine |
| 2 | Python compilation | python:3.11-slim |
| 3 | Production image | nvidia/cuda:12.4.0-devel-ubuntu22.04 |

### Key Features

- ✅ **CUDA 12.4** with full toolkit (nvcc, cuBLAS, cuDNN)
- ✅ **Python 3.11** via deadsnakes PPA
- ✅ **Cython compilation** for code protection
- ✅ **Pre-compiled CUDA packages** (no runtime compilation)
- ✅ **Non-root user** for security
- ✅ **Health checks** for container orchestration

---

## Troubleshooting

### Build Fails at Transformers Install

```bash
# Ensure GITHUB_TOKEN is exported
export GITHUB_TOKEN=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

# Verify token works
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

### Build Fails at Flash Attention

Flash Attention requires NVIDIA GPU during build:

```bash
# Verify GPU is available
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

### Out of Disk Space

```bash
# Clean Docker cache
docker system prune -a
docker builder prune -a
```

### Image Too Large

The final image is ~25-30GB due to CUDA toolkit. This is expected.

---

## File Structure

```
usf-bios/
├── web/
│   └── Dockerfile.gpu          # Main GPU Dockerfile
├── scripts/
│   ├── build-docker-gpu.sh     # Build script
│   ├── compile_to_so.py        # Cython compilation
│   └── capture_versions.py     # Version documentation
├── usf_bios/
│   ├── version.py              # Version: 2.0.11
│   └── system_guard.py         # Model lock configuration
└── DOCKER-GPU-BUILD.md         # This file
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.11 | 2026-01-24 | Dynamic GPU selection, Model lock fix, Token support |
| 2.0.10 | 2026-01-24 | Version label updates |
| 2.0.09 | 2026-01-23 | Initial GPU build |

---

## Support

For issues, contact: support@us.inc
