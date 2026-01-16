# USF BIOS - GPU Server Setup Guide

## Quick Start (After Cloning Repository)

### Step 1: Clone Repository on GPU Server

```bash
cd /workspace
git clone <your-repo-url> usf-bios
cd usf-bios
```

### Step 2: Start with Docker (Recommended)

```bash
# Build and start all services
docker compose up --build -d

# Check if services are running
docker compose ps

# View logs
docker compose logs -f
```

### Step 3: Test Everything

```bash
# Test 1: Check backend health
curl http://localhost:8000/health

# Test 2: Check GPU metrics (MOST IMPORTANT)
curl http://localhost:8000/api/system/metrics

# Test 3: Open UI in browser
# http://<your-server-ip>:3000
```

---

## Expected Responses

### Health Check (`/health`)
```json
{"status": "healthy"}
```

### GPU Metrics (`/api/system/metrics`)
```json
{
  "gpu_utilization": 0,
  "gpu_memory_used": 0.5,
  "gpu_memory_total": 24.0,
  "gpu_temperature": 35,
  "cpu_percent": 5.2,
  "ram_used": 4.1,
  "ram_total": 64.0
}
```

If `gpu_memory_total` is `0` or `null`, GPU is not detected properly.

---

## Debugging

### Check Container Status
```bash
docker compose ps
docker compose logs backend
docker compose logs frontend
```

### Enter Backend Container for Debugging
```bash
docker compose exec backend bash

# Inside container, test GPU:
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK')"
nvidia-smi
```

### Check GPU Access
```bash
# On host (not in container)
nvidia-smi

# Should show your GPU(s)
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `nvidia-smi` not found in container | Install nvidia-container-toolkit on host |
| GPU metrics show N/A | Check NVIDIA driver and pynvml installation |
| Backend not starting | Check logs: `docker compose logs backend` |
| Frontend can't reach backend | Check network: `docker compose exec frontend ping backend` |

### Install NVIDIA Container Toolkit (if needed)
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

---

## Manual Setup (Without Docker)

If Docker doesn't work, run manually:

```bash
# Terminal 1: Backend
cd /workspace/usf-bios/web/backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd /workspace/usf-bios/web/frontend
npm install
npm run build
npm start
```

---

## Stop Services

```bash
docker compose down
```

---

## Rebuild After Code Changes

```bash
docker compose down
docker compose up --build -d
```

---

## Expose Ports (RunPod)

In RunPod, make sure ports **3000** and **8000** are exposed:
- Go to Pod settings
- Add HTTP ports: 3000, 8000
- Access via the provided URLs

---

## Test Training (After UI Works)

1. Open `http://<server-ip>:3000`
2. Go to Fine-tuning tab
3. Select a small model (e.g., `Qwen/Qwen2.5-0.5B`)
4. Upload or select a dataset
5. Start training with LoRA
6. Watch metrics update in real-time

If metrics show real values during training, everything is working!
