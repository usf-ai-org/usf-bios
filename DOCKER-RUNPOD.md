# USF BIOS - Docker Image for RunPod

## Security Architecture

**Backend API is INTERNAL ONLY** - not exposed outside the container.

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                     │
│   ┌─────────────┐         ┌─────────────────────┐   │
│   │   Next.js   │ ──────► │   FastAPI Backend   │   │
│   │  Frontend   │  proxy  │   (127.0.0.1:8000)  │   │
│   │  Port 3000  │         │   INTERNAL ONLY     │   │
│   └─────────────┘         └─────────────────────┘   │
│         ▲                                           │
└─────────│───────────────────────────────────────────┘
          │
    External Access
    (Port 3000 only)
```

- **Port 3000**: Externally accessible (Next.js frontend)
- **Port 8000**: Internal only (FastAPI backend, bound to 127.0.0.1)
- All `/api/*` requests are proxied through Next.js to the internal backend

## Quick Start

### Step 1: Build and Push to Docker Hub

On your Mac (where the code is):

```bash
cd /Users/apt/Desktop/us_inc/training/train/usf-bios

# Login to Docker Hub
docker login

# Build the image (takes 10-20 minutes)
docker build -t YOUR_DOCKERHUB_USERNAME/usf-bios:latest .

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/usf-bios:latest
```

### Step 2: Deploy on RunPod

1. Go to RunPod dashboard
2. Create a new Pod
3. Select your GPU (e.g., A100, RTX 4090)
4. In **Container Image**, enter:
   ```
   YOUR_DOCKERHUB_USERNAME/usf-bios:latest
   ```
5. Expose ports: **3000, 8000**
6. Deploy

### Step 3: Access the Application

Once deployed, RunPod will provide URLs like:
- **Frontend UI**: `https://xxxxxx-3000.proxy.runpod.net`
- **Backend API**: `https://xxxxxx-8000.proxy.runpod.net`

---

## What the Container Does Automatically

When the container starts, it:

1. ✅ Checks GPU availability
2. ✅ Verifies all packages are installed (PyTorch, Transformers, PEFT, etc.)
3. ✅ Starts backend on port 8000
4. ✅ Starts frontend on port 3000
5. ✅ Logs everything to `/var/log/usf-bios/`

---

## System Status Indicators

The frontend shows system status:

| Status | Meaning | Action |
|--------|---------|--------|
| 🟢 **LIVE** | System ready | Training allowed |
| 🟡 **DEGRADED** | Missing GPU or packages | Training blocked |
| 🔴 **OFFLINE** | Backend not responding | Check logs |
| 🔵 **STARTING** | System initializing | Wait |

**Training is automatically blocked if system is not LIVE.**

---

## Debugging

### Check Logs in RunPod

```bash
# Via RunPod terminal
cat /var/log/usf-bios/startup.log
tail -f /var/log/usf-bios/backend.log
tail -f /var/log/usf-bios/frontend.log
```

### Test System Status

```bash
curl http://localhost:8000/api/system/status
```

Expected response when healthy:
```json
{
  "status": "live",
  "message": "System fully operational - Ready for training",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "cuda_available": true,
  "bios_installed": true,
  "backend_ready": true
}
```

### Check GPU

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Environment Variables (Optional)

You can set these in RunPod:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPU selection |
| `NEXT_PUBLIC_API_URL` | http://localhost:8000 | Backend URL |

---

## Ports

| Port | Service |
|------|---------|
| 3000 | Frontend (Next.js UI) |
| 8000 | Backend (FastAPI) |

---

## Rebuild After Code Changes

```bash
# On Mac
docker build -t YOUR_DOCKERHUB_USERNAME/usf-bios:latest .
docker push YOUR_DOCKERHUB_USERNAME/usf-bios:latest

# On RunPod - restart the pod or:
docker pull YOUR_DOCKERHUB_USERNAME/usf-bios:latest
```

---

## Troubleshooting

### "System Status: OFFLINE"
- Backend not running
- Check: `cat /var/log/usf-bios/backend.log`

### "System Status: DEGRADED"
- GPU not detected or packages missing
- Check: `nvidia-smi` and `curl http://localhost:8000/api/system/status`

### Frontend shows blank page
- Check: `cat /var/log/usf-bios/frontend.log`
- Verify port 3000 is exposed in RunPod

### Training fails immediately
- System status must be "LIVE"
- Check GPU memory: `nvidia-smi`
