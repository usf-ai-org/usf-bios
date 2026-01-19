# USF BIOS - AKS Deployment Guide

## Prerequisites

1. **Azure CLI** installed and logged in
2. **kubectl** configured to your AKS cluster
3. **AKS cluster with GPU node pool** (e.g., Standard_NC6s_v3)
4. **NVIDIA device plugin** installed on cluster

## Quick Start

```bash
# 1. Connect to your AKS cluster
az aks get-credentials --resource-group <resource-group> --name <cluster-name>

# 2. Deploy USF BIOS
chmod +x deploy.sh
./deploy.sh

# 3. (Optional) Pre-download models
kubectl apply -f preload-models.yaml
kubectl logs -f job/usf-model-preloader -n usf-bios

# 4. Access the UI
kubectl get svc usf-bios-service -n usf-bios
# Open the EXTERNAL-IP in your browser
```

## Storage Layout (1TB)

```
/mnt/usf-storage/                # Mounted from Azure Premium SSD
├── data/
│   ├── uploads/                 # User uploaded datasets
│   ├── output/                  # Trained LoRA adapters
│   ├── checkpoints/             # Training checkpoints
│   └── logs/                    # Training logs
├── models/                      # Pre-downloaded base models
│   └── Qwen--Qwen2.5-7B-Instruct/
└── cache/                       # HuggingFace cache
    ├── hub/
    └── datasets/
```

## Configuration

Edit `configmap.yaml` to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPPORTED_MODEL_SOURCES` | Allowed sources (huggingface,modelscope,local) | All |
| `SUPPORTED_ARCHITECTURES` | Allowed architectures (empty = all) | All |
| `MAX_CONCURRENT_JOBS` | Max parallel training jobs | 1 |

## Accessing Trained Models

### Method 1: kubectl cp
```bash
# Copy trained adapter to local machine
kubectl cp usf-bios/usf-bios-<pod>:/mnt/usf-storage/data/output/<job_id> ./my-adapter/
```

### Method 2: Azure Storage
The PVC is backed by Azure Disk. You can:
1. Create snapshots for backup
2. Mount to other pods for inference

## Monitoring

```bash
# View pod logs
kubectl logs -f deploy/usf-bios -n usf-bios

# Check GPU usage in pod
kubectl exec -it deploy/usf-bios -n usf-bios -- nvidia-smi

# Check storage usage
kubectl exec -it deploy/usf-bios -n usf-bios -- df -h /mnt/usf-storage
```

## Troubleshooting

### Pod pending (no GPU)
```bash
kubectl describe pod -n usf-bios
# Check for: "Insufficient nvidia.com/gpu"
# Solution: Add GPU node pool to cluster
```

### Storage not provisioning
```bash
kubectl get pvc -n usf-bios
kubectl describe pvc usf-storage-pvc -n usf-bios
```

### Out of GPU memory
- Use smaller batch size
- Enable gradient checkpointing
- Use 4-bit quantization

## Files

| File | Purpose |
|------|---------|
| `namespace.yaml` | Creates usf-bios namespace |
| `storage.yaml` | 1TB PVC with Premium SSD |
| `configmap.yaml` | Environment configuration |
| `deployment.yaml` | Main application deployment |
| `service.yaml` | LoadBalancer service |
| `ingress.yaml` | Optional ingress for custom domain |
| `preload-models.yaml` | Job to pre-download models |
| `deploy.sh` | Deployment script |
| `AKS-COMPUTE-FLOW.md` | Detailed compute flow documentation |
