#!/bin/bash
# USF BIOS - AKS Deployment Script
# Usage: ./deploy.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="usf-bios"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  USF BIOS - AKS Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}ERROR: kubectl not found. Please install kubectl.${NC}"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}ERROR: Not connected to a Kubernetes cluster.${NC}"
    echo "Run: az aks get-credentials --resource-group <rg> --name <cluster>"
    exit 1
fi

echo -e "  ✓ kubectl connected to cluster"

# Check for GPU nodes
echo -e "\n${YELLOW}[2/6] Checking for GPU nodes...${NC}"
GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable["nvidia.com/gpu"] != null) | .metadata.name' 2>/dev/null || echo "")

if [ -z "$GPU_NODES" ]; then
    echo -e "${YELLOW}  ⚠ No GPU nodes detected. Training will require GPU nodes.${NC}"
    echo -e "  To add GPU nodes:"
    echo -e "    az aks nodepool add --resource-group <rg> --cluster-name <cluster> \\"
    echo -e "      --name gpupool --node-count 1 --node-vm-size Standard_NC6s_v3 \\"
    echo -e "      --enable-cluster-autoscaler --min-count 0 --max-count 3"
else
    echo -e "  ✓ GPU nodes found:"
    echo "$GPU_NODES" | while read node; do
        GPU_COUNT=$(kubectl get node "$node" -o json | jq -r '.status.allocatable["nvidia.com/gpu"]')
        echo -e "    - $node ($GPU_COUNT GPUs)"
    done
fi

# Apply namespace
echo -e "\n${YELLOW}[3/6] Creating namespace...${NC}"
if [ "$1" == "--dry-run" ]; then
    kubectl apply -f "$SCRIPT_DIR/namespace.yaml" --dry-run=client
else
    kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
fi
echo -e "  ✓ Namespace created: $NAMESPACE"

# Apply storage
echo -e "\n${YELLOW}[4/6] Setting up storage (1TB PVC)...${NC}"
if [ "$1" == "--dry-run" ]; then
    kubectl apply -f "$SCRIPT_DIR/storage.yaml" --dry-run=client
else
    kubectl apply -f "$SCRIPT_DIR/storage.yaml"
fi
echo -e "  ✓ StorageClass and PVC created"

# Apply config
echo -e "\n${YELLOW}[5/6] Applying configuration...${NC}"
if [ "$1" == "--dry-run" ]; then
    kubectl apply -f "$SCRIPT_DIR/configmap.yaml" --dry-run=client
else
    kubectl apply -f "$SCRIPT_DIR/configmap.yaml"
fi
echo -e "  ✓ ConfigMap and Secrets applied"

# Deploy application
echo -e "\n${YELLOW}[6/6] Deploying application...${NC}"
if [ "$1" == "--dry-run" ]; then
    kubectl apply -f "$SCRIPT_DIR/deployment.yaml" --dry-run=client
    kubectl apply -f "$SCRIPT_DIR/service.yaml" --dry-run=client
else
    kubectl apply -f "$SCRIPT_DIR/deployment.yaml"
    kubectl apply -f "$SCRIPT_DIR/service.yaml"
fi
echo -e "  ✓ Deployment and Service created"

# Wait for deployment
if [ "$1" != "--dry-run" ]; then
    echo -e "\n${YELLOW}Waiting for deployment to be ready...${NC}"
    kubectl rollout status deployment/usf-bios -n $NAMESPACE --timeout=600s || true
    
    # Get service IP
    echo -e "\n${YELLOW}Getting service endpoint...${NC}"
    sleep 10
    EXTERNAL_IP=$(kubectl get svc usf-bios-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$EXTERNAL_IP" == "pending" ] || [ -z "$EXTERNAL_IP" ]; then
        echo -e "  ⏳ External IP is still being provisioned..."
        echo -e "  Run: kubectl get svc usf-bios-service -n $NAMESPACE --watch"
    else
        echo -e "  ✓ External IP: ${GREEN}$EXTERNAL_IP${NC}"
        echo -e "  Access UI at: ${GREEN}http://$EXTERNAL_IP${NC}"
    fi
fi

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "\nUseful commands:"
echo -e "  ${YELLOW}kubectl get pods -n $NAMESPACE${NC}              # Check pods"
echo -e "  ${YELLOW}kubectl logs -f -n $NAMESPACE deploy/usf-bios${NC} # View logs"
echo -e "  ${YELLOW}kubectl get svc -n $NAMESPACE${NC}               # Get service IP"
echo -e "  ${YELLOW}kubectl exec -it -n $NAMESPACE deploy/usf-bios -- bash${NC} # Shell access"
echo -e "\nStorage info:"
echo -e "  ${YELLOW}kubectl get pvc -n $NAMESPACE${NC}               # Check PVC status"
echo -e "  ${YELLOW}kubectl describe pvc usf-storage-pvc -n $NAMESPACE${NC}"
echo ""
