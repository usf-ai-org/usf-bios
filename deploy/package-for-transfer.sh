#!/bin/bash
# USF BIOS - Package code for transfer to GPU server
# Run this on your Mac to create a deployment package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_NAME="usf-bios-deploy-$(date +%Y%m%d-%H%M%S).tar.gz"
OUTPUT_DIR="$HOME/Desktop"

echo "=========================================="
echo "USF BIOS - Creating Deployment Package"
echo "=========================================="

cd "$PROJECT_ROOT"

# Create tarball excluding unnecessary files
echo "Creating package: $OUTPUT_DIR/$PACKAGE_NAME"

tar -czvf "$OUTPUT_DIR/$PACKAGE_NAME" \
    --exclude='node_modules' \
    --exclude='.next' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='*.egg-info' \
    .

echo ""
echo "=========================================="
echo "Package Created!"
echo "=========================================="
echo ""
echo "File: $OUTPUT_DIR/$PACKAGE_NAME"
echo "Size: $(du -h "$OUTPUT_DIR/$PACKAGE_NAME" | cut -f1)"
echo ""
echo "To transfer to GPU server, use one of these methods:"
echo ""
echo "Option 1 - SCP (if you have SSH access):"
echo "  scp $OUTPUT_DIR/$PACKAGE_NAME user@your-gpu-server:/workspace/"
echo ""
echo "Option 2 - RunPod File Browser:"
echo "  1. Open RunPod console"
echo "  2. Use the file upload feature"
echo "  3. Upload to /workspace/"
echo ""
echo "Option 3 - Git (if repo is on GitHub/GitLab):"
echo "  git clone your-repo-url /workspace/usf-bios"
echo ""
echo "After transfer, on the GPU server run:"
echo "  cd /workspace"
echo "  tar -xzvf $PACKAGE_NAME"
echo "  cd usf-bios"
echo "  chmod +x deploy/*.sh"
echo "  ./deploy/setup-gpu-server.sh"
echo ""
