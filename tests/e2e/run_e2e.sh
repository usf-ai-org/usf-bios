#!/bin/bash
# ============================================================
# USF BIOS - End-to-End Training Test Orchestrator
# Run this script on the Azure GPU server after git pull
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
API_URL="${API_URL:-http://localhost:8000/api}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
TIMEOUT="${TIMEOUT:-1800}"

echo "============================================================"
echo "USF BIOS - E2E Training Test Suite"
echo "============================================================"
echo "  Project Root: $PROJECT_ROOT"
echo "  Model:        $MODEL"
echo "  API URL:      $API_URL"
echo "  Samples:      $NUM_SAMPLES"
echo "  Timeout:      ${TIMEOUT}s per test"
echo "============================================================"

# Step 1: Prepare test datasets
echo ""
echo ">>> Step 1: Preparing test datasets..."
cd "$SCRIPT_DIR"
python prepare_test_datasets.py \
    --output-dir "$SCRIPT_DIR/test_data" \
    --num-samples "$NUM_SAMPLES"

# Step 2: Check if backend is running, start if not
echo ""
echo ">>> Step 2: Checking backend..."
if curl -s "$API_URL/system/status" > /dev/null 2>&1; then
    echo "  Backend is already running at $API_URL"
else
    echo "  Starting backend..."
    cd "$PROJECT_ROOT/web/backend"
    nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/usf_bios_backend.log 2>&1 &
    echo "  Waiting for backend to start..."
    sleep 10
    
    if curl -s "$API_URL/system/status" > /dev/null 2>&1; then
        echo "  ✓ Backend started successfully"
    else
        echo "  ✗ Backend failed to start. Check /tmp/usf_bios_backend.log"
        exit 1
    fi
fi

# Step 3: Run all training tests
echo ""
echo ">>> Step 3: Running all training tests..."
cd "$SCRIPT_DIR"
python run_all_training_tests.py \
    --api-url "$API_URL" \
    --model "$MODEL" \
    --data-dir "$SCRIPT_DIR/test_data" \
    --output-dir "$SCRIPT_DIR/test_outputs" \
    --timeout "$TIMEOUT" \
    "$@"

echo ""
echo ">>> Test results saved to: $SCRIPT_DIR/test_outputs/test_results.json"
echo ">>> Done!"
