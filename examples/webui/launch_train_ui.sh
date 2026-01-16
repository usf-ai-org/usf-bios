#!/bin/bash
# Launch USF BIOS Training WebUI
# Usage: ./launch_train_ui.sh [options]
#
# Options:
#   --port PORT     Server port (default: 7861)
#   --lang LANG     Language: en or zh (default: en)
#   --share         Create public Gradio link

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Default values
PORT=7861
LANG="en"
SHARE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --lang)
            LANG="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Launching USF BIOS Training WebUI..."
echo "   Port: $PORT"
echo "   Language: $LANG"
echo "   Project Root: $PROJECT_ROOT"
echo ""

PYTHONPATH="$PROJECT_ROOT" python usf_bios/cli/train_ui.py \
    --server_port "$PORT" \
    --lang "$LANG" \
    $SHARE
