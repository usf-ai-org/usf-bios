#!/bin/bash
# USF BIOS - GPU Server Setup Script
# Run this on your RunPod/GPU server after uploading the code

set -e

echo "=========================================="
echo "USF BIOS - GPU Server Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"

# Step 1: System packages
echo -e "\n${GREEN}[1/6] Installing system dependencies...${NC}"
apt-get update && apt-get install -y curl nodejs npm python3-pip

# Step 2: Install Node.js 18+ if needed
echo -e "\n${GREEN}[2/6] Checking Node.js version...${NC}"
if ! command -v node &> /dev/null || [[ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 18 ]]; then
    echo "Installing Node.js 18..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
fi
echo "Node.js version: $(node -v)"

# Step 3: Install Python dependencies
echo -e "\n${GREEN}[3/6] Installing Python dependencies...${NC}"
cd "$PROJECT_ROOT/web/backend"
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || pip install \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    aiofiles \
    websockets \
    pynvml \
    psutil \
    torch \
    transformers \
    datasets \
    peft \
    trl \
    accelerate \
    bitsandbytes

# Step 4: Install Frontend dependencies
echo -e "\n${GREEN}[4/6] Installing Frontend dependencies...${NC}"
cd "$PROJECT_ROOT/web/frontend"
npm install

# Step 5: Build Frontend for production
echo -e "\n${GREEN}[5/6] Building Frontend...${NC}"
npm run build

# Step 6: Create startup script
echo -e "\n${GREEN}[6/6] Creating startup scripts...${NC}"

cat > "$PROJECT_ROOT/start-services.sh" << 'EOF'
#!/bin/bash
# Start both backend and frontend services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill existing processes
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true

echo "Starting Backend on port 8000..."
cd "$SCRIPT_DIR/web/backend"
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /var/log/usf-backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Starting Frontend on port 3000..."
cd "$SCRIPT_DIR/web/frontend"
nohup npm start > /var/log/usf-frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

sleep 3

echo ""
echo "=========================================="
echo "Services Started!"
echo "=========================================="
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To check logs:"
echo "  Backend:  tail -f /var/log/usf-backend.log"
echo "  Frontend: tail -f /var/log/usf-frontend.log"
echo ""
echo "To stop: pkill -f uvicorn && pkill -f 'next start'"
echo "=========================================="
EOF

chmod +x "$PROJECT_ROOT/start-services.sh"

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start services, run:"
echo "  cd $PROJECT_ROOT && ./start-services.sh"
echo ""
echo "To test GPU metrics:"
echo "  curl http://localhost:8000/api/system/metrics"
echo ""
