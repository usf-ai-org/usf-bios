#!/bin/bash
# USF BIOS - GPU Metrics Test Script
# Run this on the GPU server to verify metrics are working

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "USF BIOS - GPU Metrics Test"
echo "=========================================="

# Test 1: Check if NVIDIA driver is available
echo -e "\n${YELLOW}[Test 1] NVIDIA Driver...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv
    echo -e "${GREEN}✓ NVIDIA driver detected${NC}"
else
    echo -e "${RED}✗ NVIDIA driver not found${NC}"
fi

# Test 2: Check if pynvml works in Python
echo -e "\n${YELLOW}[Test 2] Python NVML Library...${NC}"
python3 << 'PYEOF'
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"✓ pynvml working - Found {device_count} GPU(s)")
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        print(f"  GPU {i}: {name}")
        print(f"    Memory: {mem_info.used / 1024**3:.1f} / {mem_info.total / 1024**3:.1f} GB")
        print(f"    Temperature: {temp}°C")
        print(f"    Utilization: {util.gpu}%")
    
    pynvml.nvmlShutdown()
except ImportError:
    print("✗ pynvml not installed. Run: pip install pynvml")
except Exception as e:
    print(f"✗ pynvml error: {e}")
PYEOF

# Test 3: Check if PyTorch sees GPU
echo -e "\n${YELLOW}[Test 3] PyTorch CUDA...${NC}"
python3 << 'PYEOF'
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    Total Memory: {mem:.1f} GB")
    else:
        print("✗ PyTorch CUDA not available")
except ImportError:
    print("✗ PyTorch not installed")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
PYEOF

# Test 4: Test the API endpoint
echo -e "\n${YELLOW}[Test 4] Backend API Metrics Endpoint...${NC}"
if curl -s http://localhost:8000/api/system/metrics > /tmp/metrics.json 2>/dev/null; then
    echo -e "${GREEN}✓ API endpoint responding${NC}"
    echo "Response:"
    python3 -m json.tool /tmp/metrics.json 2>/dev/null || cat /tmp/metrics.json
else
    echo -e "${RED}✗ API not responding - Is backend running?${NC}"
    echo "  Start with: ./start-services.sh"
fi

# Test 5: Check CPU/RAM metrics
echo -e "\n${YELLOW}[Test 5] CPU/RAM Metrics...${NC}"
python3 << 'PYEOF'
try:
    import psutil
    print(f"✓ psutil working")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1)}%")
    mem = psutil.virtual_memory()
    print(f"  RAM: {mem.used / 1024**3:.1f} / {mem.total / 1024**3:.1f} GB ({mem.percent}%)")
except ImportError:
    print("✗ psutil not installed. Run: pip install psutil")
except Exception as e:
    print(f"✗ psutil error: {e}")
PYEOF

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
