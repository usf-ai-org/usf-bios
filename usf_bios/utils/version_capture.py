#!/usr/bin/env python3
# Copyright (c) US Inc. All rights reserved.
# USF BIOS - Comprehensive Version Capture Module
# Captures ALL installed package versions during Docker build.

import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path


def run_cmd(cmd, shell=True):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=60)
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def get_system_info():
    """Get system/OS information."""
    info = {}
    info['hostname'] = run_cmd('hostname')
    info['kernel'] = run_cmd('uname -r')
    info['os_release'] = run_cmd('cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION|ID)=" | head -5')
    info['architecture'] = run_cmd('uname -m')
    info['cpu_info'] = run_cmd('cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2')
    info['memory_total'] = run_cmd("free -h | grep Mem | awk '{print $2}'")
    return info


def get_linux_packages():
    """Get all installed Linux/apt packages with versions."""
    packages = {}
    output = run_cmd("dpkg-query -W -f='${Package}=${Version}\n' 2>/dev/null")
    if output and not output.startswith('ERROR'):
        for line in output.split('\n'):
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    packages[parts[0]] = parts[1]
    return packages


def get_pip_packages():
    """Get all installed pip packages with exact versions."""
    packages = {}
    output = run_cmd(f"{sys.executable} -m pip freeze 2>/dev/null")
    if output and not output.startswith('ERROR'):
        for line in output.split('\n'):
            line = line.strip()
            if '==' in line:
                parts = line.split('==', 1)
                if len(parts) == 2:
                    packages[parts[0]] = parts[1]
            elif ' @ ' in line:
                parts = line.split(' @ ', 1)
                if len(parts) == 2:
                    packages[parts[0]] = f"URL: {parts[1][:100]}..."
    return packages


def get_pip_packages_detailed():
    """Get detailed pip package info including location."""
    packages = {}
    output = run_cmd(f"{sys.executable} -m pip list --format=json 2>/dev/null")
    if output and not output.startswith('ERROR'):
        try:
            pkg_list = json.loads(output)
            for pkg in pkg_list:
                packages[pkg['name']] = pkg['version']
        except json.JSONDecodeError:
            pass
    return packages


def get_cuda_info():
    """Get CUDA and GPU information."""
    info = {}
    
    nvcc_output = run_cmd('nvcc --version 2>/dev/null')
    if nvcc_output and 'release' in nvcc_output.lower():
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                info['nvcc_version'] = line.strip()
                break
    
    smi_output = run_cmd('nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader 2>/dev/null')
    if smi_output and not smi_output.startswith('ERROR'):
        info['nvidia_smi'] = smi_output.strip()
    
    gpu_info = run_cmd('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null')
    if gpu_info and not gpu_info.startswith('ERROR'):
        info['gpu_info'] = gpu_info.strip()
    
    cuda_libs = run_cmd('ls -la /usr/local/cuda/lib64/*.so* 2>/dev/null | head -20')
    if cuda_libs:
        info['cuda_libs_sample'] = cuda_libs
    
    cudnn_header = run_cmd('cat /usr/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 | head -3')
    if cudnn_header:
        info['cudnn_header'] = cudnn_header
    
    return info


def get_python_info():
    """Get Python environment information."""
    info = {}
    info['python_version'] = sys.version
    info['python_executable'] = sys.executable
    info['python_path'] = ':'.join(sys.path[:5])
    info['pip_version'] = run_cmd(f"{sys.executable} -m pip --version")
    return info


def get_node_info():
    """Get Node.js information."""
    info = {}
    info['node_version'] = run_cmd('node --version 2>/dev/null')
    info['npm_version'] = run_cmd('npm --version 2>/dev/null')
    info['npx_version'] = run_cmd('npx --version 2>/dev/null')
    return info


def get_build_tools():
    """Get build tools versions."""
    info = {}
    info['gcc_version'] = run_cmd('gcc --version 2>/dev/null | head -1')
    info['g++_version'] = run_cmd('g++ --version 2>/dev/null | head -1')
    info['cmake_version'] = run_cmd('cmake --version 2>/dev/null | head -1')
    info['ninja_version'] = run_cmd('ninja --version 2>/dev/null')
    info['make_version'] = run_cmd('make --version 2>/dev/null | head -1')
    info['git_version'] = run_cmd('git --version 2>/dev/null')
    return info


def get_ml_framework_details():
    """Get detailed ML framework information."""
    info = {}
    
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['torch_cuda_available'] = str(torch.cuda.is_available())
        info['torch_cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
        info['torch_cudnn_version'] = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else 'N/A'
        info['torch_device_count'] = str(torch.cuda.device_count()) if torch.cuda.is_available() else '0'
    except ImportError:
        info['torch_version'] = 'NOT INSTALLED'
    
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        info['transformers_version'] = 'NOT INSTALLED'
    
    try:
        import peft
        info['peft_version'] = peft.__version__
    except ImportError:
        info['peft_version'] = 'NOT INSTALLED'
    
    try:
        import trl
        info['trl_version'] = trl.__version__
    except ImportError:
        info['trl_version'] = 'NOT INSTALLED'
    
    try:
        import accelerate
        info['accelerate_version'] = accelerate.__version__
    except ImportError:
        info['accelerate_version'] = 'NOT INSTALLED'
    
    try:
        import deepspeed
        info['deepspeed_version'] = deepspeed.__version__
    except ImportError:
        info['deepspeed_version'] = 'NOT INSTALLED'
    
    try:
        import flash_attn
        info['flash_attn_version'] = flash_attn.__version__
    except ImportError:
        info['flash_attn_version'] = 'NOT INSTALLED'
    
    try:
        import xformers
        info['xformers_version'] = xformers.__version__
    except ImportError:
        info['xformers_version'] = 'NOT INSTALLED'
    
    try:
        import bitsandbytes
        info['bitsandbytes_version'] = bitsandbytes.__version__
    except ImportError:
        info['bitsandbytes_version'] = 'NOT INSTALLED'
    
    try:
        import datasets
        info['datasets_version'] = datasets.__version__
    except ImportError:
        info['datasets_version'] = 'NOT INSTALLED'
    
    try:
        import huggingface_hub
        info['huggingface_hub_version'] = huggingface_hub.__version__
    except ImportError:
        info['huggingface_hub_version'] = 'NOT INSTALLED'
    
    try:
        import tokenizers
        info['tokenizers_version'] = tokenizers.__version__
    except ImportError:
        info['tokenizers_version'] = 'NOT INSTALLED'
    
    try:
        import safetensors
        info['safetensors_version'] = safetensors.__version__
    except ImportError:
        info['safetensors_version'] = 'NOT INSTALLED'
    
    try:
        import triton
        info['triton_version'] = triton.__version__
    except ImportError:
        info['triton_version'] = 'NOT INSTALLED'
    
    try:
        import numpy
        info['numpy_version'] = numpy.__version__
    except ImportError:
        info['numpy_version'] = 'NOT INSTALLED'
    
    try:
        import PIL
        info['pillow_version'] = PIL.__version__
    except ImportError:
        info['pillow_version'] = 'NOT INSTALLED'
    
    return info


def get_usf_bios_info():
    """Get USF BIOS package information."""
    info = {}
    # Avoid importing usf_bios to prevent circular imports
    # Instead, read version directly from version.py file
    version_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'version.py'),
        '/app/core/usf_bios/version.py',
        '/app/usf_bios/version.py',
    ]
    for vpath in version_paths:
        if os.path.exists(vpath):
            try:
                with open(vpath, 'r') as f:
                    content = f.read()
                # Extract __version__ = 'x.x.x'
                for line in content.split('\n'):
                    if line.strip().startswith('__version__'):
                        version = line.split('=')[1].strip().strip("'\"")
                        info['usf_bios_version'] = version
                        return info
            except Exception:
                pass
    # Fallback: try pip show
    pip_output = run_cmd(f"{sys.executable} -m pip show usf-bios 2>/dev/null | grep Version")
    if pip_output and 'Version:' in pip_output:
        info['usf_bios_version'] = pip_output.split(':')[1].strip()
    else:
        info['usf_bios_version'] = 'NOT INSTALLED'
    return info


def generate_report(output_path=None):
    """Generate comprehensive version report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = {
        'generated_at': timestamp,
        'system': get_system_info(),
        'python': get_python_info(),
        'cuda': get_cuda_info(),
        'node': get_node_info(),
        'build_tools': get_build_tools(),
        'ml_frameworks': get_ml_framework_details(),
        'usf_bios': get_usf_bios_info(),
        'pip_packages': get_pip_packages_detailed(),
        'linux_packages': get_linux_packages(),
    }
    
    if output_path:
        json_path = Path(output_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"JSON report saved to: {json_path}")
        
        txt_path = json_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"USF BIOS - Complete Environment Version Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"{'='*80}\n")
            f.write("SYSTEM INFORMATION\n")
            f.write(f"{'='*80}\n")
            for k, v in report['system'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("PYTHON ENVIRONMENT\n")
            f.write(f"{'='*80}\n")
            for k, v in report['python'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("CUDA / GPU INFORMATION\n")
            f.write(f"{'='*80}\n")
            for k, v in report['cuda'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("NODE.JS ENVIRONMENT\n")
            f.write(f"{'='*80}\n")
            for k, v in report['node'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("BUILD TOOLS\n")
            f.write(f"{'='*80}\n")
            for k, v in report['build_tools'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("ML FRAMEWORKS (DETAILED)\n")
            f.write(f"{'='*80}\n")
            for k, v in report['ml_frameworks'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write("USF BIOS PACKAGE\n")
            f.write(f"{'='*80}\n")
            for k, v in report['usf_bios'].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"ALL PIP PACKAGES ({len(report['pip_packages'])} total)\n")
            f.write(f"{'='*80}\n")
            for pkg in sorted(report['pip_packages'].keys()):
                f.write(f"{pkg}=={report['pip_packages'][pkg]}\n")
            f.write("\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"ALL LINUX/APT PACKAGES ({len(report['linux_packages'])} total)\n")
            f.write(f"{'='*80}\n")
            for pkg in sorted(report['linux_packages'].keys()):
                f.write(f"{pkg}={report['linux_packages'][pkg]}\n")
        
        print(f"Text report saved to: {txt_path}")
    
    return report


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Capture all installed package versions')
    parser.add_argument('-o', '--output', default='/app/data/version_report.json',
                        help='Output file path for JSON report')
    parser.add_argument('--print', action='store_true', help='Print report to stdout')
    args = parser.parse_args()
    
    report = generate_report(args.output)
    
    if args.print:
        print(json.dumps(report, indent=2, default=str))
    
    print("\n" + "="*60)
    print("VERSION CAPTURE SUMMARY")
    print("="*60)
    print(f"Total PIP packages: {len(report['pip_packages'])}")
    print(f"Total Linux packages: {len(report['linux_packages'])}")
    print(f"PyTorch: {report['ml_frameworks'].get('torch_version', 'N/A')}")
    print(f"Transformers: {report['ml_frameworks'].get('transformers_version', 'N/A')}")
    print(f"CUDA: {report['ml_frameworks'].get('torch_cuda_version', 'N/A')}")
    print(f"USF BIOS: {report['usf_bios'].get('usf_bios_version', 'N/A')}")
    print("="*60)
