#!/usr/bin/env python3
# Copyright (c) US Inc. All rights reserved.
# USF BIOS - Version Capture Script Entry Point
# This script imports directly to avoid loading heavy dependencies

import os
import sys
import glob
import importlib.util

def find_module_file(base_dirs, module_name):
    """Find module file (.py or .so) in given directories"""
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        # Check for .py file first
        py_path = os.path.join(base_dir, f'{module_name}.py')
        if os.path.exists(py_path):
            return py_path
        # Check for compiled .so file (pattern: module_name.cpython-*.so)
        so_pattern = os.path.join(base_dir, f'{module_name}.cpython-*.so')
        so_files = glob.glob(so_pattern)
        if so_files:
            return so_files[0]
        # Check for simple .so file
        so_path = os.path.join(base_dir, f'{module_name}.so')
        if os.path.exists(so_path):
            return so_path
    return None

def load_module_directly():
    """Load version_capture module directly without going through package __init__.py"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Possible directories where module could be
    search_dirs = [
        os.path.join(project_root, 'usf_bios', 'utils'),
        '/app/core/usf_bios/utils',
        '/app/usf_bios/utils',
    ]
    
    module_path = find_module_file(search_dirs, 'version_capture')
    
    if not module_path:
        print("ERROR: Cannot find version_capture module (.py or .so)")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location('version_capture', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    vc = load_module_directly()
    vc.main()
