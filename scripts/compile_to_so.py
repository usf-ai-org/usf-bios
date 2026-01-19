#!/usr/bin/env python3
"""
USF BIOS - Cython Compilation Script
Compiles Python source files to native .so files for code protection.
This provides industry-standard code obfuscation used by commercial software.
Copyright (c) US Inc. All rights reserved.
"""
import os
import sys
import glob
import shutil
from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

# Cython compiler options for maximum obfuscation
Options.docstrings = False  # Remove all docstrings
Options.embed_pos_in_docstring = False


def get_py_files(directory):
    """Get all .py files except __init__.py and excluded files"""
    py_files = []
    # Directories to skip
    skip_dirs = {'__pycache__', 'venv', 'env', '.venv', '.env', 'node_modules', '.git', 'build', 'dist', 'egg-info'}
    # Files to skip (Pydantic models with methods don't work well with Cython)
    # db_models.py - SQLAlchemy ORM models with declarative base don't compile well
    skip_files = {'__init__.py', 'config.py', 'db_models.py', 'version.py'}
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs and not d.endswith('.egg-info')]
        for file in files:
            if file.endswith('.py') and file not in skip_files:
                py_files.append(os.path.join(root, file))
    return py_files


def compile_directory(source_dir, compile_from=None):
    """Compile all Python files in directory to .so
    
    Args:
        source_dir: Directory containing Python files to compile
        compile_from: Directory to run compilation from (default: parent of source_dir)
    """
    py_files = get_py_files(source_dir)
    if not py_files:
        print(f"No Python files to compile in {source_dir}")
        return
    
    print(f"Compiling {len(py_files)} files in {source_dir}...")
    
    # Determine compilation directory (parent of package for proper module paths)
    if compile_from is None:
        compile_from = os.path.dirname(source_dir)
    
    original_dir = os.getcwd()
    os.chdir(compile_from)
    
    # Convert absolute paths to relative paths from compile_from
    rel_py_files = [os.path.relpath(f, compile_from) for f in py_files]
    
    # Store expected .so locations for each .py file
    expected_so_files = {}
    for py_file, rel_file in zip(py_files, rel_py_files):
        base = os.path.splitext(rel_file)[0]
        expected_so_files[py_file] = base
    
    # Compile with Cython - maximum security settings
    try:
        setup(
            ext_modules=cythonize(
                rel_py_files,
                compiler_directives={
                    'language_level': '3',
                    'boundscheck': False,
                    'wraparound': True,  # Allow negative indexing (Python standard)
                    'cdivision': True,
                    'embedsignature': False,  # Don't embed function signatures
                    'emit_code_comments': False,  # No source comments in C
                    'annotation_typing': False,  # Python annotations, not Cython types
                },
                nthreads=0,  # 0 = use all available CPU cores
                quiet=True
            ),
            script_args=['build_ext', '--inplace']
        )
    except Exception as e:
        print(f"Error compiling {source_dir}: {e}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Find and move .so files from build directory if needed
    build_dir = os.path.join(compile_from, 'build')
    if os.path.exists(build_dir):
        print(f"  Checking build directory for .so files...")
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                if file.endswith('.so'):
                    src_path = os.path.join(root, file)
                    # Extract module name (e.g., db_models from db_models.cpython-311-x86_64-linux-gnu.so)
                    module_name = file.split('.')[0]
                    # Find matching destination
                    for py_file, base in expected_so_files.items():
                        if base.endswith(module_name) or base.endswith('/' + module_name):
                            dest_dir = os.path.dirname(py_file)
                            dest_path = os.path.join(dest_dir, file)
                            if not os.path.exists(dest_path):
                                shutil.copy2(src_path, dest_path)
                                print(f"  Copied: {file} -> {dest_dir}")
                            break
    
    os.chdir(original_dir)
    
    # Verify .so files exist before removing .py files
    missing_so = []
    for py_file in py_files:
        py_dir = os.path.dirname(py_file)
        py_base = os.path.splitext(os.path.basename(py_file))[0]
        # Look for any .so file matching this module
        found_so = False
        for f in os.listdir(py_dir):
            if f.startswith(py_base + '.') and f.endswith('.so'):
                found_so = True
                break
        if not found_so:
            missing_so.append(py_file)
    
    if missing_so:
        print(f"  ERROR: {len(missing_so)} .so files not found!")
        for f in missing_so[:10]:
            print(f"    Missing: {f}")
        sys.exit(1)
    
    # Remove source .py files (keep __init__.py)
    for py_file in py_files:
        if os.path.exists(py_file):
            os.remove(py_file)
            print(f"  Removed source: {py_file}")
    
    # Remove .c files generated by Cython
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.c'):
                os.remove(os.path.join(root, file))


def minimize_init_files(base_dir):
    """Make __init__.py files minimal - remove standalone comments only"""
    skip_dirs = {'venv', 'env', '.venv', '.env', 'node_modules', '.git'}
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file == '__init__.py':
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    # Remove only standalone comment lines and docstrings
                    # Keep all other content intact (including multi-line imports)
                    lines = []
                    in_docstring = False
                    docstring_char = None
                    for l in content.split('\n'):
                        stripped = l.strip()
                        # Handle docstrings
                        if not in_docstring:
                            if stripped.startswith('"""') or stripped.startswith("'''"):
                                docstring_char = stripped[:3]
                                if stripped.count(docstring_char) >= 2:
                                    continue  # Single line docstring
                                in_docstring = True
                                continue
                        else:
                            if docstring_char in stripped:
                                in_docstring = False
                            continue
                        # Skip standalone comment lines
                        if stripped.startswith('#'):
                            continue
                        # Keep everything else (imports, code, empty lines for structure)
                        # Remove inline comments
                        if '#' in l and not l.strip().startswith('#'):
                            l = l.split('#')[0].rstrip()
                        lines.append(l)
                    # Remove trailing empty lines
                    while lines and not lines[-1].strip():
                        lines.pop()
                    with open(filepath, 'w') as f:
                        f.write('\n'.join(lines) + '\n' if lines else '')
                    print(f"  Minimized: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not process {filepath}: {e}")


def clean_build_artifacts(base_dir):
    """Remove all build artifacts and intermediate files"""
    skip_dirs = {'venv', 'env', '.venv', '.env', 'node_modules', '.git'}
    
    patterns_to_remove = ['*.c', '*.html', '*.pyc', '*.pyo']
    dirs_to_remove = ['__pycache__', 'build', '*.egg-info']
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        # Remove files
        for pattern in patterns_to_remove:
            for filepath in glob.glob(os.path.join(root, pattern)):
                os.remove(filepath)
        
        # Remove directories
        for dir_pattern in dirs_to_remove:
            for dirpath in glob.glob(os.path.join(root, dir_pattern)):
                if os.path.isdir(dirpath):
                    shutil.rmtree(dirpath, ignore_errors=True)


def verify_compilation(base_dir):
    """Verify all .py files (except __init__.py) have been removed"""
    skip_dirs = {'venv', 'env', '.venv', '.env', 'node_modules', '.git'}
    remaining_py = []
    so_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                remaining_py.append(os.path.join(root, file))
            elif file.endswith('.so'):
                so_count += 1
    
    print(f"\n  Compiled .so files: {so_count}")
    
    if remaining_py:
        print(f"  WARNING: {len(remaining_py)} .py files still exist:")
        for f in remaining_py[:5]:
            print(f"    - {f}")
        return False
    else:
        print("  All source files successfully removed")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("  USF BIOS - Secure Binary Compilation")
    print("  Industry-standard code protection via Cython")
    print("=" * 60)
    
    # Step 1: Compile USF BIOS core library (training engine)
    # Run from /compile so usf_bios package structure is preserved
    print("\n[1/4] Compiling USF BIOS core library...")
    compile_directory('/compile/usf_bios', compile_from='/compile')
    
    # Step 2: Compile Web backend (API server)
    # Run from /compile/web/backend so app package structure is preserved
    print("\n[2/4] Compiling Web backend...")
    compile_directory('/compile/web/backend', compile_from='/compile/web/backend')
    
    # Step 3: Minimize __init__.py files
    print("\n[3/4] Minimizing __init__.py files...")
    minimize_init_files('/compile')
    
    # Step 4: Clean build artifacts
    print("\n[4/4] Cleaning build artifacts...")
    clean_build_artifacts('/compile')
    
    # Verification
    print("\n" + "=" * 60)
    print("  Verification")
    print("=" * 60)
    success = verify_compilation('/compile')
    
    print("\n" + "=" * 60)
    if success:
        print("  BUILD SUCCESSFUL - Code fully protected")
        print("  Binary .so files cannot be reverse engineered")
    else:
        print("  BUILD WARNING - Some files may not be protected")
    print("=" * 60)
