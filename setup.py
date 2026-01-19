# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Powered by US Inc
# !/usr/bin/env python
import os
import subprocess
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from typing import List

# Custom transformers fork URL (required for UltraSafe models like USF Omega)
CUSTOM_TRANSFORMERS_URL = "git+https://github.com/apt-team-018/transformers.git"


def install_custom_transformers():
    """Uninstall standard transformers and install custom fork."""
    print("\n" + "=" * 60)
    print("  USF BIOS - Installing Custom Transformers Fork")
    print("  Required for UltraSafe model support (USF Omega, etc.)")
    print("=" * 60 + "\n")
    try:
        # Step 1: Uninstall any existing transformers
        print("  [1/2] Removing standard transformers...")
        subprocess.call([
            sys.executable, "-m", "pip", "uninstall", "-y", "transformers"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Step 2: Install custom fork
        print("  [2/2] Installing custom transformers fork...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            CUSTOM_TRANSFORMERS_URL
        ])
        print("\n✓ Custom transformers fork installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠ Warning: Could not install custom transformers: {e}")
        print("  You may need to manually run:")
        print(f"  pip install {CUSTOM_TRANSFORMERS_URL}\n")


class PostInstallCommand(install):
    """Post-installation: install custom transformers fork."""
    def run(self):
        install.run(self)
        install_custom_transformers()


class PostDevelopCommand(develop):
    """Post-develop installation: install custom transformers fork."""
    def run(self):
        develop.run(self)
        install_custom_transformers()


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'usf_bios/version.py'


def get_version():
    version_vars = {}
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), version_vars)
    return version_vars['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        # Skip git URLs - they will be handled by post-install hook
        if line.startswith('git+') or 'github.com' in line:
            return
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            relative_base = os.path.dirname(fname)
            absolute_target = os.path.join(relative_base, target)
            for info in parse_require_file(absolute_target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('http'):
                    print('skip http requirements %s' % line)
                    continue
                if line and not line.startswith('#') and not line.startswith('--'):
                    for info in parse_line(line):
                        yield info
                elif line and line.startswith('--find-links'):
                    eles = line.split()
                    for e in eles:
                        e = e.strip()
                        if 'http' in e:
                            info = dict(dependency_links=e)
                            yield info

    def gen_packages_items():
        items = []
        deps_link = []
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                if 'dependency_links' not in info:
                    parts = [info['package']]
                    if with_version and 'version' in info:
                        parts.extend(info['version'])
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            parts.append(';' + platform_deps)
                    item = ''.join(parts)
                    items.append(item)
                else:
                    deps_link.append(info['dependency_links'])
        return items, deps_link

    return gen_packages_items()


install_requires, deps_link = parse_requirements('requirements.txt')
extra_requires = {}
all_requires = []
extra_requires['eval'], _ = parse_requirements('requirements/eval.txt')
extra_requires['swanlab'], _ = parse_requirements('requirements/swanlab.txt')
extra_requires['ray'], _ = parse_requirements('requirements/ray.txt')
all_requires.extend(install_requires)
all_requires.extend(extra_requires['eval'])
all_requires.extend(extra_requires['swanlab'])
all_requires.extend(extra_requires['ray'])
extra_requires['all'] = all_requires

setup(
    name='usf_bios',
    version=get_version(),
    description='USF BIOS: AI Training & Fine-tuning Platform - Powered by US Inc',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='US Inc',
    author_email='support@us.inc',
    keywords=['transformers', 'LLM', 'lora', 'megatron', 'grpo', 'sft', 'usf-bios', 'us-inc'],
    url='https://us.inc',
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    package_data={
        '': ['utils/*', 'llm/dataset/data/*.*', 'llm/ds_config/*.json', 'plugin/loss_scale/config/*.json']
    },
    python_requires='>=3.8.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    license='Apache License 2.0',
    tests_require=parse_requirements('requirements/tests.txt'),
    install_requires=install_requires,
    extras_require=extra_requires,
    entry_points={
        'console_scripts': ['usf_bios=usf_bios.cli.main:cli_main', 'usf_bios_megatron=usf_bios.cli._megatron.main:cli_main']
    },
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    dependency_links=deps_link,
    zip_safe=False)
