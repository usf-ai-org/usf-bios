# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Powered by US Inc
"""
Entry point for running usf_bios as a module.

Usage:
    python -m usf_bios sft --model ... --dataset ...
    python -m usf_bios infer --model ...
    python -m usf_bios deploy --model ...
"""
from usf_bios.cli.main import cli_main

if __name__ == '__main__':
    cli_main()
