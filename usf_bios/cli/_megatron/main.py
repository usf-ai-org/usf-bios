# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
from typing import Dict

from usf_bios.utils import get_logger
from ..main import cli_main as usf_cli_main

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'usf_bios.cli._megatron.pt',
    'sft': 'usf_bios.cli._megatron.sft',
    'rlhf': 'usf_bios.cli._megatron.rlhf',
    'export': 'usf_bios.cli._megatron.export',
}


def cli_main():
    return usf_cli_main(ROUTE_MAPPING, is_megatron=True)


if __name__ == '__main__':
    cli_main()
