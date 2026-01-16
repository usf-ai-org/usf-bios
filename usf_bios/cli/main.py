# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Powered by US Inc
import importlib.util
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import json

from usf_bios.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'usf_bios.cli.pt',
    'sft': 'usf_bios.cli.sft',
    'infer': 'usf_bios.cli.infer',
    'merge-lora': 'usf_bios.cli.merge_lora',
    'web-ui': 'usf_bios.cli.web_ui',
    'deploy': 'usf_bios.cli.deploy',
    'rollout': 'usf_bios.cli.rollout',
    'rlhf': 'usf_bios.cli.rlhf',
    'sample': 'usf_bios.cli.sample',
    'export': 'usf_bios.cli.export',
    'eval': 'usf_bios.cli.eval',
    'app': 'usf_bios.cli.app',
    'train-ui': 'usf_bios.cli.train_ui',
}


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def prepare_config_args(argv):
    for i in range(len(argv)):
        if argv[i] == '--config':
            if i + 1 >= len(argv):
                raise ValueError('The `--config` argument requires a yaml file path.')
            from omegaconf import OmegaConf, DictConfig, ListConfig
            config = OmegaConf.load(argv[i + 1])

            def parse_dict_config(cfg: DictConfig) -> Dict[str, Any]:
                result = {}
                for key, value in cfg.items():
                    if isinstance(value, DictConfig):
                        result[key] = json.dumps(OmegaConf.to_container(value))
                    elif isinstance(value, ListConfig):
                        result[key] = list(value)
                    else:
                        result[key] = value
                return result

            # Convert yaml to cmd line
            cfg = parse_dict_config(config)
            for key, value in cfg.items():
                argv.append(f'--{key}')
                if isinstance(value, list):
                    argv.extend(value)
                else:
                    argv.append(str(value))

            # Pop --config
            argv.pop(i)
            # Pop value of --config
            argv.pop(i)
            break


def _compat_web_ui(argv):
    # [compat]
    method_name = argv[0]
    if method_name in {'web-ui', 'web_ui'} and ('--model' in argv or '--adapters' in argv or '--ckpt_dir' in argv):
        argv[0] = 'app'
        logger.warning('Please use `usf app`.')


def cli_main(route_mapping: Optional[Dict[str, str]] = None, is_megatron: bool = False) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    _compat_web_ui(argv)
    method_name = argv[0].replace('_', '-')
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    torchrun_args = get_torchrun_args()
    prepare_config_args(argv)
    python_cmd = sys.executable
    if torchrun_args is None or (not is_megatron and method_name not in {'pt', 'sft', 'rlhf', 'infer'}):
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"[USF BIOS] Running: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
