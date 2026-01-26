# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Core CLI dispatcher logic - This file gets compiled to .so
import importlib.util
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import json

from usf_bios.utils import get_logger
from usf_bios.system_guard import guard_cli_entry, check_system_valid, guard_with_integrity

logger = get_logger()

# Check system validity AND module integrity on module load
guard_with_integrity()

# Route mapping for standard CLI commands
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

# Route mapping for Megatron CLI commands
MEGATRON_ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'usf_bios.cli._megatron.pt',
    'sft': 'usf_bios.cli._megatron.sft',
    'rlhf': 'usf_bios.cli._megatron.rlhf',
    'export': 'usf_bios.cli._megatron.export',
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

            cfg = parse_dict_config(config)
            for key, value in cfg.items():
                argv.append(f'--{key}')
                if isinstance(value, list):
                    argv.extend(value)
                else:
                    argv.append(str(value))

            argv.pop(i)
            argv.pop(i)
            break


def _compat_web_ui(argv):
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
    
    module_name = route_mapping[method_name]
    spec = importlib.util.find_spec(module_name)
    file_path = spec.origin
    
    is_compiled = file_path and (file_path.endswith('.so') or file_path.endswith('.pyd'))
    
    torchrun_args = get_torchrun_args()
    prepare_config_args(argv)
    python_cmd = sys.executable
    
    if is_compiled:
        if torchrun_args is None or (not is_megatron and method_name not in {'pt', 'sft', 'rlhf', 'infer'}):
            args = [python_cmd, '-m', module_name, *argv]
        else:
            args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, '-m', module_name, *argv]
    else:
        if torchrun_args is None or (not is_megatron and method_name not in {'pt', 'sft', 'rlhf', 'infer'}):
            args = [python_cmd, file_path, *argv]
        else:
            args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    
    print(f"[USF BIOS] Running: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


def cli_main_standard() -> None:
    """Entry point for standard usf_bios CLI"""
    cli_main(ROUTE_MAPPING, is_megatron=False)


def cli_main_megatron() -> None:
    """Entry point for Megatron CLI"""
    cli_main(MEGATRON_ROUTE_MAPPING, is_megatron=True)


def try_init_unsloth():
    """Initialize unsloth if tuner_backend is set to unsloth"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuner_backend', type=str, default='peft')
    args, _ = parser.parse_known_args()
    if args.tuner_backend == 'unsloth':
        import unsloth


def sft_entry():
    """Entry point for SFT training - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    try_init_unsloth()
    from usf_bios.ray import try_init_ray
    try_init_ray()
    from usf_bios.pipelines import sft_main
    sft_main()


def merge_lora_entry():
    """Entry point for merge-lora command - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.arguments import ExportArguments
    from usf_bios.pipelines import USFPipeline, merge_lora
    
    class _MergeLoRA(USFPipeline):
        args_class = ExportArguments
        args: args_class
        
        def run(self):
            merge_lora(self.args)
    
    _MergeLoRA().main()


def pt_entry():
    """Entry point for pre-training - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from usf_bios.pipelines import pretrain_main
    pretrain_main()


def rlhf_entry():
    """Entry point for RLHF training - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from usf_bios.pipelines import rlhf_main
    rlhf_main()


def infer_entry():
    """Entry point for inference - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import infer_main
    infer_main()


def deploy_entry():
    """Entry point for deploy - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import deploy_main
    deploy_main()


def eval_entry():
    """Entry point for eval - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import eval_main
    eval_main()


def export_entry():
    """Entry point for export - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import export_main
    export_main()


def rollout_entry():
    """Entry point for rollout - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import rollout_main
    rollout_main()


def sample_entry():
    """Entry point for sample - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.ray import try_init_ray
    try_init_ray()
    from usf_bios.pipelines import sampling_main
    sampling_main()


def app_entry():
    """Entry point for app - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines import app_main
    app_main()


def web_ui_entry():
    """Entry point for web-ui - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.ui import webui_main
    webui_main()


def train_ui_entry():
    """Entry point for train-ui - guard is inside compiled code"""
    guard_with_integrity()
    from usf_bios.pipelines.webui import usf_omega_train_ui_main
    usf_omega_train_ui_main()


# Megatron entry points
def megatron_sft_entry():
    """Entry point for Megatron SFT - guard is inside compiled code"""
    guard_with_integrity()
    import os
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_sft_main
    megatron_sft_main()


def megatron_pt_entry():
    """Entry point for Megatron pre-training - guard is inside compiled code"""
    guard_with_integrity()
    import os
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_pretrain_main
    megatron_pretrain_main()


def megatron_rlhf_entry():
    """Entry point for Megatron RLHF - guard is inside compiled code"""
    guard_with_integrity()
    import os
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_rlhf_main
    megatron_rlhf_main()


def megatron_export_entry():
    """Entry point for Megatron export - guard is inside compiled code"""
    guard_with_integrity()
    import os
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_export_main
    megatron_export_main()
