# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
from typing import TYPE_CHECKING

from usf_bios.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .infer import (infer_main, deploy_main, run_deploy, rollout_main)
    from .export import (export_main, merge_lora, quantize_model, export_to_ollama)
    from .eval import eval_main
    from .app import app_main
    from .train import sft_main, pretrain_main, rlhf_main, USFSft
    from .sampling import sampling_main
    from .base import USFPipeline
    from .utils import prepare_model_template
    from .webui import train_ui_main, build_train_ui, build_usf_omega_train_ui, usf_omega_train_ui_main
else:
    _import_structure = {
        'infer': [
            'deploy_main',
            'infer_main',
            'run_deploy',
            'rollout_main',
        ],
        'export': ['export_main', 'merge_lora', 'quantize_model', 'export_to_ollama'],
        'app': ['app_main'],
        'eval': ['eval_main'],
        'train': ['sft_main', 'pretrain_main', 'rlhf_main', 'USFSft'],
        'sampling': ['sampling_main'],
        'base': ['USFPipeline'],
        'utils': ['prepare_model_template'],
        'webui': ['train_ui_main', 'build_train_ui', 'build_usf_omega_train_ui', 'usf_omega_train_ui_main'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
