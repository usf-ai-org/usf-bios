# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Dict, Type

import gradio as gr

from usf_bios.arguments import get_supported_tuners
from usf_bios.utils import get_device_count, get_logger
from ..base import BaseUI
from ..llm_train import LLMTrain
from .advanced import RLHFAdvanced
from .dataset import RLHFDataset
from .hyper import RLHFHyper
from .model import RLHFModel
from .optimizer import RLHFOptimizer
from .quantization import RLHFQuantization
from .report_to import RLHFReportTo
from .rlhf import RLHF
from .runtime import RLHFRuntime
from .save import RLHFSave
from .tuner import RLHFTuner

logger = get_logger()


class LLMRLHF(LLMTrain):
    group = 'llm_rlhf'

    sub_ui = [
        RLHFModel,
        RLHFDataset,
        RLHFHyper,
        RLHFRuntime,
        RLHFTuner,
        RLHFOptimizer,
        RLHF,
        RLHFQuantization,
        RLHFSave,
        RLHFReportTo,
        RLHFAdvanced,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_rlhf': {
            'label': {
                'en': 'LLM RLHF',
            }
        },
        'train_stage': {
            'label': {
                'en': 'Train Stage'
            },
            'info': {
                'en': 'Please choose matched dataset'
            }
        },
        'submit_alert': {
            'value': {
                'en': 'Task started, please check the tensorboard or log file, '
                'do not close the terminal, otherwise the training process will be interrupted'
            }
        },
        'dataset_alert': {
            'value': {
                'en': 'Please input or select a dataset'
            }
        },
        'submit': {
            'value': {
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'en': 'Dry-run'
            },
            'info': {
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'en': 'Choose GPU'
            },
            'info': {
                'en': 'Select GPU to train'
            }
        },
        'rlhf_type': {
            'label': {
                'en': 'RLHF type'
            },
        },
        'train_type': {
            'label': {
                'en': 'Train type'
            },
            'info': {
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'en': 'Seed'
            },
            'info': {
                'en': 'Select a random seed'
            }
        },
        'torch_dtype': {
            'label': {
                'en': 'Training Precision'
            },
            'info': {
                'en': 'Select the training precision'
            }
        },
        'envs': {
            'label': {
                'en': 'Extra env vars'
            },
        },
        'use_ddp': {
            'label': {
                'en': 'Use DDP'
            },
            'info': {
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'ddp_num': {
            'label': {
                'en': 'Number of DDP sharding'
            },
            'info': {
                'en': 'The data parallel size of DDP'
            }
        },
        'use_liger_kernel': {
            'label': {
                'en': 'Use Liger kernel'
            },
            'info': {
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'en': 'Sequence parallel size',
            },
            'info': {
                'en': 'Currently supports CPT/SFT/DPO/GRPO',
            }
        },
        'deepspeed': {
            'label': {
                'en': 'DeepSpeed',
            },
            'info': {
                'en': 'Choose from the dropbox or fill in a valid path',
            }
        },
        'resume_checkpoint_alert': {
            'value': {
                'en': 'Detected that "args.json" is in {}, will start breakpoint resume training from this checkpoint'
            }
        },
        'resume_only_model_alert': {
            'value': {
                'en': '"args.json" is detected in {}, but optimizer parameters are not detected. '
                'Only model parameters will be loaded to start breakpoint continuation training'
            }
        },
        'more_params': {
            'label': {
                'en': 'Other params'
            },
            'info': {
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        },
        'extra_params': {
            'label': {
                'en': 'Extra settings'
            },
        },
        'train_param': {
            'label': {
                'en': 'Train settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_rlhf', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                RLHFModel.build_ui(base_tab)
                RLHFDataset.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='rlhf_type', scale=2)
                        gr.Dropdown(elem_id='train_type', scale=2, choices=list(get_supported_tuners()))
                        gr.Textbox(elem_id='seed', scale=2)
                        gr.Dropdown(elem_id='torch_dtype', scale=2)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=2)
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='gpu_id',
                            multiselect=True,
                            choices=[str(i) for i in range(device_count)] + ['cpu'],
                            value=default_device,
                            scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                        gr.Dropdown(
                            elem_id='deepspeed',
                            scale=4,
                            allow_custom_value=True,
                            value=None,
                            choices=['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'])
                        gr.Textbox(elem_id='sequence_parallel_size', lines=1, scale=4)
                RLHFHyper.build_ui(base_tab)
                RLHFRuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Textbox(elem_id='envs', scale=12)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                RLHFTuner.build_ui(base_tab)
                RLHFOptimizer.build_ui(base_tab)
                RLHF.build_ui(base_tab)
                with gr.Accordion(elem_id='extra_params', open=False):
                    with gr.Tabs():
                        RLHFAdvanced.build_ui(base_tab)
                        RLHFQuantization.build_ui(base_tab)
                        RLHFSave.build_ui(base_tab)
                        RLHFReportTo.build_ui(base_tab)
                    with gr.Row():
                        gr.Textbox(elem_id='more_params', lines=4, scale=20)

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                cls.element('train_type').change(
                    RLHFHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])
                cls.element('rlhf_type').change(
                    RLHF.update_beta, inputs=[base_tab.element('rlhf_type')], outputs=[base_tab.element('beta')])

                submit.click(
                    cls.train_local,
                    list(cls.valid_elements().values()), [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab'),
                        cls.element('running_tasks'),
                        cls.element('train_record'),
                    ],
                    queue=True)

                base_tab.element('running_tasks').change(
                    partial(RLHFRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + RLHFRuntime.all_plots)
                RLHFRuntime.element('kill_task').click(
                    RLHFRuntime.kill_task,
                    [RLHFRuntime.element('running_tasks')],
                    [RLHFRuntime.element('running_tasks')] + [RLHFRuntime.element('log')] + RLHFRuntime.all_plots,
                ).then(RLHFRuntime.reset, [], [RLHFRuntime.element('logging_dir')] + [RLHFHyper.element('output_dir')])

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'optimizer'], [RLHFTuner.tabs_to_filter, RLHFOptimizer.tabs_to_filter])
        }
        return tabs_relation_dict

    @classmethod
    def filter_rlhf_args(cls, uncleaned_kwargs):
        cur_rlhf_type = uncleaned_kwargs.get('rlhf_type', 'dpo')
        cur_selected = RLHF.rlhf_args_dict.pop(cur_rlhf_type, None)
        for _, vals in RLHF.rlhf_args_dict.items():
            for rlhf_arg in vals:
                if uncleaned_kwargs.get(rlhf_arg) and (cur_selected is None or rlhf_arg not in cur_selected):
                    uncleaned_kwargs.pop(rlhf_arg)
