# Copyright (c) US Inc. All rights reserved.
from typing import Dict, Type

import gradio as gr

from usf_bios.arguments import get_supported_tuners
from usf_bios.utils import get_device_count, get_logger
from ..base import BaseUI
from ..llm_train import LLMTrain
from .advanced import GRPOAdvanced
from .dataset import GRPODataset
from .external_rollout import LLMRollout
from .grpo_advanced import GrpoAdvanced
from .hyper import GRPOHyper
from .model import GRPOModel
from .optimizer import GRPOOptimizer
from .quantization import GRPOQuantization
from .report_to import GRPOReportTo
from .reward import Reward
from .rollout import Rollout
from .runtime import GRPORuntime
from .save import GRPOSave
from .tuner import GRPOTuner

logger = get_logger()


class LLMGRPO(LLMTrain):
    group = 'llm_grpo'

    sub_ui = [
        GRPOModel, GRPODataset, Reward, GRPORuntime, Rollout, GRPOSave, GRPOTuner, GRPOOptimizer, GRPOHyper,
        GRPOQuantization, GRPOAdvanced, GrpoAdvanced, GRPOReportTo, LLMRollout
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_grpo': {
            'label': {
                'en': 'LLM GRPO',
            }
        },
        'external_alert': {
            'value': {
                'en': 'Err: {} \nRollout model deployment is incomplete, '
                'please check the logs and start training later!'
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
        with gr.TabItem(elem_id='llm_grpo', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                GRPOModel.build_ui(base_tab)
                GRPODataset.build_ui(base_tab)
                Reward.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_type', scale=4, choices=list(get_supported_tuners()))
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                        gr.Textbox(elem_id='sequence_parallel_size', lines=1, scale=4)
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='gpu_id',
                            multiselect=True,
                            choices=[str(i) for i in range(device_count)] + ['cpu'],
                            value=default_device,
                            scale=8)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                        gr.Dropdown(
                            elem_id='deepspeed',
                            scale=4,
                            allow_custom_value=True,
                            value=None,
                            choices=['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'])
                GRPOHyper.build_ui(base_tab)
                GRPORuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Textbox(elem_id='envs', scale=12)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Rollout.build_ui(base_tab)
                LLMRollout.set_lang(cls.lang)
                LLMRollout.build_ui(LLMRollout)
                GRPOTuner.build_ui(base_tab)
                with gr.Accordion(elem_id='extra_params', open=False):
                    with gr.Tabs():
                        GrpoAdvanced.build_ui(base_tab)
                        GRPOAdvanced.build_ui(base_tab)
                        GRPOQuantization.build_ui(base_tab)
                        GRPOSave.build_ui(base_tab)
                        GRPOReportTo.build_ui(base_tab)
                    with gr.Row():
                        gr.Textbox(elem_id='more_params', lines=4, scale=20)

                cls.element('train_type').change(
                    GRPOHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])

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
                Rollout.element('vllm_mode').change(LLMRollout.external_rollout_display, Rollout.element('vllm_mode'),
                                                    LLMRollout.element('llm_rollout'))
                LLMRollout.element('rollout').click(
                    LLMRollout.rollout_model,
                    list(LLMRollout.valid_elements().values())
                    + [cls.element('model'), cls.element('model_type'),
                       cls.element('template')],
                    [LLMRollout.element('rollout_runtime_tab'),
                     LLMRollout.element('rollout_running_tasks')])

                GRPORuntime.element('kill_task').click(
                    GRPORuntime.kill_task,
                    [GRPORuntime.element('running_tasks')],
                    [GRPORuntime.element('running_tasks')] + [GRPORuntime.element('log')] + GRPORuntime.all_plots,
                ).then(GRPORuntime.reset, [], [GRPORuntime.element('logging_dir')] + [GRPOHyper.element('output_dir')])

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('ddp_num').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])
                GRPOHyper.element('gradient_accumulation_steps').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])
                GRPOHyper.element('per_device_train_batch_size').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'optimizer', 'vllm_mode'],
                                [GRPOTuner.tabs_to_filter, GRPOOptimizer.tabs_to_filter, Rollout.tabs_to_filter])
        }
        return tabs_relation_dict
