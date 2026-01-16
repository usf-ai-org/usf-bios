# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Hyper(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'hyper_param': {
            'label': {
                'en': 'Hyper settings(more params->Extra settings)',
            },
        },
        'per_device_train_batch_size': {
            'label': {
                'en': 'Train batch size',
            },
            'info': {
                'en': 'Set the train batch size',
            }
        },
        'per_device_eval_batch_size': {
            'label': {
                'en': 'Val batch size',
            },
            'info': {
                'en': 'Set the val batch size',
            }
        },
        'learning_rate': {
            'label': {
                'en': 'Learning rate',
            },
            'info': {
                'en': 'Set the learning rate',
            }
        },
        'eval_steps': {
            'label': {
                'en': 'Eval steps',
            },
            'info': {
                'en': 'Set the step interval to validate',
            }
        },
        'num_train_epochs': {
            'label': {
                'en': 'Train epoch',
            },
            'info': {
                'en': 'Set the max train epoch',
            }
        },
        'gradient_accumulation_steps': {
            'label': {
                'en': 'Gradient accumulation steps',
            },
            'info': {
                'en': 'Set the gradient accumulation steps',
            }
        },
        'attn_impl': {
            'label': {
                'en': 'Flash Attention Type',
            },
        },
        'neftune_noise_alpha': {
            'label': {
                'en': 'NEFTune noise coefficient'
            },
            'info': {
                'en': 'Use NEFTune to improve performance, normally the value should be 5 or 10'
            }
        },
        'save_steps': {
            'label': {
                'en': 'Save steps',
            },
            'info': {
                'en': 'Set the save steps',
            }
        },
        'output_dir': {
            'label': {
                'en': 'The output dir',
            },
            'info': {
                'en': 'Set the output folder',
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='hyper_param', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='per_device_train_batch_size', minimum=1, maximum=256, step=2, scale=20)
                    gr.Slider(elem_id='per_device_eval_batch_size', minimum=1, maximum=256, step=2, scale=20)
                    gr.Textbox(elem_id='learning_rate', value='1e-4', lines=1, scale=20)
                    gr.Textbox(elem_id='num_train_epochs', lines=1, scale=20)
                    gr.Slider(
                        elem_id='gradient_accumulation_steps',
                        minimum=1,
                        maximum=256,
                        step=2,
                        value=1 if cls.group == 'llm_grpo' else 16,
                        scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='eval_steps', lines=1, value='500', scale=20)
                    gr.Textbox(elem_id='save_steps', value='500', lines=1, scale=20)
                    gr.Textbox(elem_id='output_dir', scale=20)
                    gr.Dropdown(
                        elem_id='attn_impl',
                        value=None,
                        choices=[None, 'sdpa', 'eager', 'flash_attention_2', 'flash_attention_3'],
                        scale=20)
                    gr.Slider(elem_id='neftune_noise_alpha', minimum=0.0, maximum=20.0, step=0.5, scale=20)

    @staticmethod
    def update_lr(sft_type):
        if sft_type == 'full':
            return 1e-5
        else:
            return 1e-4
