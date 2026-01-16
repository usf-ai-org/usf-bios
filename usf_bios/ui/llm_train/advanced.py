# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Advanced(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'advanced_tab': {
            'label': {
                'en': 'Advanced settings'
            },
        },
        'tuner_backend': {
            'label': {
                'en': 'Tuner backend'
            },
            'info': {
                'en': 'The tuner backend'
            }
        },
        'weight_decay': {
            'label': {
                'en': 'Weight decay'
            },
            'info': {
                'en': 'Set the weight decay'
            }
        },
        'logging_steps': {
            'label': {
                'en': 'Logging steps'
            },
            'info': {
                'en': 'Set the logging interval'
            }
        },
        'lr_scheduler_type': {
            'label': {
                'en': 'The LrScheduler type'
            },
            'info': {
                'en': 'Set the LrScheduler type'
            }
        },
        'warmup_ratio': {
            'label': {
                'en': 'Lr warmup ratio'
            },
            'info': {
                'en': 'Set the warmup ratio in total steps'
            }
        },
        'truncation_strategy': {
            'label': {
                'en': 'Dataset truncation strategy'
            },
            'info': {
                'en': 'How to deal with the rows exceed the max length'
            }
        },
        'max_steps': {
            'label': {
                'en': 'Max steps',
            },
            'info': {
                'en': 'Set the max steps, if the value > 0 then num_train_epochs has no effects',
            }
        },
        'max_grad_norm': {
            'label': {
                'en': 'Max grad norm',
            },
            'info': {
                'en': 'Set the max grad norm',
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='advanced_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='tuner_backend', scale=20)
                    gr.Textbox(elem_id='weight_decay', lines=1, scale=20)
                    gr.Textbox(elem_id='logging_steps', lines=1, scale=20)
                    gr.Textbox(elem_id='lr_scheduler_type', lines=1, scale=20)
                with gr.Row():
                    gr.Dropdown(elem_id='truncation_strategy', value=None, scale=20)
                    gr.Textbox(elem_id='max_steps', lines=1, scale=20)
                    gr.Textbox(elem_id='max_grad_norm', lines=1, scale=20)
                    gr.Slider(elem_id='warmup_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=20)
