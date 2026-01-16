# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Reward(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'reward_funcs': {
            'label': {
                'en': 'Reward functions'
            },
            'info': {
                'en': 'GRPO algorithm reward function'
            }
        },
        'reward_weights': {
            'label': {
                'en': 'The weight of each reward function'
            },
            'info': {
                'en': 'The weights of each reward function are separated by spaces'
            }
        },
        'reward_param': {
            'label': {
                'en': 'Reward settings(more params->GRPO advanced settings)'
            },
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='reward_param', open=True):
            with gr.Row():
                gr.Dropdown(
                    elem_id='reward_funcs',
                    multiselect=True,
                    choices=['accuracy', 'format', 'cosine', 'repetition', 'soft_overlong'],
                    scale=2,
                    allow_custom_value=True)
                gr.Textbox(elem_id='reward_weights', lines=1, scale=2)
