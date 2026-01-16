# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class LoRA(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'lora_tab': {
            'label': {
                'en': 'LoRA settings'
            },
        },
        'lora_rank': {
            'label': {
                'en': 'The LoRA rank'
            }
        },
        'lora_alpha': {
            'label': {
                'en': 'The LoRA alpha'
            }
        },
        'lora_dropout': {
            'label': {
                'en': 'The LoRA dropout'
            }
        },
        'use_rslora': {
            'label': {
                'en': 'Use rsLoRA'
            }
        },
        'use_dora': {
            'label': {
                'en': 'Use DoRA'
            }
        },
        'lora_dtype': {
            'label': {
                'en': 'The dtype of LoRA'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='lora_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='lora_rank', value=8, minimum=1, maximum=512, step=8, scale=2)
                    gr.Slider(elem_id='lora_alpha', value=32, minimum=1, maximum=512, step=8, scale=2)
                    gr.Textbox(elem_id='lora_dropout', scale=2)
                with gr.Row():
                    gr.Dropdown(elem_id='lora_dtype', scale=2, value=None)
                    gr.Checkbox(elem_id='use_rslora', scale=2)
                    gr.Checkbox(elem_id='use_dora', scale=2)
