# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Generate(BaseUI):

    group = 'llm_infer'

    locale_dict = {
        'max_new_tokens': {
            'label': {
                'en': 'Max new tokens'
            },
        },
        'temperature': {
            'label': {
                'en': 'Temperature'
            },
        },
        'top_k': {
            'label': {
                'en': 'Top-k'
            },
        },
        'top_p': {
            'label': {
                'en': 'Top-p'
            },
        },
        'repetition_penalty': {
            'label': {
                'en': 'Repetition Penalty'
            },
        },
        'system': {
            'label': {
                'en': 'System'
            },
            'info': {
                'en': 'System can be modified after the model weights loaded'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Textbox(elem_id='max_new_tokens', lines=1, value='2048')
            gr.Slider(elem_id='temperature', minimum=0.0, maximum=10, step=0.1, value=0.3)
            gr.Slider(elem_id='top_k', minimum=1, maximum=100, step=5, value=20)
            gr.Slider(elem_id='top_p', minimum=0.0, maximum=1.0, step=0.05, value=0.7)
            gr.Slider(elem_id='repetition_penalty', minimum=0.0, maximum=10, step=0.05, value=1.05)
        with gr.Row():
            gr.Textbox(elem_id='system', lines=4, scale=20)
