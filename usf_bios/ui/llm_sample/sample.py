# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Sample(BaseUI):

    group = 'llm_sample'

    locale_dict = {
        'sampler_type': {
            'label': {
                'en': 'Sampler type'
            },
        },
        'sampler_engine': {
            'label': {
                'en': 'Infer engine'
            },
        },
        'num_return_sequences': {
            'label': {
                'en': 'Num of original sequences returned by sampling'
            },
        },
        'n_best_to_keep': {
            'label': {
                'en': 'Num of best sequences'
            },
        },
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
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Dropdown(elem_id='sampler_type', choices=['sample', 'distill'], value='sample', scale=5)
            gr.Dropdown(
                elem_id='sampler_engine',
                choices=['transformers', 'lmdeploy', 'vllm', 'no', 'client'],
                value='transformers',
                scale=5)
            gr.Slider(elem_id='num_return_sequences', minimum=1, maximum=128, step=1, value=64, scale=5)
            gr.Slider(elem_id='n_best_to_keep', minimum=1, maximum=64, step=1, value=5, scale=5)
        with gr.Row():
            gr.Textbox(elem_id='max_new_tokens', lines=1, value='2048')
            gr.Slider(elem_id='temperature', minimum=0.0, maximum=10, step=0.1, value=1.0)
            gr.Slider(elem_id='top_k', minimum=1, maximum=100, step=5, value=20)
            gr.Slider(elem_id='top_p', minimum=0.0, maximum=1.0, step=0.05, value=0.7)
            gr.Slider(elem_id='repetition_penalty', minimum=0.0, maximum=10, step=0.05, value=1.05)
