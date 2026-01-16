# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Quantization(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'quantization_tab': {
            'label': {
                'en': 'Quantization settings'
            },
        },
        'quant_method': {
            'label': {
                'en': 'Quantization method'
            },
            'info': {
                'en': 'Default is bnb if quantization_bit is specified'
            }
        },
        'quant_bits': {
            'label': {
                'en': 'Quantization bit'
            },
            'info': {
                'en': 'Set the quantization bit, 0 for no quantization'
            }
        },
        'bnb_4bit_compute_dtype': {
            'label': {
                'en': 'Computational data type'
            },
        },
        'bnb_4bit_quant_type': {
            'label': {
                'en': 'Quantization data type'
            },
        },
        'bnb_4bit_use_double_quant': {
            'label': {
                'en': 'Use double quantization'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='quantization_tab'):
            with gr.Row():
                gr.Dropdown(elem_id='quant_bits', value=None)
                gr.Dropdown(elem_id='quant_method', value=None)
                gr.Dropdown(elem_id='bnb_4bit_compute_dtype', value=None)
                gr.Dropdown(elem_id='bnb_4bit_quant_type', value=None)
                gr.Checkbox(elem_id='bnb_4bit_use_double_quant', value=None)
