# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from usf_bios.dataset import get_dataset_list
from ..base import BaseUI


class Export(BaseUI):

    group = 'llm_export'

    locale_dict = {
        'merge_lora': {
            'label': {
                'en': 'Merge LoRA'
            },
            'info': {
                'en': 'The output path is in the sibling directory as the input checkpoint. '
                'Please refer to the runtime log for more specific information.'
            },
        },
        'device_map': {
            'label': {
                'en': 'The device_map when merge-lora'
            },
            'info': {
                'en': 'If GPU memory is not enough, fill in cpu'
            },
        },
        'quant_bits': {
            'label': {
                'en': 'Quantize bits'
            },
        },
        'quant_method': {
            'label': {
                'en': 'Quantize method'
            },
        },
        'quant_n_samples': {
            'label': {
                'en': 'Sampled rows from calibration dataset'
            },
        },
        'max_length': {
            'label': {
                'en': 'The quantize sequence length'
            },
        },
        'output_dir': {
            'label': {
                'en': 'Output dir'
            },
        },
        'dataset': {
            'label': {
                'en': 'Calibration datasets'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Checkbox(elem_id='merge_lora', scale=10)
            gr.Textbox(elem_id='device_map', scale=20)
        with gr.Row():
            gr.Dropdown(elem_id='quant_bits', scale=20)
            gr.Dropdown(elem_id='quant_method', scale=20)
            gr.Textbox(elem_id='quant_n_samples', scale=20)
            gr.Textbox(elem_id='max_length', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='output_dir', scale=20)
            gr.Dropdown(
                elem_id='dataset', multiselect=True, allow_custom_value=True, choices=get_dataset_list(), scale=20)
