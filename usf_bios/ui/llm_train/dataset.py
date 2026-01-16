# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from usf_bios.dataset import get_dataset_list
from ..base import BaseUI


class Dataset(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'dataset': {
            'label': {
                'en': 'Dataset Code'
            },
            'info': {
                'en': 'The dataset(s) to train the models, support multi select and local folder/files'
            }
        },
        'max_length': {
            'label': {
                'en': 'The max length',
            },
            'info': {
                'en': 'Set the max length input to the model',
            }
        },
        'split_dataset_ratio': {
            'label': {
                'en': 'Split ratio of eval dataset'
            },
            'info': {
                'en': 'Split the datasets by this ratio for eval'
            }
        },
        'padding_free': {
            'label': {
                'en': 'Padding-free batching'
            },
            'info': {
                'en': 'Flatten the data in a batch to avoid data padding'
            }
        },
        'dataset_param': {
            'label': {
                'en': 'Dataset settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='dataset_param', open=True):
            with gr.Row():
                gr.Dropdown(
                    elem_id='dataset', multiselect=True, choices=get_dataset_list(), scale=20, allow_custom_value=True)
                gr.Slider(elem_id='split_dataset_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=10)
                gr.Slider(elem_id='max_length', minimum=32, maximum=32768, value=1024, step=1, scale=10)
                gr.Checkbox(elem_id='padding_free', scale=10)
