# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Optimizer(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'galore_tab': {
            'label': {
                'en': 'GaLore Settings'
            },
        },
        'use_galore': {
            'label': {
                'en': 'Use GaLore'
            },
            'info': {
                'en': 'Use GaLore to reduce GPU memory usage in full parameter training'
            }
        },
        'galore_rank': {
            'label': {
                'en': 'The rank of GaLore'
            },
        },
        'galore_update_proj_gap': {
            'label': {
                'en': 'Projection matrix update interval'
            },
            'info': {
                'en': 'Update interval of GaLore decomposition matrix'
            },
        },
        'galore_with_embedding': {
            'label': {
                'en': 'Use GaLore with embedding'
            },
            'info': {
                'en': 'Whether to apply GaLore to embedding'
            },
        },
        'lorap_tab': {
            'label': {
                'en': 'LoRA+ settings'
            },
        },
        'lorap_lr_ratio': {
            'label': {
                'en': 'LoRA+ lr ratio'
            },
            'info': {
                'en': 'When using LoRA, specify this parameter to use LoRA+, and the recommended value is 10 to 16'
            }
        },
        'muon_tab': {
            'label': {
                'en': 'Muon Settings'
            },
        },
        'use_muon': {
            'label': {
                'en': 'Use Muon'
            },
            'info': {
                'en': 'Using the Muon optimizer, set `--optimizer muon` in the command line'
            }
        },
        'multimodal_tab': {
            'label': {
                'en': 'Multimodal Settings'
            },
        },
        'vit_lr': {
            'label': {
                'en': 'Learning rate of ViT'
            },
        },
        'aligner_lr': {
            'label': {
                'en': 'Learning rate of aligner'
            },
        },
        'optimizer_params': {
            'label': {
                'en': 'Optimizer params'
            },
        },
    }

    tabs_to_filter = {
        'galore': ['use_galore', 'galore_with_embedding', 'galore_rank', 'galore_update_proj_gap'],
        'lorap': ['lorap_lr_ratio'],
        'multimodal': ['vit_lr', 'aligner_lr'],
        'muon': ['use_muon']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='optimizer_params', open=False):
            with gr.Tabs():
                with gr.TabItem(elem_id='galore_tab'):
                    with gr.Row():
                        gr.Checkbox(elem_id='use_galore', scale=4)
                        gr.Checkbox(elem_id='galore_with_embedding', scale=4)
                        gr.Slider(elem_id='galore_rank', minimum=8, maximum=256, step=8, scale=4)
                        gr.Slider(elem_id='galore_update_proj_gap', minimum=10, maximum=1000, step=50, scale=4)
                with gr.TabItem(elem_id='lorap_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='lorap_lr_ratio', scale=4)
                with gr.TabItem(elem_id='multimodal_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='vit_lr', lines=1, scale=20)
                        gr.Textbox(elem_id='aligner_lr', lines=1, scale=20)
                with gr.TabItem(elem_id='muon_tab'):
                    with gr.Row():
                        gr.Checkbox(elem_id='use_muon', scale=4)
