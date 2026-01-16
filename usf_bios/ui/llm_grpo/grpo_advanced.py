# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Type

import gradio as gr

from usf_bios.arguments import BaseArguments
from usf_bios.model import ModelType, get_model_list
from ..base import BaseUI


class GrpoAdvanced(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'grpo_advanced_tab': {
            'label': {
                'en': 'GRPO advanced settings'
            },
        },
        'loss_type': {
            'label': {
                'en': 'Loss normalization type'
            }
        },
        'epsilon': {
            'label': {
                'en': 'Clip coefficient'
            }
        },
        'epsilon_high': {
            'label': {
                'en': 'Upper clip coefficient'
            }
        },
        'move_model_batches': {
            'label': {
                'en': 'Batches of model params moving'
            },
            'info': {
                'en': ('How many batches to divide the model into '
                       'when moving parameters to an inference framework such as vLLM')
            }
        },
        'multi_turn_scheduler': {
            'label': {
                'en': 'Multi turn Scheduler'
            },
            'info': {
                'en': 'Multi turn of GRPO parameters, pass in the corresponding plugin name'
            }
        },
        'max_turns': {
            'label': {
                'en': 'Max num of multi turn'
            }
        },
        'dynamic_sample': {
            'label': {
                'en': 'Dynamic sampling'
            },
            'info': {
                'en': 'Filter out data with a reward standard deviation of 0 within the group and sample new data'
            }
        },
        'max_resample_times': {
            'label': {
                'en': 'Max num of resampling times'
            },
            'info': {
                'en': 'Limit the number of resampling times when dynamic_sample is set'
            }
        },
        'overlong_filter': {
            'label': {
                'en': 'Skip overlong samples'
            },
            'info': {
                'en': 'Skip overlong truncated samples and exclude them from loss calculation'
            }
        },
        'beta': {
            'label': {
                'en': 'KL regularization coefficient'
            }
        },
        'vllm_enable_prefix_caching': {
            'label': {
                'en': 'Enable prefix cache'
            },
            'info': {
                'en': 'vLLM transparent transmission parameters in colocate mode'
            }
        },
        'log_completions': {
            'label': {
                'en': 'Record generated content'
            },
            'info': {
                'en': 'Whether to record the model generation content during training'
            }
        },
        'num_iterations': {
            'label': {
                'en': 'Num of updates per batch'
            }
        },
        'reward_model': {
            'label': {
                'en': 'Reward Model id or path'
            },
            'info': {
                'en': 'The actual model id or model path'
            }
        },
        'reward_model_type': {
            'label': {
                'en': 'Select Reward Model Type'
            },
            'info': {
                'en': 'Base model type supported by USF BIOS'
            }
        },
        'reward_model_plugin': {
            'label': {
                'en': 'Reward model logic'
            },
            'info': {
                'en': 'Use reward_model_plugin to customize the processing logic of the reward model'
            }
        },
        'external_plugins': {
            'label': {
                'en': 'External plugin file'
            },
            'info': {
                'en': 'List of external plugin files that will be registered into the plugin module'
            }
        },
        'ref_model_type': {
            'label': {
                'en': 'Ref model type'
            },
            'info': {
                'en': 'Model type supported by USF BIOS'
            }
        },
        'ref_model': {
            'label': {
                'en': 'Ref model id or path'
            },
            'info': {
                'en': 'The actual model id or path'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='grpo_advanced_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='loss_type', choices=['grpo', 'bnpo', 'dr_grpo'], value='grpo', scale=4)
                    gr.Textbox(elem_id='epsilon', value=0.2, lines=1, scale=4)
                    gr.Textbox(elem_id='epsilon_high', value=None, lines=1, scale=4)
                    gr.Textbox(elem_id='beta', value=0.04, lines=1, scale=4)
                    gr.Textbox(elem_id='num_iterations', lines=1, scale=4)
                with gr.Row():
                    gr.Textbox(elem_id='move_model_batches', lines=1, scale=4)
                    gr.Checkbox(elem_id='dynamic_sample', scale=4)
                    gr.Slider(elem_id='max_resample_times', minimum=1, maximum=16, step=1, value=3, scale=4)
                    gr.Checkbox(elem_id='overlong_filter', scale=4)
                    gr.Checkbox(elem_id='vllm_enable_prefix_caching', scale=4)
                with gr.Row():
                    gr.Checkbox(elem_id='log_completions', scale=4)
                    gr.Textbox(elem_id='multi_turn_scheduler', lines=1, scale=4)
                    gr.Textbox(elem_id='max_turns', lines=1, scale=4)
                    gr.Textbox(elem_id='external_plugins', lines=1, scale=8)

            with gr.Row():
                gr.Textbox(elem_id='reward_model_plugin', lines=1, scale=8)
                gr.Dropdown(elem_id='reward_model', multiselect=True, choices=get_model_list(), scale=8)
                gr.Dropdown(
                    elem_id='reward_model_type',
                    multiselect=True,
                    choices=ModelType.get_model_name_list(),
                    allow_custom_value=True,
                    scale=4)
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=12, value=None, choices=get_model_list(), allow_custom_value=True)
                    gr.Dropdown(elem_id='ref_model_type', choices=ModelType.get_model_name_list(), value=None, scale=8)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
        cls.element('reward_model').change(
            partial(cls.update_input_models, allow_keys=['reward_model_type'], is_reward_model=True, has_record=False),
            inputs=[cls.element('reward_model')],
            outputs=[cls.element('reward_model_type')])

    @classmethod
    def update_input_models(cls,
                            models,
                            allow_keys=None,
                            has_record=False,
                            arg_cls=BaseArguments,
                            is_reward_model=False):
        if models is None:
            return gr.update()
        rm_type_str = ''
        for model in models:
            rm_type_str = ' '.join([
                rm_type_str,
                cls.update_input_model(
                    model,
                    allow_keys=allow_keys,
                    has_record=has_record,
                    arg_cls=arg_cls,
                    is_reward_model=is_reward_model)['value']
            ])

        return gr.update(value=rm_type_str.strip())
