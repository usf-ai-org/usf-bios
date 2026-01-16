# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Rollout(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'num_generations': {
            'label': {
                'en': 'Number of samples'
            },
            'info': {
                'en': 'The number of samples for each prompt, that is, the G value in the paper'
            }
        },
        'max_completion_length': {
            'label': {
                'en': 'Max completion length'
            },
            'info': {
                'en': 'Maximum generation length in GRPO algorithm'
            }
        },
        'async_generate': {
            'label': {
                'en': 'Async generate'
            },
            'info': {
                'en': 'Asynchronous rollout to increase training speed'
            }
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
        'use_vllm': {
            'label': {
                'en': 'Using vLLM'
            },
            'info': {
                'en': 'Whether to use vLLM as the infer_backend of generation by GRPO'
            }
        },
        'vllm_mode': {
            'label': {
                'en': 'vLLM Integration Mode'
            },
            'info': {
                'en': 'Server mode uses the vLLM server deployed by usf rollout for sampling, '
                'colocate mode uses vLLM deployed in the program'
            }
        },
        'vllm_gpu_memory_utilization': {
            'label': {
                'en': 'GPU memory utilization'
            },
            'info': {
                'en': 'vLLM transparent transmission parameters'
            }
        },
        'vllm_tensor_parallel_size': {
            'label': {
                'en': 'Tensor parallel size'
            },
            'info': {
                'en': 'vLLM transparent transmission parameters'
            }
        },
        'vllm_max_model_len': {
            'label': {
                'en': 'Max model len'
            },
            'info': {
                'en': 'vLLM transparent transmission parameters'
            }
        },
        'sleep_level': {
            'label': {
                'en': 'Sleep level'
            },
            'info': {
                'en': 'Release vLLM memory during training'
            }
        },
        'vllm_server_host': {
            'label': {
                'en': 'vLLM server host'
            },
        },
        'vllm_server_port': {
            'label': {
                'en': 'vLLM server port'
            },
        },
        'vllm_server_timeout': {
            'label': {
                'en': 'Server timeout'
            },
            'info': {
                'en': 'Timeout for connecting to vLLM server'
            }
        },
        'offload_model': {
            'label': {
                'en': 'Offload model'
            },
            'info': {
                'en': 'Whether to offload the model during vLLM inference'
            }
        },
        'offload_optimizer': {
            'label': {
                'en': 'Offload optimizer'
            },
            'info': {
                'en': 'Whether to offload optimizer parameters during vLLM inference'
            }
        },
        'colocate_param': {
            'label': {
                'en': 'Colocate mode parameters'
            }
        },
        'server_param': {
            'label': {
                'en': 'Server mode parameters'
            }
        },
        'rollout_param': {
            'label': {
                'en': 'Rollout settings(more params->GRPO advanced settings)'
            }
        }
    }

    tabs_to_filter = {
        'colocate': [
            'vllm_enable_prefix_caching', 'vllm_gpu_memory_utilization', 'vllm_tensor_parallel_size',
            'vllm_max_model_len', 'sleep_level', 'offload_model', 'offload_optimizer'
        ],
        'server': ['async_generate', 'vllm_server_host', 'vllm_server_port', 'vllm_server_timeout'],
        'llm_rollout':
        ['tensor_parallel_size', 'data_parallel_size', 'max_model_len', 'gpu_memory_utilization', 'port']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rollout_param', open=False):
            with gr.Row():
                gr.Slider(elem_id='temperature', minimum=0.0, maximum=10, step=0.1, value=1.0)
                gr.Slider(elem_id='top_k', minimum=1, maximum=100, step=5, value=80)
                gr.Slider(elem_id='top_p', minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                gr.Slider(elem_id='repetition_penalty', minimum=0.0, maximum=10, step=0.05, value=1.05)

            with gr.Row():
                gr.Checkbox(elem_id='use_vllm', value=True, scale=4)
                gr.Dropdown(elem_id='vllm_mode', choices=['colocate', 'server'], scale=4)
                gr.Slider(elem_id='num_generations', minimum=1, maximum=64, step=1, scale=4)
                gr.Textbox(elem_id='max_completion_length', lines=1, value='512', scale=4)

            with gr.Accordion(elem_id='colocate_param', open=True):
                with gr.Row():
                    gr.Textbox(elem_id='vllm_gpu_memory_utilization', lines=1, value='0.5', scale=4)
                    gr.Textbox(elem_id='vllm_tensor_parallel_size', lines=1, value='1', scale=4)
                    gr.Textbox(elem_id='vllm_max_model_len', lines=1, value='', scale=4)
                    gr.Dropdown(elem_id='sleep_level', choices=['0', '1'], value='0', scale=4, allow_custom_value=True)
                    gr.Checkbox(elem_id='offload_model', value=True, scale=4)
                    gr.Checkbox(elem_id='offload_optimizer', value=True, scale=4)
            with gr.Accordion(elem_id='server_param', open=True):
                with gr.Row():
                    gr.Checkbox(elem_id='async_generate', scale=4)
                    gr.Textbox(elem_id='vllm_server_host', value='127.0.0.1', scale=4)
                    gr.Textbox(elem_id='vllm_server_port', lines=1, scale=4)
                    gr.Textbox(elem_id='vllm_server_timeout', lines=1, scale=4, value=120)

    @staticmethod
    def update_num_gen(per_device_batch_size, steps_per_generation, num_processes):
        return int(per_device_batch_size) * int(steps_per_generation) * int(num_processes)
