# Copyright (c) US Inc. All rights reserved.
from usf_bios.utils import get_logger
from ..llm_infer import Runtime

logger = get_logger()


class SampleRuntime(Runtime):

    group = 'llm_sample'

    cmd = 'sample'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'en': 'Runtime'
            },
        },
        'running_cmd': {
            'label': {
                'en': 'Command line'
            },
            'info': {
                'en': 'The actual command'
            }
        },
        'show_log': {
            'value': {
                'en': 'Show running status'
            },
        },
        'stop_show_log': {
            'value': {
                'en': 'Stop showing running status'
            },
        },
        'log': {
            'label': {
                'en': 'Logging content'
            },
            'info': {
                'en': 'Please press "Show running status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'en': 'Running sampling'
            },
            'info': {
                'en': 'Started by usf sample'
            }
        },
        'refresh_tasks': {
            'value': {
                'en': 'Find sampling'
            },
        },
        'kill_task': {
            'value': {
                'en': 'Kill running task'
            },
        },
    }
