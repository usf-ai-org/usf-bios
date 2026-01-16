# Copyright (c) US Inc. All rights reserved.
from usf_bios.utils import get_logger
from ..llm_infer import Runtime

logger = get_logger()


class ExportRuntime(Runtime):

    group = 'llm_export'

    cmd = 'export'

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
                'en': 'Show export status'
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
                'en': 'Please press "Show export status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'en': 'Running export task'
            },
            'info': {
                'en': 'All tasks started by usf export'
            }
        },
        'refresh_tasks': {
            'value': {
                'en': 'Find export'
            },
        },
        'kill_task': {
            'value': {
                'en': 'Kill export'
            },
        },
    }
