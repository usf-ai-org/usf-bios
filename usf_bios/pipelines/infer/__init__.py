# Copyright (c) US Inc. All rights reserved.
from typing import TYPE_CHECKING

from usf_bios.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .infer import infer_main, USFInfer
    from .rollout import rollout_main
    from .deploy import deploy_main, USFDeploy, run_deploy
else:
    _import_structure = {
        'rollout': ['rollout_main'],
        'infer': ['infer_main', 'USFInfer'],
        'deploy': ['deploy_main', 'USFDeploy', 'run_deploy'],
        'protocol': ['RequestConfig', 'Function'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
