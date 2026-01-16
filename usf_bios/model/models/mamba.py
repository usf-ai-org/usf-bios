# Copyright (c) US Inc. All rights reserved.
from transformers import PreTrainedModel

from usf_bios.template import TemplateType
from usf_bios.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class MambaLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger.info_debug(
            '[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
            'be really slow!')
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.mamba,
        [
            ModelGroup([
                Model('mamba-130m-hf', 'state-spaces/mamba-130m-hf'),
                Model('mamba-370m-hf', 'state-spaces/mamba-370m-hf'),
                Model('mamba-390m-hf', 'state-spaces/mamba-390m-hf'),
                Model('mamba-790m-hf', 'state-spaces/mamba-790m-hf'),
                Model('mamba-1.4b-hf', 'state-spaces/mamba-1.4b-hf'),
                Model('mamba-2.8b-hf', 'state-spaces/mamba-2.8b-hf'),
            ])
        ],
        MambaLoader,
        template=TemplateType.default,
        architectures=['MambaForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.39.0'],
    ))
