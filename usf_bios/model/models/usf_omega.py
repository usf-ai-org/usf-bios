# Copyright (c) Ultrasafe AI. All rights reserved.
# USF Omega Model Registration
from usf_bios.template import TemplateType
from usf_bios.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import register_model

logger = get_logger()

# Register usf_omega in transformers' CONFIG_MAPPING so AutoConfig/AutoModelForCausalLM
# can recognize this custom model type. UsfOmega is architecturally Mistral-based.
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    class UsfOmegaConfig(MistralConfig):
        model_type = "usf_omega"

    UsfOmegaForCausalLM = MistralForCausalLM

    AutoConfig.register("usf_omega", UsfOmegaConfig)
    AutoModelForCausalLM.register(UsfOmegaConfig, UsfOmegaForCausalLM, exist_ok=True)
except Exception:
    pass

register_model(
    ModelMeta(
        LLMModelType.usf_omega, [
            ModelGroup([
                Model('arpitsh018/usf-omega-40b-base', 'arpitsh018/usf-omega-40b-base'),
            ])
        ],
        template=TemplateType.usf_omega,
        architectures=['UsfOmegaForCausalLM']))
