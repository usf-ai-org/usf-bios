# Copyright (c) US Inc. All rights reserved.

from .adapter import Adapter, AdapterConfig
from .llamapro import LLaMAPro, LLaMAProConfig
from .longlora.longlora import LongLoRA, LongLoRAConfig
from .lora import LoRA, LoRAConfig
from .neftune import NEFTune, NEFTuneConfig
from .part import Part, PartConfig
from .prompt import Prompt, PromptConfig
from .reft import Reft, ReftConfig
from .restuning import ResTuning, ResTuningConfig
from .scetuning.scetuning import SCETuning, SCETuningConfig
from .side import Side, SideConfig


class USFTuners:
    ADAPTER = 'ADAPTER'
    PROMPT = 'PROMPT'
    LORA = 'LORA'
    SIDE = 'SIDE'
    RESTUNING = 'RESTUNING'
    LONGLORA = 'longlora'
    NEFTUNE = 'neftune'
    LLAMAPRO = 'LLAMAPRO'
    SCETUNING = 'SCETuning'
    PART = 'part'
    REFT = 'reft'


USF_MAPPING = {
    USFTuners.ADAPTER: (AdapterConfig, Adapter),
    USFTuners.PROMPT: (PromptConfig, Prompt),
    USFTuners.LORA: (LoRAConfig, LoRA),
    USFTuners.SIDE: (SideConfig, Side),
    USFTuners.RESTUNING: (ResTuningConfig, ResTuning),
    USFTuners.LONGLORA: (LongLoRAConfig, LongLoRA),
    USFTuners.NEFTUNE: (NEFTuneConfig, NEFTune),
    USFTuners.SCETUNING: (SCETuningConfig, SCETuning),
    USFTuners.LLAMAPRO: (LLaMAProConfig, LLaMAPro),
    USFTuners.PART: (PartConfig, Part),
    USFTuners.REFT: (ReftConfig, Reft),
}
