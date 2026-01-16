import math
import unittest

import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn

from usf_bios.tuners import LoRAConfig, USF_BIOS


class TestMergedLinear(unittest.TestCase):

    def test_usf_bios_lora_forward(self):

        from usf_bios.tuners.lora import MergedLinear

        def reset_parameters(self):
            nn.Linear.reset_parameters(self)
            if hasattr(self, 'lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.ones_(self.lora_B)

        MergedLinear.reset_parameters = reset_parameters

        model = AutoModel.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        lora_config = LoRAConfig(
            target_modules=['query', 'key', 'value'], use_merged_linear=True, enable_lora=[True, True, True])
        outputs = model(**inputs)
        model = USF_BIOS.prepare_model(model, config=lora_config)
        model.eval()
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        USF_BIOS.merge_and_unload(model)
        outputs_merged = model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_reactivate.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_merged.logits, atol=1e-4))
