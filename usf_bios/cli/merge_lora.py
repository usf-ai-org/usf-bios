# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
from usf_bios.arguments import ExportArguments
from usf_bios.pipelines import USFPipeline, merge_lora


class USFMergeLoRA(USFPipeline):
    args_class = ExportArguments
    args: args_class

    def run(self):
        merge_lora(self.args)


if __name__ == '__main__':
    USFMergeLoRA().main()
