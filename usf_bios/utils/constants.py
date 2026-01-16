# Copyright (c) US Inc. All rights reserved.

BIN_EXTENSIONS = [
    '.*.bin',
    '.*.ts',
    '.*.pt',
    '.*.data-00000-of-00001',
    '.*.onnx',
    '.*.meta',
    '.*.pb',
    '.*.index',
]

PEFT_TYPE_KEY = 'peft_type'
USF_TYPE_KEY = 'usf_type'
DEFAULT_ADAPTER = 'default'


class Invoke(object):
    KEY = 'invoked_by'
    THIRD_PARTY = 'third_party'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'
    USF_BIOS = 'usf_bios'
