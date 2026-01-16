# Copyright (c) US Inc. All rights reserved.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from usf_bios.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'usf_bios.trainers.Seq2SeqTrainer',
        'seq_cls': 'usf_bios.trainers.Trainer',
        'embedding': 'usf_bios.trainers.EmbeddingTrainer',
        'reranker': 'usf_bios.trainers.RerankerTrainer',
        'generative_reranker': 'usf_bios.trainers.RerankerTrainer',
        # rlhf
        'dpo': 'usf_bios.rlhf_trainers.DPOTrainer',
        'orpo': 'usf_bios.rlhf_trainers.ORPOTrainer',
        'kto': 'usf_bios.rlhf_trainers.KTOTrainer',
        'cpo': 'usf_bios.rlhf_trainers.CPOTrainer',
        'rm': 'usf_bios.rlhf_trainers.RewardTrainer',
        'ppo': 'usf_bios.rlhf_trainers.PPOTrainer',
        'grpo': 'usf_bios.rlhf_trainers.GRPOTrainer',
        'gkd': 'usf_bios.rlhf_trainers.GKDTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'usf_bios.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'usf_bios.trainers.TrainingArguments',
        'embedding': 'usf_bios.trainers.TrainingArguments',
        'reranker': 'usf_bios.trainers.TrainingArguments',
        'generative_reranker': 'usf_bios.trainers.TrainingArguments',
        # rlhf
        'dpo': 'usf_bios.rlhf_trainers.DPOConfig',
        'orpo': 'usf_bios.rlhf_trainers.ORPOConfig',
        'kto': 'usf_bios.rlhf_trainers.KTOConfig',
        'cpo': 'usf_bios.rlhf_trainers.CPOConfig',
        'rm': 'usf_bios.rlhf_trainers.RewardConfig',
        'ppo': 'usf_bios.rlhf_trainers.PPOConfig',
        'grpo': 'usf_bios.rlhf_trainers.GRPOConfig',
        'gkd': 'usf_bios.rlhf_trainers.GKDConfig',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
        else:
            train_method = args.task_type
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args._prepare_training_args(args_dict)
        training_args = training_args_cls(**args_dict)
        return training_args
