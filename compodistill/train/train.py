"""Unified entry point for the CompoDistill three-stage pipeline.

    --stage dpt   Distilled Pre-Training   : logit distillation, post-connector training
    --stage dft   Distilled Fine-Tuning    : logit + visual-attention distillation (LoRA)
    --stage sft   Supervised Fine-Tuning   : plain LM loss, teacher-free
                                             (also used for the vanilla pretrain/finetune
                                              recipes that produce teacher and baselines)

See scripts/train/*.sh for the exact per-stage configurations used in the paper.
"""
import copy
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers

from compodistill.data.dataset import make_supervised_data_module
from compodistill.model import CompoDistillConfig, CompoDistillForConditionalGeneration
from compodistill.train.trainer import DistillTrainer
from compodistill.training_recipe import TrainingRecipeFactory
from compodistill.utils import (
    DataArguments,
    DistillArguments,
    ModelArguments,
    TrainingArguments,
    logger_setting,
)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    return model_args


def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    # flash_attention_2 only supports torch.float16 and torch.bfloat16
    llm_args['attn_implementation'] = model_arguments.attn_implementation
    return llm_args


def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args


def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    connector_args['post_connector_use'] = model_arguments.post_connector_use
    return connector_args


def build_model(model_arguments, data_arguments, training_arguments, pretrain_param_merge):
    """Build a CompoDistill model and load its weights, either from the base HuggingFace
    models or from a previous-stage checkpoint (training_arguments.pretrained_model_path)."""
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_args['pretrain_param_merge'] = pretrain_param_merge

    model_config = CompoDistillConfig()
    model_config.load_from_config(model_arguments)
    model = CompoDistillForConditionalGeneration(model_config)

    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    return model, model_config, training_recipe


def is_merged_checkpoint(path):
    """A merged single-directory checkpoint (e.g. downloaded from our HuggingFace release),
    as opposed to the training layout with language_model/, vision_tower/, connector/."""
    if not os.path.isdir(path):
        return True  # HuggingFace Hub repo id
    return any(os.path.exists(os.path.join(path, name))
               for name in ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin'))


def load_teacher_model(model_arguments, data_arguments, training_arguments, distill_arguments):
    teacher_model_arguments = copy.deepcopy(model_arguments)
    teacher_data_arguments = copy.deepcopy(data_arguments)
    teacher_training_arguments = copy.deepcopy(training_arguments)

    teacher_model_arguments.model_name_or_path = distill_arguments.teacher_model_name_or_path
    teacher_model_arguments.vision_tower = distill_arguments.teacher_vision_tower
    teacher_model_arguments.vision_tower2 = distill_arguments.teacher_vision_tower2
    teacher_model_arguments.connector_type = distill_arguments.teacher_connector_type
    teacher_model_arguments.post_connector_use = False
    teacher_training_arguments.pretrained_model_path = distill_arguments.pretrained_teacher_model_path

    # fp16 overflows for these backbones (generalizability experiments)
    name = teacher_model_arguments.model_name_or_path.lower()
    if 'qwen2.5' in name or 'mobilellama' in name:
        print(f">>>>> Teacher {name}: using bfloat16")
        teacher_training_arguments.fp16 = False
        teacher_training_arguments.bf16 = True

    teacher_path = distill_arguments.pretrained_teacher_model_path
    if teacher_path is not None and is_merged_checkpoint(teacher_path):
        dtype = (torch.float16 if teacher_training_arguments.fp16
                 else (torch.bfloat16 if teacher_training_arguments.bf16 else torch.float32))
        print(f"Loading merged teacher checkpoint from {teacher_path}")
        teacher_model = CompoDistillForConditionalGeneration.from_pretrained(teacher_path, torch_dtype=dtype)
    else:
        teacher_model, _, _ = build_model(
            teacher_model_arguments, teacher_data_arguments, teacher_training_arguments,
            distill_arguments.pretrain_param_merge)

    teacher_model.config.use_cache = False
    teacher_model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    teacher_model.requires_grad_(False)
    return teacher_model


def setup_post_connector(model, teacher_model=None, pretrained_connector_path=None):
    """Replace the student's connector with the (frozen) teacher-shaped connector and attach
    the trainable post-connector that maps the teacher's hidden size to the student's.

    Stage dpt: clone the teacher's connector and create a fresh post-connector.
    Stage dft: same, then overwrite both with the DPT checkpoint (--pretrained_connector_path).
    Stage sft: no teacher is loaded; shapes are recovered from the DFT checkpoint.
    """
    connector = model.connector

    state = None
    if pretrained_connector_path is not None:
        pcn_file = os.path.join(pretrained_connector_path, 'connector', 'pytorch_model.bin')
        print(f"Loading connector and post-connector weights from {pcn_file}")
        state = torch.load(pcn_file, map_location='cpu')

    if teacher_model is not None:
        # Rebinding .data reshapes the student connector to the teacher's dimensions in place.
        for (_, student_param), (_, teacher_param) in zip(connector.named_parameters(),
                                                          teacher_model.connector.named_parameters()):
            student_param.data = teacher_param.data.clone()
        post_in, post_out = teacher_model.config.hidden_size, model.config.hidden_size
    else:
        assert state is not None, "stage sft with post_connector_use requires --pretrained_connector_path"
        for name, param in connector._connector.named_parameters():
            param.data = state[f'_connector.{name}'].clone().to(device=param.device)
        post_out, post_in = state['post_connector.weight'].shape

    connector.post_connector = nn.Linear(post_in, post_out, device=model.device)

    if state is not None:
        result = connector.load_state_dict(state)
        print(f"Post-connector loading -- missing: {result.missing_keys}, "
              f"unexpected: {result.unexpected_keys}")

    connector.post_connector_use = True
    model.config.post_connector_use = True
    model.config.connector_hidden_size = post_in

    for param in connector.post_connector.parameters():
        param.requires_grad = True
    for param in connector._connector.parameters():
        param.requires_grad = False


def train():
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DistillArguments))
    model_arguments, data_arguments, training_arguments, distill_arguments = \
        parser.parse_args_into_dataclasses()

    stage = distill_arguments.stage
    use_teacher = stage in ('dpt', 'dft')

    logger_setting(getattr(training_arguments, 'output_dir', None))

    # Student
    model, model_config, training_recipe = build_model(
        model_arguments, data_arguments, training_arguments, distill_arguments.pretrain_param_merge)
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Teacher (stage dpt / dft only)
    teacher_model = None
    if use_teacher:
        assert distill_arguments.teacher_model_name_or_path is not None, \
            f"stage {stage} requires a teacher (--teacher_model_name_or_path)"
        assert distill_arguments.teacher_conv_version == data_arguments.conv_version, \
            "Different conv versions for teacher and student are not supported"
        teacher_model = load_teacher_model(
            model_arguments, data_arguments, training_arguments, distill_arguments)

    if model_arguments.post_connector_use:
        if stage in ('dft', 'sft'):
            assert distill_arguments.pretrained_connector_path is not None, \
                f"stage {stage} requires --pretrained_connector_path (the previous stage's output directory)"
        setup_post_connector(model, teacher_model, distill_arguments.pretrained_connector_path)
        torch.cuda.empty_cache()

    if os.environ.get("LOCAL_RANK", "0") == "0":
        with open(os.path.join(training_arguments.output_dir, 'trainable_params.txt'), 'w') as f:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(name + "\n")

    # Data
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)

    torch.cuda.empty_cache()
    trainer = DistillTrainer(
        model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_arguments,
        use_attn_loss=distill_arguments.use_attn_loss,
        **data_module)

    if list(pathlib.Path(training_arguments.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
