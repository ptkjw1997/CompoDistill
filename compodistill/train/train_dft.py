import pdb

import os
import copy
import json
import sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from packaging import version
import pathlib

import tokenizers
import transformers

from compodistill.train.distill_trainer import DistillTrainer
from compodistill.training_recipe import TrainingRecipeFactory
from compodistill.utils import *
from compodistill.model import *
from compodistill.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


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
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
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


def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DistillArguments))
    model_arguments, data_arguments, training_arguments, distill_arguments = parser.parse_args_into_dataclasses()

    # Copy Arguments for Teacher 
    teacher_model_arguments = copy.deepcopy(model_arguments)
    teacher_data_arguments = copy.deepcopy(data_arguments)
    teacher_training_arguments = copy.deepcopy(training_arguments)

    teacher_model_arguments.model_name_or_path = distill_arguments.teacher_model_name_or_path
    teacher_model_arguments.vision_tower = distill_arguments.teacher_vision_tower
    teacher_model_arguments.vision_tower2 = distill_arguments.teacher_vision_tower2
    teacher_model_arguments.connector_type = distill_arguments.teacher_connector_type
    teacher_model_arguments.post_connector_use = False
    teacher_training_arguments.pretrained_model_path = distill_arguments.pretrained_teacher_model_path

    assert distill_arguments.teacher_conv_version == data_arguments.conv_version, \
        "Not Support Different Conv Version for Teacher and Student"
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    # Student Recipe
    training_recipe = TrainingRecipeFactory(
        training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_args['pretrain_param_merge'] = distill_arguments.pretrain_param_merge
    model_config = CompoDistillConfig()
    model_config.load_from_config(model_arguments)
    model = CompoDistillForConditionalGeneration(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Teacher Recipe
    teacher_training_recipe = TrainingRecipeFactory(
        teacher_training_arguments.training_recipe)(teacher_training_arguments)
    teacher_model_args = load_settings(teacher_model_arguments, teacher_data_arguments, teacher_training_arguments)
    teacher_model_args = teacher_training_recipe.add_args(teacher_model_args)
    teacher_model_args['pretrain_param_merge'] = distill_arguments.pretrain_param_merge

    teacher_model_config = CompoDistillConfig()
    teacher_model_config.load_from_config(teacher_model_arguments)
    teacher_model = CompoDistillForConditionalGeneration(teacher_model_config)
    if distill_arguments.pretrained_teacher_model_path is not None:
        teacher_model = teacher_training_recipe.load(teacher_model, teacher_model_args)
    else:
        teacher_model.load_llm(**teacher_model_args['llm'])
        teacher_model.load_vision_tower(**teacher_model_args['vision_tower'])
        teacher_model.load_connector(**teacher_model_args['connector'])
    teacher_model.config.use_cache = False
    teacher_model.config.image_aspect_ratio = data_arguments.image_aspect_ratio

    teacher_model.requires_grad_(False)
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False

    # Post Connector Use Case.
    if model_arguments.post_connector_use :
        print(f"Notice : Post Connector Use Case")
        for (name_a, param_a), (name_b, param_b) in zip(model.connector.named_parameters(), teacher_model.connector.named_parameters()):
            print(f"{name_b} -> {name_a} : Paramter Clonning")
            param_a.data = param_b.data.clone()
            if hasattr(param_a, "out_features"):
                param_a.out_features = param_b.out_features
        model.connector.post_connector_use = True

        model.connector.post_connector = nn.Linear(teacher_model_config.hidden_size, model_config.hidden_size, device = model.device)
        

        pcn_dir = "Distilled_Pre-Trained_Model_Path"
        print(f"Notice: Loaing Post Connector Form : {pcn_dir}")

        pcn_dict = torch.load(f"{pcn_dir}/connector/pytorch_model.bin", map_location = model.device)
        res = model.connector.load_state_dict(pcn_dict)
        print(f"In Post Connector Loading :\nMissing Keys :{res.missing_keys}\nUnexpected Keys :{res.unexpected_keys}")
        
        del pcn_dict

        torch.cuda.empty_cache()

        for name, param in model.connector.post_connector.named_parameters() :
            param.requires_grad = True

        for name, param in model.connector._connector.named_parameters() :
            param.requires_grad = False

    # Data Module
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    # log_trainable_params(model)  # not work well with zero3
    torch.cuda.empty_cache()
    trainer = DistillTrainer(model=model,
                            teacher_model=teacher_model,
                            tokenizer=tokenizer,
                            args=training_arguments,
                            use_attn_loss = distill_arguments.use_attn_loss,
                            **data_module)
    if distill_arguments.curriculum :
        trainer.curriculum = True
    else :
        trainer.curriculum = False

    if list(pathlib.Path(training_arguments.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
