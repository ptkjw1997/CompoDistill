#!/bin/bash

# DATASET_PATH
DISTILL_DATA_PATH=CHANGE_TO_YOUR_DATA_DIR/llava_v1_5_mix665k.json 
DISTILL_IMAGE_PATH=CHANGE_TO_YOUR_DATA_DIR/images/

DISTILL_TRAIN_RECIPE=lora
MODEL_MAX_LENGTH=2048
PRETRAIN_DISTILL=False
USE_ATTN_LOSS=True


# STUDENT MLLM Setting
STUDENT_LLM_VERSION=Qwen/Qwen1.5-1.8B
STUDENT_VT_VERSION=google/siglip-so400m-patch14-384
STUDENT_VT_VERSION2=""
STUDENT_CN_VERSION=mlp2x_gelu
STUDENT_CONV_VERSION=qwen2_base
STUDENT_VERSION=base-lora-zero2-r128
STUDENT_POST_CN_USE=True

STUDENT_VT_VARIANT="${STUDENT_VT_VERSION#*/}"
STUDENT_LLM_VARIANT="${STUDENT_LLM_VERSION#*/}"

# TEACHER MLLM Setting
TEACHER_LLM_VERSION=Qwen/Qwen1.5-4B
TEACHER_VT_VERSION=google/siglip-so400m-patch14-384
TEACHER_VT_VERSION2=""
TEACHER_CN_VERSION=mlp2x_gelu
TEACHER_CONV_VERSION=qwen2_base
TEACHER_VERSION=base-lora-zero2-r128

TEACHER_VT_VARIANT="${TEACHER_VT_VERSION#*/}"
TEACHER_LLM_VARIANT="${TEACHER_LLM_VERSION#*/}"

OUTPUT_FILE_NAME="CompoDistill-${STUDENT_LLM_VARIANT}-${STUDENT_VT_VARIANT}-${STUDENT_VERSION}-distill_finetune"

if [ "$USE_ATTN_LOSS" == "True" ]; then
    OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME}-attn"
fi

OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME}-post_cn-SFT"

echo "Output File Name: ${OUTPUT_FILE_NAME}"

args=(
    --deepspeed                     ./scripts/zero2.json
    --data_path                     "$DISTILL_DATA_PATH"
    --image_folder                  "$DISTILL_IMAGE_PATH"

    # Distillation Experiment Argument
    --pretrain_param_merge          True
    --use_attn_loss                 "$USE_ATTN_LOSS"
    --pretrain_distill              "$PRETRAIN_DISTILL"
    --final_sft                     True

    # STUDENT Args
    --conv_version                  "$STUDENT_CONV_VERSION"
    --model_name_or_path            "$STUDENT_LLM_VERSION"
    --vision_tower                  "$STUDENT_VT_VERSION"
    --vision_tower2                 "$STUDENT_VT_VERSION2"
    --connector_type                "$STUDENT_CN_VERSION"
    --post_connector_use            "$STUDENT_POST_CN_USE"
    --mm_vision_select_layer        -2

    # --attn_implementation           flash_attention_2
    --attn_implementation           sdpa

    --model_max_length              $MODEL_MAX_LENGTH
    --tokenizer_use_fast            False


    # Teacher Args
    --teacher_conv_version          "$TEACHER_CONV_VERSION"
    --teacher_model_name_or_path    "$TEACHER_LLM_VERSION"
    --teacher_vision_tower          "$TEACHER_VT_VERSION"
    --teacher_vision_tower2         "$TEACHER_VT_VERSION2"
    --teacher_connector_type        "$TEACHER_CN_VERSION"


    # Training Args
    --training_recipe               "$DISTILL_TRAIN_RECIPE"
    --tune_type_llm                 lora
    --tune_type_vision_tower        frozen
    --tune_vision_tower_from_layer  0
    --tune_type_connector           full
    --lora_r                        128
    --lora_alpha                    256
    --learning_rate                 2e-4
    --weight_decay                  0.
    --warmup_ratio                  0.03
    --lr_scheduler_type             "cosine"
    --logging_steps                 1
    
    --fp16                          True
    --tf32                          False

    --pretrained_model_path         ./checkpoints/Distilled_Fine-Tuned_Model_Path"
    --pretrained_teacher_model_path ./checkpoints/CompoDistill-${TEACHER_LLM_VARIANT}-${TEACHER_VT_VARIANT}-${TEACHER_VERSION}-finetune
    --output_dir                    ./checkpoints/"${OUTPUT_FILE_NAME}"
    
    --num_train_epochs              0.8
    --per_device_train_batch_size   2
    --per_device_eval_batch_size    4
    --gradient_accumulation_steps   8
    --evaluation_strategy           "no"
    --save_strategy                 "steps"
    --save_steps                    1000
    --save_total_limit              2
    --gradient_checkpointing        True
    --dataloader_num_workers        8
    --report_to                     wandb
    --run_name                      "${OUTPUT_FILE_NAME}"

    --group_by_modality_length      False
    
    
    # Data Args
    --image_aspect_ratio            square
    --lazy_preprocess               True
    --is_multimodal                 True
)

deepspeed --include localhost:0,2,4,5,6,7 compodistill/train/train_sft.py "${args[@]}"