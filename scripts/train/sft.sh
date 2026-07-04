#!/bin/bash
# Stage 3. Supervised Fine-Tuning (SFT).
#
# Final teacher-free instruction tuning of the distilled student (LoRA continues from the
# DFT checkpoint, connector and post-connector re-loaded from the same directory).
#
# Effective batch size = 6 (per device) x 4 (grad accum) x 8 (GPUs) = 192.

DATA_PATH=CHANGE_TO_YOUR_DATA_DIR/llava_v1_5_mix665k.json
IMAGE_PATH=CHANGE_TO_YOUR_DATA_DIR/images

# Student
STUDENT_LLM_VERSION=Qwen/Qwen1.5-1.8B
STUDENT_VT_VERSION=google/siglip-so400m-patch14-384
STUDENT_CN_VERSION=mlp2x_gelu
CONV_VERSION=qwen2_base
MODEL_MAX_LENGTH=1536

STUDENT_LLM_VARIANT="${STUDENT_LLM_VERSION#*/}"
OUTPUT_NAME="CompoDistill-${STUDENT_LLM_VARIANT}-SigLIP-SFT"
echo "Output: ${OUTPUT_NAME}"

args=(
    --deepspeed                     ./scripts/zero2.json
    --stage                         sft
    --data_path                     "$DATA_PATH"
    --image_folder                  "$IMAGE_PATH"

    # Distillation
    --pretrain_param_merge          True
    --pretrained_connector_path     ./checkpoints/CompoDistill-${STUDENT_LLM_VARIANT}-SigLIP-DFT

    # Student
    --conv_version                  "$CONV_VERSION"
    --model_name_or_path            "$STUDENT_LLM_VERSION"
    --vision_tower                  "$STUDENT_VT_VERSION"
    --vision_tower2                 ""
    --connector_type                "$STUDENT_CN_VERSION"
    --post_connector_use            True
    --mm_vision_select_layer        -2
    --attn_implementation           sdpa
    --model_max_length              $MODEL_MAX_LENGTH
    --tokenizer_use_fast            False

    # Training
    --training_recipe               lora
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

    --pretrained_model_path         ./checkpoints/CompoDistill-${STUDENT_LLM_VARIANT}-SigLIP-DFT
    --output_dir                    ./checkpoints/"$OUTPUT_NAME"

    --num_train_epochs              1
    --per_device_train_batch_size   6
    --per_device_eval_batch_size    4
    --gradient_accumulation_steps   4
    --evaluation_strategy           "no"
    --save_strategy                 "steps"
    --save_steps                    1000
    --save_total_limit              1
    --gradient_checkpointing        True
    --dataloader_num_workers        8
    --report_to                     "${REPORT_TO:-none}"
    --run_name                      "$OUTPUT_NAME"
    --group_by_modality_length      False

    # Data
    --image_aspect_ratio            square
    --lazy_preprocess               True
    --is_multimodal                 True
)

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29505 compodistill/train/train.py "${args[@]}"
