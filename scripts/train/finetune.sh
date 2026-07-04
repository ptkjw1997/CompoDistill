#!/bin/bash
# Stage 0-b. Vanilla visual instruction tuning (standard TinyLLaVA/LLaVA recipe).
#
#   * teacher       : LLM_VERSION=Qwen/Qwen1.5-4B  bash scripts/train/finetune.sh
#   * SFT baseline  : LLM_VERSION=Qwen/Qwen1.5-1.8B bash scripts/train/finetune.sh
#
# The teacher checkpoint produced here is the distillation teacher of dpt.sh / dft.sh.

DATA_PATH=CHANGE_TO_YOUR_DATA_DIR/llava_v1_5_mix665k.json
IMAGE_PATH=CHANGE_TO_YOUR_DATA_DIR/images

LLM_VERSION=${LLM_VERSION:-Qwen/Qwen1.5-4B}
VT_VERSION=google/siglip-so400m-patch14-384
CN_VERSION=mlp2x_gelu
CONV_VERSION=qwen2_base
MODEL_MAX_LENGTH=2048

LLM_VARIANT="${LLM_VERSION#*/}"
OUTPUT_NAME="CompoDistill-${LLM_VARIANT}-SigLIP-finetune"
echo "Output: ${OUTPUT_NAME}"

deepspeed --include localhost:0,1,2,3 --master_port 29502 compodistill/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --stage sft \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_PATH" \
    --is_multimodal True \
    --conv_version "$CONV_VERSION" \
    --model_name_or_path "$LLM_VERSION" \
    --vision_tower "$VT_VERSION" \
    --vision_tower2 "" \
    --connector_type "$CN_VERSION" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation sdpa \
    --fp16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path ./checkpoints/CompoDistill-${LLM_VARIANT}-SigLIP-pretrain \
    --output_dir ./checkpoints/"$OUTPUT_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "${REPORT_TO:-none}" \
    --tokenizer_use_fast False \
    --run_name "$OUTPUT_NAME"
