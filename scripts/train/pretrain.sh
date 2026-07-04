#!/bin/bash
# Stage 0-a. Vanilla connector pre-training (standard TinyLLaVA/LLaVA recipe).
# Produces the initial student (or teacher) checkpoint consumed by the later stages.
#
# Set LLM_VERSION=Qwen/Qwen1.5-4B to pre-train the teacher.

DATA_PATH=CHANGE_TO_YOUR_DATA_DIR/blip_laion_cc_sbu_558k.json
IMAGE_PATH=CHANGE_TO_YOUR_DATA_DIR/images

LLM_VERSION=${LLM_VERSION:-Qwen/Qwen1.5-1.8B}
VT_VERSION=google/siglip-so400m-patch14-384
CN_VERSION=mlp2x_gelu
MODEL_MAX_LENGTH=2048

LLM_VARIANT="${LLM_VERSION#*/}"
OUTPUT_NAME="CompoDistill-${LLM_VARIANT}-SigLIP-pretrain"
echo "Output: ${OUTPUT_NAME}"

deepspeed --include localhost:0,1,2,3 --master_port 29501 compodistill/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --stage sft \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_PATH" \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path "$LLM_VERSION" \
    --vision_tower "$VT_VERSION" \
    --vision_tower2 "" \
    --connector_type "$CN_VERSION" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation sdpa \
    --fp16 True \
    --training_recipe common \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir ./checkpoints/"$OUTPUT_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
