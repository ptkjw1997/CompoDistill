#!/bin/bash
# TextVQA. Prepare the data following the LLaVA v1.5 evaluation guide:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
#
# Usage: bash scripts/eval/textvqa.sh <MODEL_PATH> [EVAL_DIR]

MODEL_PATH=${1:-./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT}
EVAL_DIR=${2:-CHANGE_TO_YOUR_DATA_DIR/eval}
MODEL_NAME=$(basename "$MODEL_PATH")

python -m compodistill.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base

python -m compodistill.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl
