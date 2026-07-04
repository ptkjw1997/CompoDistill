#!/bin/bash
# POPE (object hallucination). Prepare the data following the LLaVA v1.5 evaluation guide:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
#
# Usage: bash scripts/eval/pope.sh <MODEL_PATH> [EVAL_DIR]

MODEL_PATH=${1:-./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT}
EVAL_DIR=${2:-CHANGE_TO_YOUR_DATA_DIR/eval}
MODEL_NAME=$(basename "$MODEL_PATH")

python -m compodistill.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base

python -m compodistill.eval.eval_pope \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
