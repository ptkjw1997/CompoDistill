#!/bin/bash
# SugarCrepe (compositional reasoning). Images/captions are downloaded automatically
# from HuggingFace (HuggingFaceM4/SugarCrepe_*).
#
# Usage: bash scripts/eval/sugarcrepe.sh <MODEL_PATH>

MODEL_PATH=${1:-./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT}
MODEL_NAME=$(basename "$MODEL_PATH")

python -m compodistill.eval.eval_sugarcrepe \
    --model-path "$MODEL_PATH" \
    --conv-mode qwen2_base \
    --output-dir results/sugarcrepe/"$MODEL_NAME"
