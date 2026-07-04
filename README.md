# CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2510.12184-b31b1b.svg)](https://arxiv.org/abs/2510.12184)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

The official implementation of **CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs**.

## Overview

Recently, efficient Multimodal Large Language Models (MLLMs) have gained significant attention as a solution to their high computational complexity, making them more practical for real-world applications. In this regard, the knowledge distillation (KD) approach has emerged as a promising alternative, which transfers the rich visual and linguistic knowledge from a larger model (teacher) to a smaller model (student). However, we observe that existing KD methods struggle to effectively distill the teacher MLLM's rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Through a systematic analysis, we identify visual attention misalignment between student and teacher as the main cause of this issue. Based on this insight, we propose **CompoDistill**, a novel KD framework that explicitly aligns the student's visual attention with that of the teacher to enhance the student's visual perception abilities. Our extensive experiments show that CompoDistill significantly improves performance on compositional reasoning tasks that require visual perception abilities while maintaining strong performance on visual question answering tasks, as done in existing studies. Furthermore, CompoDistill demonstrates effectiveness with a more advanced backbone, highlighting its generalizability.

<p align="center">
  <img src="img/Main_Architecture.png" width="700px">
</p>

## Released Models

| Model | Base LLM | Description | Link |
|---|---|---|---|
| CompoDistill-Teacher-4B | Qwen1.5-4B | Teacher MLLM (LLaVA-style visual instruction tuning) | [🤗 HuggingFace](https://huggingface.co/ptkjw1997/CompoDistill-Teacher-4B) |
| CompoDistill-SFT-2B | Qwen1.5-1.8B | Student baseline trained without distillation | [🤗 HuggingFace](https://huggingface.co/ptkjw1997/CompoDistill-SFT-2B) |
| CompoDistill-2B | Qwen1.5-1.8B | Final CompoDistill student (DPT → DFT → SFT) | [🤗 HuggingFace](https://huggingface.co/ptkjw1997/CompoDistill-2B) |

All released checkpoints are merged full weights and can be used without this repository:

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

repo = "ptkjw1997/CompoDistill-2B"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                             torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(repo)

image = Image.open("example.jpg")
print(model.chat("What is happening in this image?", tokenizer,
                 image=image, image_processor=image_processor))
```

or with this repository: `python inference_example.py --model ptkjw1997/CompoDistill-2B --image example.jpg --question "..."`.

## Installation

```bash
conda create -n compodistill python=3.10 -y
conda activate compodistill
pip install -r requirements.txt
```

A CPU-only sanity check (model assembly, distillation losses, HF release round-trip):

```bash
python tests/smoke_test.py
```

## Data Preparation

Following LLaVA-1.5:

* **Pre-training / DPT**: [LLaVA pre-training data](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) (`blip_laion_cc_sbu_558k.json` + images)
* **Fine-tuning / DFT / SFT**: [LLaVA visual instruction data](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) (`llava_v1_5_mix665k.json` + COCO/GQA/OCR-VQA/TextVQA/VG images)

Set `DATA_PATH` / `IMAGE_PATH` in the scripts under `scripts/train/` accordingly.

## Training

CompoDistill trains a student MLLM (e.g. Qwen1.5-1.8B + SigLIP) with a teacher MLLM
(e.g. Qwen1.5-4B + SigLIP) in three stages, all driven by a single entry point
`compodistill/train/train.py --stage {dpt,dft,sft}`:

| Stage | Script | Loss | Trained modules |
|---|---|---|---|
| 1. Distilled Pre-Training (DPT) | `scripts/train/dpt.sh` | LM + logit distillation | post-connector |
| 2. Distilled Fine-Tuning (DFT) | `scripts/train/dft.sh` | LM + logit distillation + **visual attention alignment** | LoRA(LLM) + post-connector |
| 3. Supervised Fine-Tuning (SFT) | `scripts/train/sft.sh` | LM | LoRA(LLM) + post-connector |

The student's connector is the teacher's frozen connector followed by a trainable linear
*post-connector* (teacher hidden → student hidden); stage boundaries pass it along via
`--pretrained_connector_path`.

**0) Prerequisites** — teacher and student initialization (standard TinyLLaVA recipes):

```bash
# student connector pre-training (also run once with LLM_VERSION=Qwen/Qwen1.5-4B for the teacher)
bash scripts/train/pretrain.sh
# teacher visual instruction tuning
LLM_VERSION=Qwen/Qwen1.5-4B bash scripts/train/finetune.sh
# (optional) SFT-2B baseline without distillation
LLM_VERSION=Qwen/Qwen1.5-1.8B bash scripts/train/finetune.sh
```

You can skip the teacher training by pointing `--pretrained_teacher_model_path` at the
released merged teacher (`ptkjw1997/CompoDistill-Teacher-4B`).

**1–3) CompoDistill pipeline**:

```bash
bash scripts/train/dpt.sh   # Stage 1: Distilled Pre-Training
bash scripts/train/dft.sh   # Stage 2: Distilled Fine-Tuning (attention distillation)
bash scripts/train/sft.sh   # Stage 3: Supervised Fine-Tuning
```

Set `REPORT_TO=wandb` to log to Weights & Biases. GPU allocations in the scripts follow the
paper's setup (4×/8× 48GB GPUs); adjust `--include localhost:...` and the batch settings to
your hardware (keep the effective batch size noted at the top of each script).

## Evaluation

Scripts are provided for the three benchmarks below; the remaining benchmarks in the paper
(GQA, VQAv2, MME, MMMU, SugarCrepe++, Winoground, BiVLC, SADE, etc.) follow the same
generation protocol — adapt `compodistill/eval/model_vqa_loader.py` / `eval_sugarcrepe.py`,
or use the corresponding official evaluation repositories.

```bash
# SugarCrepe (compositional reasoning; data auto-downloaded from HuggingFace)
bash scripts/eval/sugarcrepe.sh ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT

# POPE / TextVQA (prepare data following the LLaVA-1.5 evaluation guide)
bash scripts/eval/pope.sh    ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT <EVAL_DIR>
bash scripts/eval/textvqa.sh ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT <EVAL_DIR>
```

Both training-layout checkpoints (LoRA + component folders) and merged HuggingFace
releases are supported by the evaluation loader.

## Uploading a Trained Model to HuggingFace

```bash
python scripts/upload/prepare_hf_model.py \
    --checkpoint ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT \
    --output-dir ./hf_models/CompoDistill-2B \
    --push --repo-id <username>/CompoDistill-2B   # requires `huggingface-cli login`
```

This merges the LoRA weights, bundles the standalone `hf_release/` modeling files
(`trust_remote_code`), and writes a model card.

## Acknowledgement

This codebase is built upon [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)
and [LLaVA](https://github.com/haotian-liu/LLaVA).

## Citation

```bibtex
@article{kim2025compodistill,
  title={CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs},
  author={Kim, Jiwan and Kim, Kibum and Seo, Sangwoo and Park, Chanyoung},
  journal={arXiv preprint arXiv:2510.12184},
  year={2025}
}
```
