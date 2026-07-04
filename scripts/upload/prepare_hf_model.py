"""Convert a training checkpoint into a self-contained HuggingFace release.

Loads a training-layout checkpoint (merging LoRA and reconstructing the post-connector),
saves merged full weights together with the standalone modeling files from hf_release/,
and optionally pushes everything to the Hub.

Examples:
    # 1) export locally
    python scripts/upload/prepare_hf_model.py \
        --checkpoint ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT \
        --output-dir ./hf_models/CompoDistill-2B

    # 2) export and push (requires `huggingface-cli login`)
    python scripts/upload/prepare_hf_model.py \
        --checkpoint ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT \
        --output-dir ./hf_models/CompoDistill-2B \
        --push --repo-id <username>/CompoDistill-2B
"""
import argparse
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from compodistill.model import load_pretrained_model

HF_RELEASE_DIR = os.path.join(REPO_ROOT, 'hf_release')

MODEL_CARD = """---
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - multimodal
  - knowledge-distillation
  - compositional-reasoning
  - compodistill
---

# {model_name}

{description}

Released with the paper **CompoDistill: Attention Distillation for Compositional Reasoning
in Multimodal LLMs** ([arXiv:2510.12184](https://arxiv.org/abs/2510.12184)).
Training and evaluation code: https://github.com/ptkjw1997/CompoDistill

## Usage

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

repo = "{repo_id}"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                             torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(repo)

image = Image.open("example.jpg")
print(model.chat("What is happening in this image?", tokenizer,
                 image=image, image_processor=image_processor))
```

## Citation

```bibtex
{bibtex}
```
"""

BIBTEX = """@article{kim2025compodistill,
  title={CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs},
  author={Kim, Jiwan and Kim, Kibum and Seo, Sangwoo and Park, Chanyoung},
  journal={arXiv preprint arXiv:2510.12184},
  year={2025}
}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Training-layout checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default=None, help="<username>/<repo> on the Hub")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--description", type=str, default="A CompoDistill model checkpoint.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print(f"Loading (and merging) {args.checkpoint} ...")
    model, tokenizer, image_processor, _ = load_pretrained_model(args.checkpoint)

    config = model.config
    config.use_cache = True
    config.auto_map = {
        "AutoConfig": "configuration_compodistill.CompoDistillConfig",
        "AutoModelForCausalLM": "modeling_compodistill.CompoDistillForConditionalGeneration",
    }
    config.architectures = ["CompoDistillForConditionalGeneration"]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged weights to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

    for filename in ("configuration_compodistill.py", "modeling_compodistill.py"):
        shutil.copy(os.path.join(HF_RELEASE_DIR, filename), os.path.join(args.output_dir, filename))

    model_name = args.model_name or os.path.basename(args.output_dir.rstrip('/'))
    repo_id = args.repo_id or f"<username>/{model_name}"
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(MODEL_CARD.format(model_name=model_name, repo_id=repo_id,
                                  description=args.description, bibtex=BIBTEX))

    print("Done. Contents:")
    for name in sorted(os.listdir(args.output_dir)):
        print("  ", name)

    if args.push:
        assert args.repo_id, "--push requires --repo-id"
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.repo_id)
        print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
