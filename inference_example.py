"""Minimal inference example.

Two ways to load a CompoDistill model:

1) From a HuggingFace release (merged weights, no local package needed):
       python inference_example.py --model <repo-or-dir> --image example.jpg --question "..."

2) From a local training checkpoint (LoRA layout produced by scripts/train/*.sh):
       python inference_example.py --model ./checkpoints/CompoDistill-Qwen1.5-1.8B-SigLIP-SFT ...
"""
import argparse
import os

import torch
from PIL import Image


def load_from_hf(path, device):
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                 torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(path)
    return model, tokenizer, image_processor


def load_from_training_checkpoint(path, device):
    from compodistill.model import load_pretrained_model
    model, tokenizer, image_processor, _ = load_pretrained_model(path)
    model.to(device)

    # wrap generation with the qwen2_base conversation template
    from compodistill.data import TextPreprocess
    from compodistill.utils import DEFAULT_IMAGE_TOKEN, Message

    text_processor = TextPreprocess(tokenizer, 'qwen2_base')

    def chat(prompt, tokenizer, image=None, image_processor=None, max_new_tokens=512):
        question = (DEFAULT_IMAGE_TOKEN + '\n' + prompt) if image is not None else prompt
        msg = Message()
        msg.add_message(question)
        input_ids = text_processor(msg.messages, mode='eval')['input_ids'].unsqueeze(0).to(device)
        images = None
        if image is not None:
            images = image_processor(image.convert('RGB'), return_tensors='pt')['pixel_values']
            images = images.to(device=model.device, dtype=model.dtype)
        output_ids = model.generate(input_ids, images=images, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    max_new_tokens=max_new_tokens, use_cache=True)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    model.chat = chat
    return model, tokenizer, image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF repo id / merged directory / training checkpoint directory")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, default="Describe this image in detail.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    is_training_ckpt = os.path.isdir(args.model) and \
        os.path.exists(os.path.join(args.model, 'adapter_config.json'))
    if is_training_ckpt:
        model, tokenizer, image_processor = load_from_training_checkpoint(args.model, args.device)
    else:
        model, tokenizer, image_processor = load_from_hf(args.model, args.device)

    image = Image.open(args.image)
    answer = model.chat(args.question, tokenizer, image=image, image_processor=image_processor)
    print(f"Q: {args.question}\nA: {answer}")


if __name__ == "__main__":
    main()
