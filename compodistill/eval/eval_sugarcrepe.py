"""SugarCrepe evaluation.

For every sample the model picks which of two captions (the positive and a hard-negative)
matches the image. Each pair is asked twice -- once with the positive caption as option (A)
and once as option (B) -- and we report the accuracy of each ordering plus their average.

Usage:
    python -m compodistill.eval.eval_sugarcrepe --model-path <checkpoint> --output-dir results/sugarcrepe
"""
import argparse
import json
import os
import re

import torch
from tqdm import tqdm

from compodistill.data import ImagePreprocess, TextPreprocess
from compodistill.model import load_pretrained_model
from compodistill.utils import (
    DEFAULT_IMAGE_TOKEN,
    KeywordsStoppingCriteria,
    Message,
    disable_torch_init,
)

SUBSETS = ['swap_obj', 'swap_att', 'replace_obj', 'replace_rel', 'replace_att', 'add_obj', 'add_att']


def build_prompt(text_processor, caption_a, caption_b):
    msg = Message()
    prompt = (DEFAULT_IMAGE_TOKEN + "\n" + "Which caption best describes the image?\n"
              f"(A) {caption_a}\n (B) {caption_b}\n"
              "Answer with the option's alphabet from the given choices directly.")
    msg.add_message(prompt)
    return text_processor(msg.messages, mode='eval')


def parse_choice(response):
    match = re.search(r'\b([AB])\b', response.strip().upper())
    return match.group(1) if match else None


@torch.inference_mode()
def answer(model, tokenizer, text_processor, image_tensor, conv, max_new_tokens=32):
    input_ids = conv['input_ids'].unsqueeze(0).to(model.device)
    stop_str = text_processor.template.separator.apply()[1]
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="qwen2_base")
    parser.add_argument("--output-dir", type=str, default="results/sugarcrepe")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: cap samples per subset.")
    args = parser.parse_args()

    from datasets import load_dataset

    disable_torch_init()
    model, tokenizer, image_processor, _ = load_pretrained_model(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    image_preprocess = ImagePreprocess(image_processor, model.config)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    for subset in SUBSETS:
        dataset = load_dataset(f"HuggingFaceM4/SugarCrepe_{subset}")['test']
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        records, correct = [], {'pos_first': 0, 'neg_first': 0}
        for sample in tqdm(dataset, desc=subset):
            pos_caption, neg_caption = sample['tested_labels']
            image_tensor = image_preprocess(sample['image'].convert("RGB"))
            image_tensor = image_tensor.unsqueeze(0).to(dtype=model.dtype, device=model.device)

            record = {'positive': pos_caption, 'negative': neg_caption}
            for order, (a, b, answer_key) in {
                'pos_first': (pos_caption, neg_caption, 'A'),
                'neg_first': (neg_caption, pos_caption, 'B'),
            }.items():
                conv = build_prompt(text_processor, a, b)
                response = answer(model, tokenizer, text_processor, image_tensor, conv)
                choice = parse_choice(response)
                correct[order] += int(choice == answer_key)
                record[order] = {'response': response, 'choice': choice, 'answer': answer_key}
            records.append(record)

        n = len(records)
        accs = {order: correct[order] / n * 100 for order in correct}
        accs['average'] = sum(accs.values()) / 2
        summary[subset] = accs
        print(f"[{subset}] pos-first {accs['pos_first']:.2f} | neg-first {accs['neg_first']:.2f} "
              f"| average {accs['average']:.2f}")

        with open(os.path.join(args.output_dir, f"{subset}.json"), 'w') as f:
            json.dump({'accuracy': accs, 'records': records}, f, indent=2)

    summary['overall_average'] = sum(v['average'] for v in summary.values()) / len(summary)
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n===== SugarCrepe summary =====")
    for subset in SUBSETS:
        print(f"{subset:>12}: {summary[subset]['average']:.2f}")
    print(f"{'overall':>12}: {summary['overall_average']:.2f}")


if __name__ == "__main__":
    main()
