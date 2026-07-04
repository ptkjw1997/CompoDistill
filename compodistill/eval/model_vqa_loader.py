"""Generate answers for a jsonl question file (LLaVA evaluation format).

Used for POPE and TextVQA; score the resulting answer file with eval_pope.py or
eval_textvqa.py. See scripts/eval/*.sh.
"""
import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from compodistill.data import ImagePreprocess, TextPreprocess
from compodistill.model import load_pretrained_model
from compodistill.utils import DEFAULT_IMAGE_TOKEN, Message, disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


class QuestionDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image = Image.open(os.path.join(self.image_folder, line["image"])).convert('RGB')
        image_tensor = self.image_processor(image)

        msg = Message()
        msg.add_message(DEFAULT_IMAGE_TOKEN + '\n' + line["text"])
        result = self.text_processor(msg.messages, mode='eval')

        return result['input_ids'], image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, _ = load_pretrained_model(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(args.device)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    image_preprocess = ImagePreprocess(image_processor, model.config)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = QuestionDataset(questions, args.image_folder, text_processor, image_preprocess)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        input_ids = input_ids.to(device=args.device, non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=model.dtype, device=args.device, non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_sizes=image_sizes,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_file.write(json.dumps({"question_id": line["question_id"],
                                   "prompt": line["text"],
                                   "text": outputs,
                                   "answer_id": shortuuid.uuid(),
                                   "model_id": os.path.basename(model_path.rstrip('/')),
                                   "metadata": {}}) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen2_base")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    eval_model(args)
