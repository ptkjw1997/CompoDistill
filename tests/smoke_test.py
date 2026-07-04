"""CPU smoke tests (no GPU or trained checkpoint required).

Run from the repository root:  python tests/smoke_test.py

T1  imports + template registry + tokenizer/template round-trip (offline tokenizer)
T2  numerical equivalence: extracted loss functions vs original inline implementations
T3  tiny end-to-end: model assembly, setup_post_connector (dpt/dft/sft paths),
    DistillTrainer.compute_loss_distill forward+backward on CPU, teacher-free path
T5  HF flat modeling round-trip (save merged -> AutoModelForCausalLM trust_remote_code)

"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

REPO = os.environ.get("COMPODISTILL_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# any directory or hub repo with a Qwen1.5 tokenizer works here
TOKENIZER_PATH = os.environ.get("COMPODISTILL_TOKENIZER", "Qwen/Qwen1.5-1.8B")
sys.path.insert(0, REPO)
os.chdir(REPO)

torch.manual_seed(0)
PASS = []


def ok(name):
    PASS.append(name)
    print(f"  PASS: {name}")


# ---------------------------------------------------------------- T1: imports
print("== T1: imports & template ==")
from compodistill.model import (CompoDistillConfig, CompoDistillForConditionalGeneration,
                                load_pretrained_model)
from compodistill.train.trainer import (DistillTrainer, attention_alignment_loss,
                                        logit_distillation_loss, ATTENTION_TARGET_LAYERS)
from compodistill.train.train import setup_post_connector
from compodistill.utils import DEFAULT_IMAGE_TOKEN, Message, TrainingArguments
from compodistill.data import TextPreprocess, ImagePreprocess
from compodistill.data.template import TEMPlATE_FACTORY

assert set(TEMPlATE_FACTORY) >= {'pretrain', 'llama', 'qwen2_base'}, TEMPlATE_FACTORY.keys()
ok("imports and template registry (pretrain/llama/qwen2_base)")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
tokenizer.pad_token = tokenizer.unk_token

tp = TextPreprocess(tokenizer, 'qwen2_base')
msg = Message()
msg.add_message(DEFAULT_IMAGE_TOKEN + "\nWhat is in the image?", "A cat.")
enc = tp(msg.messages, mode='train')
assert (enc['input_ids'] == -200).sum() == 1, "image token not inserted"
assert (enc['labels'] != -100).sum() > 0, "no answer tokens in labels"
# the qwen2_base system prompt occupies exactly 31 tokens before the image token
img_pos = (enc['input_ids'] == -200).nonzero()[0].item()
assert img_pos == 31, f"image token position {img_pos} != 31 (trainer default image_token_start)"
enc_eval = tp(msg.messages[:1] + [{'from': 'gpt', 'value': None}], mode='eval')
assert enc_eval['prompt'].endswith("ASSISTANT:")
ok(f"qwen2_base template round-trip (image token at position {img_pos})")


# ------------------------------------------- T2: loss numerical equivalence
print("== T2: loss equivalence vs original implementation ==")


def original_kd_loss(logit, teacher_logit, teacher_labels, label_pad_token_id=-100):
    # verbatim from CompoDistill_2/compodistill/train/distill_trainer.py (lines 419-430)
    if teacher_logit.shape[-1] != logit.shape[-1]:
        teacher_logit = teacher_logit[:, :, :logit.shape[-1]]
    teacher_prob = F.softmax(teacher_logit, dim=-1)
    student_logprob = F.log_softmax(logit, dim=-1)
    inf_mask = torch.isinf(student_logprob)
    prod_prob = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
    loss_mask = (teacher_labels != label_pad_token_id)
    x = torch.sum(prod_prob, dim=-1).view(-1)
    return -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)


def original_attn_loss(attn, teacher_attn):
    # verbatim from CompoDistill_2/compodistill/train/distill_trainer.py (lines 436-461)
    attn_mu, attn_sigma = attn.mean(dim=[2, 3]), attn.std(dim=[2, 3])
    teacher_attn_mu, teacher_attn_sigma = teacher_attn.mean(dim=[2, 3]), teacher_attn.std(dim=[2, 3])
    attn_filter = (attn <= (attn_mu + attn_sigma).unsqueeze(2).unsqueeze(3)) & \
        (teacher_attn <= (teacher_attn_mu + teacher_attn_sigma).unsqueeze(2).unsqueeze(3))
    attn_filtered = attn * attn_filter
    teacher_attn_filtered = teacher_attn * attn_filter
    n_layer, n_batch, n_token_a, n_token_b = attn.shape
    mask = (teacher_attn_filtered != 0).float()
    attn_sum = (attn_filtered * mask).sum(dim=-2)
    attn_count = mask.sum(dim=-2).clamp(min=1)
    attn_flatten = (attn_sum / attn_count).view(n_layer * n_batch, -1)
    teacher_sum = (teacher_attn_filtered * mask).sum(dim=-2)
    teacher_count = mask.sum(dim=-2).clamp(min=1)
    teacher_attn_flatten = (teacher_sum / teacher_count).view(n_layer * n_batch, -1)
    attn_loss = 1 - F.cosine_similarity(attn_flatten, teacher_attn_flatten)
    return attn_loss.sum() / n_batch


B, T, V_s, V_t = 2, 12, 100, 120
logit = torch.randn(B, T, V_s)
teacher_logit = torch.randn(B, T, V_t)
labels = torch.randint(0, V_s, (B, T))
labels[:, :5] = -100
a = original_kd_loss(logit, teacher_logit, labels)
b = logit_distillation_loss(logit, teacher_logit, labels)
assert torch.allclose(a, b, atol=1e-6), (a, b)
ok(f"logit distillation loss identical (value {b.item():.6f}, teacher-vocab truncation covered)")

L, N = 7, 30
attn = torch.rand(L, B, N, N)
teacher_attn = torch.rand(L, B, N, N)
a = original_attn_loss(attn, teacher_attn)
b = attention_alignment_loss(attn, teacher_attn)
assert torch.allclose(a, b, atol=1e-6), (a, b)
ok(f"attention alignment loss identical (value {b.item():.6f})")


# --------------------------------------------------- T3: tiny end-to-end
print("== T3: tiny end-to-end (build, post-connector, trainer loss) ==")
from transformers import Qwen2Config, SiglipVisionConfig


def tiny_config(n_layers, hidden):
    cfg = CompoDistillConfig()
    cfg.llm_model_name_or_path = 'qwen1.5-tiny'          # matched by LLMFactory ('qwen1.5')
    cfg.tokenizer_name_or_path = TOKENIZER_PATH
    cfg.vision_model_name_or_path = 'siglip-tiny'        # matched by VisionTowerFactory ('siglip')
    cfg.connector_type = 'mlp2x_gelu'
    cfg.text_config = Qwen2Config(
        hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=len(tokenizer) + 10,
        max_position_embeddings=4096, attn_implementation='eager')
    cfg.hidden_size = hidden
    cfg.vocab_size = cfg.text_config.vocab_size
    cfg.vision_config = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=112, patch_size=14, attn_implementation='eager')
    cfg.vision_config.model_name_or_path = 'google/siglip-so400m-patch14-384'
    cfg.vision_config.model_name_or_path2 = ''
    cfg.vision_hidden_size = 32
    cfg.tokenizer_model_max_length = 512
    cfg.tokenizer_padding_side = 'right'
    cfg.tokenizer_use_fast = False
    cfg.cache_dir = None
    return cfg


student = CompoDistillForConditionalGeneration(tiny_config(24, 64))
teacher = CompoDistillForConditionalGeneration(tiny_config(40, 96))
teacher.requires_grad_(False)
ok("tiny student (24 layers) and teacher (40 layers) built")

# --- stage dpt path: clone teacher connector + fresh post-connector
setup_post_connector(student, teacher_model=teacher, pretrained_connector_path=None)
assert student.connector.post_connector_use
assert student.connector._connector[0].weight.shape == teacher.connector._connector[0].weight.shape
assert student.connector.post_connector.weight.shape == (64, 96)
assert not any(p.requires_grad for p in student.connector._connector.parameters())
assert all(p.requires_grad for p in student.connector.post_connector.parameters())
assert student.config.connector_hidden_size == 96
ok("setup_post_connector: dpt path (teacher clone + fresh post-connector)")

# save a fake DPT connector checkpoint, then test the dft and sft paths
tmpdir = tempfile.mkdtemp(prefix="compodistill_test_")
dpt_dir = os.path.join(tmpdir, "dpt_out")
os.makedirs(os.path.join(dpt_dir, "connector"))
torch.save(student.connector.state_dict(), os.path.join(dpt_dir, "connector", "pytorch_model.bin"))

student_dft = CompoDistillForConditionalGeneration(tiny_config(24, 64))
setup_post_connector(student_dft, teacher_model=teacher, pretrained_connector_path=dpt_dir)
for (n1, p1), (n2, p2) in zip(student.connector.named_parameters(), student_dft.connector.named_parameters()):
    assert n1 == n2 and torch.equal(p1, p2), f"dft connector mismatch at {n1}"
ok("setup_post_connector: dft path (checkpoint overwrite matches)")

student_sft = CompoDistillForConditionalGeneration(tiny_config(24, 64))
setup_post_connector(student_sft, teacher_model=None, pretrained_connector_path=dpt_dir)
for (n1, p1), (n2, p2) in zip(student.connector.named_parameters(), student_sft.connector.named_parameters()):
    assert n1 == n2 and torch.equal(p1, p2), f"sft connector mismatch at {n1}"
ok("setup_post_connector: sft path (no teacher, shapes recovered from checkpoint)")

# --- batch through the template + collator
from compodistill.data.dataset import DataCollatorForSupervisedDataset
collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
instances = []
for q, ans in [("What is in the image?", "A cat."), ("Describe the picture.", "A small dog runs.")]:
    m = Message()
    m.add_message(DEFAULT_IMAGE_TOKEN + "\n" + q, ans)
    d = tp(m.messages, mode='train')
    d['image'] = torch.randn(3, 112, 112)
    instances.append(d)
batch = collator(instances)
batch['images'] = torch.stack([i['image'] for i in instances])

# --- DistillTrainer on CPU (no deepspeed): distillation loss forward/backward
targs = TrainingArguments(output_dir=os.path.join(tmpdir, "trainer_out"), per_device_train_batch_size=2,
                          use_cpu=True, report_to=[])
NUM_IMG = 8 * 8 - 1  # 112/14=8 -> 64 patches, first dropped by the vision tower
trainer = DistillTrainer(model=student, teacher_model=teacher, args=targs, tokenizer=tokenizer,
                         use_attn_loss=True, image_token_start=31, num_image_tokens=NUM_IMG)
assert trainer.student_target_layers == ATTENTION_TARGET_LAYERS[24][0]
assert trainer.teacher_group == ATTENTION_TARGET_LAYERS[40][1]
loss = trainer.compute_loss_distill(student, batch)
assert loss.isfinite(), loss
loss.backward()
grad = student.connector.post_connector.weight.grad
assert grad is not None and grad.abs().sum() > 0, "post-connector got no gradient"
assert student.connector._connector[0].weight.grad is None, "frozen connector received gradient"
ok(f"DistillTrainer.compute_loss_distill forward+backward (loss {loss.item():.4f}, "
   f"lm {trainer._stored_metrics['train']['loss/lm'][-1]:.4f}, "
   f"distill {trainer._stored_metrics['train']['loss/distill'][-1]:.4f}, "
   f"attn {trainer._stored_metrics['train']['loss/attn'][-1]:.4f})")

# --- teacher-free path (stage sft / vanilla recipes)
trainer_sft = DistillTrainer(model=student_sft, teacher_model=None, args=targs, tokenizer=tokenizer)
loss_sft = trainer_sft.compute_loss(student_sft, {k: v for k, v in batch.items()})
assert loss_sft.isfinite()
ok(f"teacher-free compute_loss falls back to standard Trainer (loss {loss_sft.item():.4f})")


# --------------------------------------------- T5: HF flat modeling round-trip
print("== T5: HF release round-trip (trust_remote_code) ==")
import shutil

hf_dir = os.path.join(tmpdir, "hf_model")
cfg = student.config
cfg.use_cache = True
cfg.auto_map = {
    "AutoConfig": "configuration_compodistill.CompoDistillConfig",
    "AutoModelForCausalLM": "modeling_compodistill.CompoDistillForConditionalGeneration",
}
cfg.architectures = ["CompoDistillForConditionalGeneration"]
student.save_pretrained(hf_dir, safe_serialization=True)
for f in ("configuration_compodistill.py", "modeling_compodistill.py"):
    shutil.copy(os.path.join(REPO, "hf_release", f), os.path.join(hf_dir, f))

from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True)
missing_like = [n for n, p in hf_model.named_parameters() if not p.isfinite().all()]
assert not missing_like

state_a = student.state_dict()
state_b = hf_model.state_dict()
assert set(state_a) == set(state_b), (set(state_a) ^ set(state_b))
for k in state_a:
    assert torch.equal(state_a[k], state_b[k]), f"weight mismatch: {k}"
ok("flat HF model reloads with identical parameters")

with torch.no_grad():
    out_a = student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], images=batch['images'])
    out_b = hf_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                     labels=batch['labels'], images=batch['images'])
assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4), \
    (out_a.logits - out_b.logits).abs().max()
ok(f"flat HF model forward matches package model (max diff "
   f"{(out_a.logits - out_b.logits).abs().max().item():.2e})")

print(f"\nALL {len(PASS)} CHECKS PASSED")
print("tmpdir:", tmpdir)

# ------------------------------------------- T7: package loads a merged release dir
print("== T7: load_pretrained_model on a merged release ==")
merged_dir = os.path.join(REPO, "hf_models", "CompoDistill-2B")
if os.path.isdir(merged_dir):
    m2, tok2, ip2, _ = load_pretrained_model(merged_dir, torch_dtype=torch.float32)
    assert m2.connector.post_connector_use and hasattr(m2.connector, 'post_connector')
    assert m2.connector.post_connector.weight.shape == (2048, 2560)
    del m2
    ok("merged release loads through the package path (post-connector from config)")
else:
    print("  SKIP (no merged release found)")

print(f"\nALL {len(PASS)} CHECKS PASSED")
