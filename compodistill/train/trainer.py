from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from accelerate.utils import DeepSpeedPlugin, is_deepspeed_available

from ..utils.train_utils import *

if is_deepspeed_available():
    import deepspeed  # noqa: F401


# Layers whose visual self-attention is aligned between student and teacher, indexed by
# the LLM's number of hidden layers: (target layers, group size). The teacher averages
# `group` consecutive layers starting at each target layer so that both sides produce the
# same number of attention maps.
ATTENTION_TARGET_LAYERS = {
    24: ([8, 9, 10, 11, 12, 13, 14], 2),    # Qwen1.5-0.5B / Qwen1.5-1.8B
    28: ([9, 10, 11, 12, 13, 14, 15], 3),   # Qwen2.5-7B
    32: ([11, 12, 13, 14, 15, 16, 17], 3),  # MobileLLaMA-2.7B
    40: ([13, 14, 15, 16, 17, 18, 19], 5),  # Qwen1.5-4B
}


def logit_distillation_loss(student_logits, teacher_logits, labels, ignore_index=-100):
    """Response (logit) distillation: forward KL between teacher and student distributions,
    averaged over answer tokens (positions whose label is not `ignore_index`).

    When the vocabularies differ, the teacher's logits are truncated to the student's
    vocabulary size (Qwen1.5 checkpoints share the same leading vocabulary)."""
    if teacher_logits.shape[-1] != student_logits.shape[-1]:
        teacher_logits = teacher_logits[:, :, :student_logits.shape[-1]]

    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_logprob = F.log_softmax(student_logits, dim=-1)
    inf_mask = torch.isinf(student_logprob)
    prod_prob = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)

    loss_mask = (labels != ignore_index)
    x = torch.sum(prod_prob, dim=-1).view(-1)
    return -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)


def attention_alignment_loss(student_attn, teacher_attn):
    """CompoDistill visual-attention alignment loss.

    Both inputs are (num_layers, batch, num_image_tokens, num_image_tokens) attention maps
    over the image tokens. Attention sinks are removed by keeping only entries below
    mean + std on both sides; the surviving entries are mean-pooled over the query axis and
    the loss is one minus the cosine similarity of the pooled maps."""
    student_mu, student_sigma = student_attn.mean(dim=[2, 3]), student_attn.std(dim=[2, 3])
    teacher_mu, teacher_sigma = teacher_attn.mean(dim=[2, 3]), teacher_attn.std(dim=[2, 3])

    attn_filter = (student_attn <= (student_mu + student_sigma).unsqueeze(2).unsqueeze(3)) & \
                  (teacher_attn <= (teacher_mu + teacher_sigma).unsqueeze(2).unsqueeze(3))

    student_filtered = student_attn * attn_filter
    teacher_filtered = teacher_attn * attn_filter

    n_layer, n_batch = student_attn.shape[:2]
    mask = (teacher_filtered != 0).float()
    count = mask.sum(dim=-2).clamp(min=1)

    student_pooled = ((student_filtered * mask).sum(dim=-2) / count).view(n_layer * n_batch, -1)
    teacher_pooled = ((teacher_filtered * mask).sum(dim=-2) / count).view(n_layer * n_batch, -1)

    loss = 1 - F.cosine_similarity(student_pooled, teacher_pooled)
    return loss.sum() / n_batch


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def split_to_even_chunks(indices, lengths, num_chunks):
    """Split a list of indices into `chunks` chunks of roughly equal lengths."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""Sampler that groups features of roughly the same length together while keeping a
    bit of randomness."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class DistillTrainer(Trainer):
    """Trainer for the CompoDistill pipeline.

    With a teacher model, the training loss is
        L = L_lm + L_logit_distill (+ L_attn when use_attn_loss)
    computed in `compute_loss_distill`. Without a teacher (stage sft and the vanilla
    pretrain/finetune recipes) it behaves like a standard HuggingFace Trainer.

    The student is trained under the DeepSpeed config passed via --deepspeed
    (`student_deepspeed_config`) while the frozen teacher is sharded with ZeRO-3
    (`teacher_deepspeed_config`); both paths are resolved relative to the repository root.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        disable_dropout: bool = True,
        use_attn_loss: bool = False,
        student_deepspeed_config: str = './scripts/zero2.json',
        teacher_deepspeed_config: str = './scripts/zero3.json',
        # Position of the image tokens inside the multimodal sequence: the qwen2_base
        # system prompt occupies the first 31 tokens and SigLIP-so400m-384 yields
        # 27 x 27 = 729 image tokens.
        image_token_start: int = 31,
        num_image_tokens: int = 729,
    ):
        self.teacher_model = teacher_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.use_attn_loss = use_attn_loss
        self.output_attentions = True if use_attn_loss else None
        self.image_token_start = image_token_start
        self.num_image_tokens = num_image_tokens

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.teacher_model is not None:
                disable_dropout_in_model(self.teacher_model)

        # Student and teacher run under separate DeepSpeed plugins (accelerate >= 0.34).
        use_dual_deepspeed = self.teacher_model is not None and getattr(args, 'deepspeed', None)
        if use_dual_deepspeed:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            student_plugin_config = HfTrainerDeepSpeedConfig(student_deepspeed_config)
            student_plugin_config.trainer_config_process(args)
            student_plugin = DeepSpeedPlugin(hf_ds_config=student_plugin_config)

            teacher_plugin_config = HfTrainerDeepSpeedConfig(teacher_deepspeed_config)
            teacher_plugin_config.trainer_config_process(args)
            teacher_plugin = DeepSpeedPlugin(hf_ds_config=teacher_plugin_config)

            args.deepspeed_plugin = {"student": student_plugin, "teacher": teacher_plugin}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.teacher_model is not None:
            if use_dual_deepspeed:
                self.accelerator.state.select_deepspeed_plugin("teacher")
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
            self.teacher_model.eval()
            if use_dual_deepspeed:
                self.accelerator.state.select_deepspeed_plugin("student")

            self._setup_attention_targets()

    def _setup_attention_targets(self):
        student_layers = self.model.config.text_config.num_hidden_layers
        teacher_layers = self.teacher_model.config.text_config.num_hidden_layers
        assert student_layers in ATTENTION_TARGET_LAYERS, \
            f"No attention-alignment targets registered for a {student_layers}-layer student."
        assert teacher_layers in ATTENTION_TARGET_LAYERS, \
            f"No attention-alignment targets registered for a {teacher_layers}-layer teacher."
        self.student_target_layers = ATTENTION_TARGET_LAYERS[student_layers][0]
        self.teacher_target_layers, self.teacher_group = ATTENTION_TARGET_LAYERS[teacher_layers]

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if self.teacher_model is not None:
            return self.compute_loss_distill(model, inputs, return_outputs)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def compute_loss_distill(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        image_span = slice(self.image_token_start, self.image_token_start + self.num_image_tokens)

        with torch.no_grad():
            teacher_outputs, teacher_labels, _ = self.teacher_forward(self.teacher_model, inputs)
            teacher_logits = teacher_outputs.logits
            if self.use_attn_loss:
                # (L, B, H, N, N) -> head mean -> group mean over target layers -> (7, B, N, N)
                teacher_attn = torch.stack(teacher_outputs.attentions)[:, :, :, image_span, image_span]
                teacher_attn = teacher_attn.mean(dim=2)
                teacher_attn = torch.stack(
                    [teacher_attn[i:i + self.teacher_group].mean(dim=0) for i in self.teacher_target_layers], dim=0)

        outputs = model(**inputs, output_attentions=self.output_attentions)

        ar_loss = outputs.loss
        distill_loss = logit_distillation_loss(
            outputs.logits, teacher_logits, teacher_labels, ignore_index=self.label_pad_token_id)

        loss_outputs = {
            'loss/lm': ar_loss.mean().item(),
            'loss/distill': distill_loss.mean().item(),
        }

        if self.use_attn_loss:
            student_attn = torch.stack(outputs.attentions).mean(dim=2)[self.student_target_layers, :, image_span, image_span]
            attn_loss = attention_alignment_loss(student_attn, teacher_attn)
            loss_outputs['loss/attn'] = attn_loss.mean().item()
        else:
            attn_loss = torch.zeros_like(ar_loss)

        losses = ar_loss + distill_loss + attn_loss
        self.store_metrics(loss_outputs, train_eval="train")

        loss_outputs["loss"] = losses.mean().item()
        if return_outputs:
            return losses.mean(), loss_outputs
        return losses.mean()

    def teacher_forward(
        self,
        model,
        inputs,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        use_cache = model.config.use_cache

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            position_ids=None,
            attention_mask=inputs['attention_mask'],
            past_key_values=None,
            labels=inputs['labels'],
            images=inputs['images'],
        )

        outputs = model.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=None,
            return_dict=None,
        )
        return outputs, labels, attention_mask

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log `logs` on the various objects watching training, including stored metrics."""
        # logs either have 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
