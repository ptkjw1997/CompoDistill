import pdb
import os
from collections import defaultdict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler
from datasets import Dataset

from transformers import Trainer
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from ..utils.train_utils import *
from accelerate.utils import is_deepspeed_available, DeepSpeedPlugin, get_active_deepspeed_plugin
from torch.distributed import all_reduce, ReduceOp

if is_deepspeed_available():
    import deepspeed

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

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
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

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
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

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

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
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
    ):
        assert teacher_model is not None, f"Do Not Support No Teacher Model Given"

        self.teacher_model = teacher_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if disable_dropout:
            disable_dropout_in_model(model)
            disable_dropout_in_model(self.teacher_model)

        # Hyperparameter for Distillation
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        if self.use_attn_loss :
            self.output_attentions = True
        else :
            self.output_attentions = None

        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
        zero2_plugin_config = HfTrainerDeepSpeedConfig("./scripts/zero2.json")
        zero2_plugin_config.trainer_config_process(args)
        zero2_plugin = DeepSpeedPlugin(hf_ds_config=zero2_plugin_config)

        zeor3_plugin_config = HfTrainerDeepSpeedConfig("./scripts/zero3.json")
        zeor3_plugin_config.trainer_config_process(args)
        zero3_plugin = DeepSpeedPlugin(hf_ds_config=zeor3_plugin_config)

        args.deepspeed_plugin = {"student":zero2_plugin, "teacher":zero3_plugin}

        super().__init__(
            model = model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            optimizers = optimizers,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        
        self.accelerator.state.select_deepspeed_plugin("teacher")

        self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
        self.teacher_model.eval()
        self.accelerator.state.select_deepspeed_plugin("student")
        
        #### Target Attention ####
        if self.model.config.text_config.num_hidden_layers == 24 : # Qwen1.5 2B / 0.5B
            self.student_target_layers = [8, 9, 10, 11, 12, 13, 14] 
        elif self.model.config.text_config.num_hidden_layers == 40 : # Qwen1.5 4B
            self.student_target_layers = [13, 14, 15, 16, 17, 18, 19]
        elif self.model.config.text_config.num_hidden_layers == 28 : # Qwen2.5 7B
            self.student_target_layers = [9, 10, 11, 12, 13, 14, 15]
        else :
            assert False, f"Not Recommend Student Model Layers : {self.model.config.text_config.num_hidden_layers}"

        if self.teacher_model.config.text_config.num_hidden_layers == 24 : # Qwen1.5 2B
            self.teacher_target_layers = [8, 9, 10, 11, 12, 13, 14]
            self.teacher_group = 2
        elif self.teacher_model.config.text_config.num_hidden_layers == 40 : # Qwen1.5 4B
            self.teacher_target_layers = [13, 14, 15, 16, 17, 18, 19]
            self.teacher_group = 5
        elif self.teacher_model.config.text_config.num_hidden_layers == 28 : # Qwen2.5 7B
            self.teacher_target_layers = [9, 10, 11, 12, 13, 14, 15]
            self.teacher_group = 3
        elif self.teacher_model.config.text_config.num_hidden_layers == 32 : # MobileLLaMA-2.7B
            self.teacher_target_layers = [11, 12, 13, 14, 15, 16, 17]
            self.teacher_group = 3
        else :
            assert False, f"Not Recommend Teacher model Layers : {self.teacher_model.config.text_config.num_hidden_layers}"

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

            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if self.teacher_model :
            return self.compute_loss_distill(model, inputs, return_outputs, num_items_in_batch)
        else :
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def compute_loss_distill(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        """
        Auto-Regression Loss
        Distillation Loss
        Attention Loss
        """

        ## Teacher Logit
        with torch.no_grad() :
            teacher_outputs, teacher_labels, teacher_mask = self.teacher_forward(self.teacher_model, inputs)
            teacher_logit = teacher_outputs.logits
            if self.output_attentions :

                teacher_attn = torch.stack(teacher_outputs.attentions)[:, :, :, 31:(31+729), 31:(31+729)]  # (L, B, H, 729, 729)

                teacher_attn = teacher_attn.mean(dim=2)  # (L, B, 729, 729)
                teacher_attn = torch.stack([teacher_attn[i:i+self.teacher_group].mean(dim=0) for i in self.teacher_target_layers], dim=0)  # (7, B, 729, 729)

        outputs = model(**inputs, output_attentions = self.output_attentions)

        logit = outputs.logits
        if self.output_attentions :
            attn = torch.stack(outputs.attentions).mean(dim = 2)[self.student_target_layers, :, 31:(31+729), 31:(31+729)] 

        ar_loss = outputs.loss 

        # Logit Distillation Loss
        if teacher_logit.shape[-1] != logit.shape[-1] :
            teacher_logit = teacher_logit[:, :, :logit.shape[-1]]

        teacher_prob = F.softmax(teacher_logit, dim = -1)
        student_logprob = F.log_softmax(logit, dim = -1)
        inf_mask = torch.isinf(student_logprob)
        prod_prob = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)

        loss_mask = (teacher_labels != self.label_pad_token_id)
        x = torch.sum(prod_prob, dim = -1).view(-1)

        distill_loss = -torch.sum(x * loss_mask.view(-1), dim = 0) / torch.sum(loss_mask.view(-1), dim = 0)

        loss_outputs = {}
        loss_outputs['loss/distill'] = distill_loss.mean().item()
        loss_outputs['loss/lm'] = ar_loss.mean().item()

        if self.use_attn_loss :
            attn_mu, attn_sigma = attn.mean(dim = [2, 3]), attn.std(dim = [2, 3])
            teacher_attn_mu, teacher_attn_sigma = teacher_attn.mean(dim = [2, 3]), teacher_attn.std(dim = [2, 3])

            attn_filter = (attn <= (attn_mu + attn_sigma).unsqueeze(2).unsqueeze(3)) & \
                                (teacher_attn <= (teacher_attn_mu + teacher_attn_sigma).unsqueeze(2).unsqueeze(3))

            # Cosine Similarity
            attn_filtered = attn * attn_filter
            teacher_attn_filtered = teacher_attn * attn_filter

            n_layer, n_batch, n_token_a, n_token_b = attn.shape
            mask = (teacher_attn_filtered != 0).float()

            attn_sum = (attn_filtered * mask).sum(dim=-2)
            attn_count = mask.sum(dim=-2).clamp(min=1)  # 
            attn_flatten = (attn_sum / attn_count).view(n_layer * n_batch, -1)

            teacher_sum = (teacher_attn_filtered * mask).sum(dim=-2)
            teacher_count = mask.sum(dim=-2).clamp(min=1)
            teacher_attn_flatten = (teacher_sum / teacher_count).view(n_layer * n_batch, -1)

            # Cosine similarity
            attn_loss = 1 - F.cosine_similarity(attn_flatten, teacher_attn_flatten)
            attn_loss = attn_loss.sum() / n_batch
            loss_outputs['loss/attn'] = attn_loss.mean().item()

        else :
            attn_loss = torch.zeros_like(ar_loss)
        

        losses = ar_loss + distill_loss + attn_loss

        self.store_metrics(loss_outputs, train_eval = "train")

        loss_outputs["loss"] = losses.mean().item()
        if return_outputs:
            return losses.mean(), loss_outputs
        else:
            return losses.mean()

    def teacher_forward(
            self,
            model, 
            inputs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        use_cache = model.config.use_cache
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        images = inputs['images']

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images,
        )


        return model.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=None,
            return_dict=None
        ), labels, attention_mask

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either have 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

