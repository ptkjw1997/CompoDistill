from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)

    model_name_or_path: Optional[str] = field(default="Qwen/Qwen1.5-1.8B")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='')
    vision_tower2: Optional[str] = field(default='')
    connector_type: str = field(default='linear')
    post_connector_use: bool = field(default=False)

    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=128)
    num_resampler_layers: Optional[int] = field(default=3)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tokenizer_use_fast: bool = field(default=False)
    tokenizer_padding_side: str = field(default='right')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    conv_version: str = 'pretrain'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="full") # support only: frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)

    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default=None)


@dataclass
class DistillArguments:
    """Arguments for the CompoDistill three-stage pipeline and the teacher model."""

    stage: str = field(
        default='sft',
        metadata={"help": "One of {dpt, dft, sft}. "
                          "dpt: Distilled Pre-Training (logit distillation, teacher required). "
                          "dft: Distilled Fine-Tuning (logit + attention distillation, teacher required). "
                          "sft: plain Supervised Fine-Tuning (no teacher)."})

    # Teacher model (required for stage dpt / dft)
    teacher_conv_version: Optional[str] = field(default=None)
    teacher_model_name_or_path: Optional[str] = field(default=None)
    teacher_vision_tower: Optional[str] = field(default=None)
    teacher_vision_tower2: Optional[str] = field(default=None)
    teacher_connector_type: Optional[str] = field(default=None)
    pretrained_teacher_model_path: Optional[str] = field(default=None)

    # Distillation options
    pretrain_param_merge: Optional[bool] = field(default=True)
    use_attn_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Align the student's visual attention with the teacher's (stage dft)."})
    pretrained_connector_path: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint directory whose connector/pytorch_model.bin initializes the "
                          "(teacher-shaped) connector and post-connector. "
                          "Stage dft: pass the DPT output directory. Stage sft: pass the DFT output directory."})

    def __post_init__(self):
        assert self.stage in ('dpt', 'dft', 'sft'), f"Unknown stage: {self.stage}"
