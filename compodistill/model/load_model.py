import os
from collections import OrderedDict

import torch
import torch.nn as nn

from .configuration_compodistill import CompoDistillConfig
from .modeling_compodistill import CompoDistillForConditionalGeneration


def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


def _reconstruct_post_connector(model, connector_ckp):
    """Rebuild the CompoDistill connector from a checkpoint: the MLP keeps the teacher's
    hidden size and a linear post-connector maps it down to the student's hidden size.
    Rebinding .data reshapes the freshly built (student-shaped) MLP in place."""
    for name, param in model.connector._connector.named_parameters():
        param.data = connector_ckp[f'_connector.{name}'].clone()

    if 'post_connector.weight' in connector_ckp:
        out_features, in_features = connector_ckp['post_connector.weight'].shape
        model.connector.post_connector = nn.Linear(in_features, out_features)
    else:  # nn.Sequential(GELU, Linear) variant used in the 0.5B ablations
        out_features, in_features = connector_ckp['post_connector.1.weight'].shape
        model.connector.post_connector = nn.Sequential(nn.GELU(), nn.Linear(in_features, out_features))

    result = model.connector.load_state_dict(connector_ckp, strict=False)
    assert not result.unexpected_keys, f"Unexpected connector keys: {result.unexpected_keys}"

    model.connector.post_connector_use = True
    model.config.post_connector_use = True
    model.config.connector_hidden_size = in_features


def load_pretrained_model(model_name_or_path, torch_dtype=None, **kwargs):
    """Load a CompoDistill checkpoint for inference/evaluation.

    Supports both
      * merged single-directory checkpoints (our HuggingFace releases), and
      * training-layout LoRA checkpoints (adapter_config.json + language_model/,
        vision_tower/, connector/); the LoRA weights are merged after loading.

    Returns (model, tokenizer, image_processor, context_len). The model is left on CPU;
    move it to your device afterwards (e.g. `model.to('cuda')`).
    """
    if torch_dtype is None:
        # Qwen2.5-teacher variants were trained in bfloat16, everything else in float16.
        torch_dtype = torch.bfloat16 if 'qwen2.5' in model_name_or_path.lower() else torch.float16

    if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
        model_config = CompoDistillConfig.from_pretrained(model_name_or_path)
        model = CompoDistillForConditionalGeneration(model_config)

        language_model_ckp = load_base_ckp_for_lora(
            os.path.join(model_name_or_path, 'language_model', 'pytorch_model.bin'))
        result = model.language_model.load_state_dict(language_model_ckp, strict=False)
        assert not result.unexpected_keys, f"Unexpected language model keys: {result.unexpected_keys}"

        vision_tower_ckp = load_base_ckp_for_lora(
            os.path.join(model_name_or_path, 'vision_tower', 'pytorch_model.bin'))
        model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)

        connector_ckp = load_base_ckp_for_lora(
            os.path.join(model_name_or_path, 'connector', 'pytorch_model.bin'))
        if any(k.startswith('post_connector') for k in connector_ckp):
            _reconstruct_post_connector(model, connector_ckp)
        else:
            model.connector.load_state_dict(connector_ckp, strict=False)

        model.to(torch_dtype)
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_name_or_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    else:
        model = CompoDistillForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **kwargs)

    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    return model, tokenizer, image_processor, context_len
