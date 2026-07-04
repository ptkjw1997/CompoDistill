"""Standalone CompoDistill configuration for HuggingFace Hub releases (trust_remote_code).

Self-contained counterpart of compodistill/model/configuration_compodistill.py: it rebuilds
the text/vision sub-configs from the serialized dictionaries in config.json, so loading a
released checkpoint needs neither the compodistill package nor the original base models.
"""
from transformers import CONFIG_MAPPING, AutoConfig, PretrainedConfig

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


def _build_sub_config(config, name_or_path):
    if isinstance(config, PretrainedConfig):
        return config
    if isinstance(config, dict):
        model_type = config.get('model_type')
        if model_type in CONFIG_MAPPING:
            return CONFIG_MAPPING[model_type](**config)
    if name_or_path:
        return AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    return None


class CompoDistillConfig(PretrainedConfig):

    model_type = "compodistill"

    def __init__(
        self,
        llm_model_name_or_path='',
        tokenizer_name_or_path=None,
        vision_model_name_or_path='',
        vision_model_name_or_path2='',
        connector_type=None,
        text_config=None,
        hidden_size=2048,
        vocab_size=32000,
        ignore_index=IGNORE_INDEX,
        image_token_index=IMAGE_TOKEN_INDEX,
        pad_token=None,
        pad_token_id=None,
        tokenizer_padding_side='right',
        tokenizer_model_max_length=2048,
        vision_config=None,
        vision_hidden_size=None,
        vision_feature_layer=-2,
        vision_feature_select_strategy='patch',
        image_aspect_ratio='square',
        use_cache=False,
        cache_dir=None,
        tokenizer_use_fast=False,
        post_connector_use=False,
        connector_hidden_size=None,
        **kwargs
    ):
        self.llm_model_name_or_path = llm_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or self.llm_model_name_or_path
        self.vision_model_name_or_path = vision_model_name_or_path
        self.vision_model_name_or_path2 = vision_model_name_or_path2
        self.connector_type = connector_type

        self.ignore_index = IGNORE_INDEX
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.tokenizer_padding_side = tokenizer_padding_side
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_aspect_ratio = image_aspect_ratio
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.tokenizer_use_fast = tokenizer_use_fast

        # CompoDistill post-connector: the connector keeps the teacher's hidden size
        # (connector_hidden_size) and a linear post-connector maps it to the student's.
        self.post_connector_use = post_connector_use
        self.connector_hidden_size = connector_hidden_size

        self.text_config = _build_sub_config(text_config, self.llm_model_name_or_path)
        if self.text_config is not None:
            self.hidden_size = getattr(self.text_config, 'hidden_size', hidden_size)
            self.vocab_size = getattr(self.text_config, 'vocab_size', vocab_size)
        else:
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size

        self.vision_config = _build_sub_config(vision_config, self.vision_model_name_or_path)
        if self.vision_config is not None:
            self.vision_config = getattr(self.vision_config, 'vision_config', self.vision_config)
            self.vision_hidden_size = getattr(self.vision_config, 'hidden_size', vision_hidden_size)
        else:
            self.vision_hidden_size = vision_hidden_size

        super().__init__(**kwargs)
