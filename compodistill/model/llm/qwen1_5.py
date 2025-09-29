from transformers import AutoTokenizer, Qwen2ForCausalLM

from . import register_llm

@register_llm('qwen1.5')
def return_qwen1_5class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
