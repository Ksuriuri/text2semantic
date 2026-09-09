"""vLLM 0.21 text-only registration for the exported speech Qwen3.5."""
import torch
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)


class SpeechQwen35ForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    # The upstream text-only class is not registered in 0.21 and does not
    # declare hybrid cache metadata. Reuse the exact Qwen3.5 implementations.
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func.__func__)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("Speech Qwen3.5 only accepts precomputed text/speaker embeddings")
        # The training text backbone expands scalar positions equally across
        # all three RoPE axes. Keep that representation (and zero delta).
        return torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1).clone(), 0
