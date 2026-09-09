"""vLLM 0.21 text-only registration for the exported speech Qwen3.5."""
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)


class SpeechQwen35ForCausalLM(Qwen3_5ForCausalLM, IsHybrid):
    # The upstream text-only class is not registered in 0.21 and does not
    # declare hybrid cache metadata. Reuse the exact Qwen3.5 implementations.
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func.__func__)
