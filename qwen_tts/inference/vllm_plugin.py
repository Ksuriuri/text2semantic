"""Load the speech model registration in each spawned vLLM process."""
def register():
    from vllm import ModelRegistry
    if "SpeechQwen35ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("SpeechQwen35ForCausalLM",
            "qwen_tts.inference.vllm_model:SpeechQwen35ForCausalLM")
