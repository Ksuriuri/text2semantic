"""Load the speech model registration in each spawned vLLM process."""
def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model("SpeechQwen35ForCausalLM",
        "qwen_tts.inference.vllm_model:SpeechQwen35ForCausalLM")
