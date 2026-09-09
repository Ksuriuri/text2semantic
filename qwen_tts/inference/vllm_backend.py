"""Asynchronous continuous-batching semantic generation; optional vLLM import."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoTokenizer

from ..core.models.configuration_text2semantic import Text2SemanticConfig
from ..core.models.modeling_text2semantic import Text2SemanticForCausalLM
from ..semantic_codec import MaskGCTFeatureExtractor
from ..text_conditioning import condition_inference_text, validate_conditioning_tokens
from ..text_template import tokenize_tts_prompt
from .async_utils import offload


class PromptEncoder(nn.Module):
    _encode_speaker_prefix = Text2SemanticForCausalLM._encode_speaker_prefix

    def __init__(self, directory, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        directory = Path(directory)
        self.config = Text2SemanticConfig.from_dict(json.loads(
            (directory / "conditioning_config.json").read_text()))
        # Meta initialization avoids allocating the Qwen backbone twice.
        with torch.device("meta"):
            full = Text2SemanticForCausalLM(self.config)
        self.text_embedding = full.get_input_embeddings()
        self.speech_embedding = full.speech_embedding
        self.speaker_encoder = full.speaker_encoder
        self.speaker_projection = full.speaker_projection
        self.speaker_boundary_embedding = full.speaker_boundary_embedding
        self.speaker_gradient_checkpointing = False
        del full
        self.load_state_dict(load_file(str(directory / "conditioning.safetensors")), assign=True)
        self.to(device=device, dtype=dtype).eval()
        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, ids, features, lengths, prefix):
        device = self.text_embedding.weight.device
        speaker = self._encode_speaker_prefix(features, lengths)[0]
        text = self.text_embedding(torch.tensor(ids, device=device))
        speech = self.speech_embedding(torch.tensor(
            [self.config.speech_bos_token_id, *prefix], device=device))
        return torch.cat([speaker, text, speech], dim=0).contiguous()


class VLLMSemanticBackend:
    def __init__(self, directory, *, w2v_bert_path, stats_path, device="cuda:0",
                 gpu_memory_utilization=0.35, max_model_len=4096, max_num_seqs=16):
        from importlib.metadata import version
        if version("vllm") != "0.21.0":
            raise RuntimeError("This adapter is pinned to vllm==0.21.0")
        from vllm import ModelRegistry
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        directory = Path(directory)
        if not (directory / "READY").is_file():
            raise ValueError("Run vllm_export before starting the engine")
        ModelRegistry.register_model("SpeechQwen35ForCausalLM",
            "qwen_tts.inference.vllm_model:SpeechQwen35ForCausalLM")
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(directory), skip_tokenizer_init=True,
            enable_prompt_embeds=True, dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            enable_prefix_caching=False, enforce_eager=True))
        self.encoder = PromptEncoder(directory, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(directory / "text_tokenizer")
        self.features = MaskGCTFeatureExtractor(w2v_bert_path=w2v_bert_path,
            stats_path=stats_path, device=device)
        self.prepare_lock = asyncio.Lock()

    def _prepare(self, text, ref_audio, language, emotion, prefix):
        text = condition_inference_text(text, language=language, emotion=emotion)
        validate_conditioning_tokens(self.tokenizer)
        features, lengths = self.features.encode_files([str(ref_audio)], max_audio_seconds=15)
        ids = tokenize_tts_prompt(self.tokenizer, text)
        embeds = self.encoder(ids, features, lengths, prefix)
        config = self.encoder.config
        # Dummy positions are BOS (masked from output), not arbitrary semantic
        # IDs. Real prefix IDs preserve the repetition-penalty history.
        history = [config.speech_bos_token_id] * (len(embeds) - len(prefix)) + prefix
        return {"prompt_embeds": embeds.cpu(), "prompt_token_ids": history,
                "prompt_is_token_ids": [False] * len(history)}, features, lengths

    async def generate(self, text, ref_audio, *, language=None, emotion=None,
                       temperature=0.5, top_k=30, max_new_tokens=750,
                       repetition_penalty=1.0, seed=0, prefix_codes=None):
        from vllm import SamplingParams
        prefix = [] if prefix_codes is None else torch.as_tensor(prefix_codes).reshape(-1).tolist()
        config = self.encoder.config
        if any(i < 0 or i >= config.semantic_vocab_size for i in prefix):
            raise ValueError("Invalid semantic prefix")
        if temperature < 0 or repetition_penalty < 1 or max_new_tokens <= 0:
            raise ValueError("Invalid sampling parameters")
        async with self.prepare_lock:
            prompt, features, lengths = await offload(
                self._prepare, text, ref_audio, language, emotion, prefix)
        params = SamplingParams(temperature=temperature, top_k=top_k or -1,
            top_p=1.0, repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens, seed=seed, detokenize=False,
            stop_token_ids=[config.speech_eos_token_id],
            allowed_token_ids=[*range(config.semantic_vocab_size), config.speech_eos_token_id])
        request_id = uuid.uuid4().hex
        final = None
        try:
            async for output in self.engine.generate(prompt, params, request_id):
                final = output
        except BaseException:
            await self.engine.abort(request_id)
            raise
        if final is None:
            raise RuntimeError("Engine returned no output")
        ids = final.outputs[0].token_ids
        ids = [i for i in ids if i != config.speech_eos_token_id]
        if not ids or any(i < 0 or i >= config.semantic_vocab_size for i in ids):
            raise RuntimeError("Engine returned invalid/empty semantic codes")
        return torch.tensor(ids, dtype=torch.long), features, int(lengths[0])

    def close(self):
        self.engine.shutdown()
