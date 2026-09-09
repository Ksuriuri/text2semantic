#!/usr/bin/env python3
"""TTS API with asynchronous AR and acoustic microbatch scheduling."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
import uuid

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# vLLM spawn inherits sys.path after acoustic repositories were added. Those
# repositories also contain webui.py; resolve our sibling before importing it.
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir in sys.path:
    sys.path.remove(_scripts_dir)
sys.path.insert(0, _scripts_dir)
import webui
from qwen_tts.inference.batch_queue import MicrobatchQueue
from qwen_tts.inference.async_utils import offload


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    synthesis_text: str = Field(min_length=1, max_length=20000)
    wav_base64: str
    vocoder_backend: str | None = None
    language: str | None = None
    emotion: str | None = None
    temperature: float = Field(default=0.5, gt=0)
    top_k: int = Field(default=8, ge=0)
    max_new_tokens: int = Field(default=1500, ge=1, le=1500)
    repetition_penalty: float = Field(default=10.0, ge=1)
    speaker_sim_boost: bool = False
    seed: int = Field(default=-1, ge=-1, le=2**32-1)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.models = webui.InferenceApp(args)
        self.ar = None
        if args.ar_backend == "vllm":
            if not args.vllm_model:
                raise ValueError("--vllm-model is required for vLLM")
            from qwen_tts.inference.vllm_backend import VLLMSemanticBackend
            self.ar = VLLMSemanticBackend(args.vllm_model,
                w2v_bert_path=args.w2v_bert_path, stats_path=args.stats_path,
                device=args.device, gpu_memory_utilization=args.vllm_memory_fraction)
        self.queues = {name: MicrobatchQueue(model.vocode_batch,
            max_batch_size=args.acoustic_batch_size, max_wait_ms=args.batch_wait_ms)
            for name, model in self.models.vocoders.items()}
        self.hf_lock = asyncio.Lock()
        self.active = 0
        self.completed = 0
        self.error = None
        self.max_concurrent = args.max_concurrent

    def healthy(self):
        return not self.error and all(not q.task.done() and not q.failed for q in self.queues.values()) and (
            self.ar is None or not self.ar.engine.errored)

    async def synthesize(self, item):
        item = item.model_copy(update={
            "vocoder_backend": item.vocoder_backend or self.models.default_vocoder,
            "emotion": (item.emotion or "").strip() or None,
            "seed": secrets.randbits(32) if item.seed < 0 else item.seed})
        if item.vocoder_backend not in self.queues:
            raise ValueError("Requested acoustic backend is not loaded")
        if not self.healthy():
            raise RuntimeError("Inference worker unavailable")
        if self.active >= self.max_concurrent:
            raise asyncio.QueueFull
        self.active += 1
        start = time.monotonic()
        try:
            audio = base64.b64decode(item.wav_base64, validate=True)
            if not audio or len(audio) > 20 * 1024 * 1024:
                raise ValueError("Reference audio must be between 1 byte and 20 MiB")
            with tempfile.TemporaryDirectory(prefix="tts-request-") as directory:
                ref = Path(directory) / "reference.wav"
                ref.write_bytes(audio)
                text = webui.ttn.normalize(item.synthesis_text).strip()
                if not text:
                    raise ValueError("Text is empty after normalization")
                if item.speaker_sim_boost:
                    async with self.hf_lock:
                        segments, prefix, _ = await offload(self.models._plan_segments,
                            text, str(ref), True)
                else:
                    segments = webui.split_tts_text.plan_segments(text) or [text]
                    prefix = None
                async def segment(index, value):
                    segment_seed = (item.seed + index) % 2**32
                    before_ar = time.monotonic()
                    if self.ar is not None:
                        codes, features, length = await self.ar.generate(value, ref,
                            language=item.language, emotion=item.emotion,
                            temperature=item.temperature, top_k=item.top_k,
                            max_new_tokens=item.max_new_tokens,
                            repetition_penalty=item.repetition_penalty, seed=segment_seed, prefix_codes=prefix)
                    else:
                        # Baseline HF uses process-global RNG, so serialize it.
                        async with self.hf_lock:
                            codes, features, length, _ = await offload(
                                self.models._generate_semantics, text=value, ref_path=str(ref),
                                temperature=item.temperature, top_k=item.top_k,
                                max_new_tokens=item.max_new_tokens,
                                repetition_penalty=item.repetition_penalty, seed=segment_seed,
                                language=item.language, emotion=item.emotion,
                                need_prompt_features=item.vocoder_backend == "s2vae", segments=[value], prefix_codes=prefix)
                    ar_ms = (time.monotonic()-before_ar)*1000
                    wave, info = await self.queues[item.vocoder_backend].submit(dict(
                        codes=codes, ref_audio=str(ref), prompt_features=features,
                        prompt_feature_length=length, seed=segment_seed))
                    return wave, dict(info, ar_ms=ar_ms)
                # Bound the number of active segments per request while allowing
                # multiple requests to join the same acoustic batch.
                results = []
                for offset in range(0, len(segments), self.args.acoustic_batch_size):
                    tasks = [asyncio.create_task(segment(i, segments[i])) for i in
                             range(offset, min(offset+self.args.acoustic_batch_size, len(segments)))]
                    try:
                        results.extend(await asyncio.gather(*tasks))
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                sample_rate = results[0][1]["sample_rate"]
                wave = torch.cat([result[0] for result in results], dim=-1)
                buf = io.BytesIO()
                sf.write(buf, wave.squeeze(0).numpy(), sample_rate, format="WAV", subtype="PCM_16")
                self.completed += 1
                return buf.getvalue(), dict(sample_rate=sample_rate,
                    duration=wave.shape[-1]/sample_rate,
                    total_ms=(time.monotonic()-start)*1000,
                    segments=[result[1] for result in results])
        finally:
            self.active -= 1

    async def deep_check(self):
        ref = getattr(self.args, "warmup_ref", None)
        if not ref:
            raise ValueError("Configure --warmup-ref for inference probes")
        return await self.synthesize(SynthesisRequest(synthesis_text="你好，欢迎使用语音合成。",
            wav_base64=base64.b64encode(Path(ref).read_bytes()).decode(),
            vocoder_backend=self.models.default_vocoder, max_new_tokens=75))

    async def close(self):
        for queue in self.queues.values():
            await queue.close()
        if self.ar is not None:
            self.ar.close()


def create_app(args, pipeline_factory=Pipeline):
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = pipeline_factory(args)
        try:
            if getattr(args, "warmup_ref", None):
                await app.state.pipeline.deep_check()
            yield
        finally:
            await app.state.pipeline.close()

    app = FastAPI(title="Text2Semantic TTS", lifespan=lifespan)
    key = os.environ.get("TTS_API_KEY")

    @app.middleware("http")
    async def auth_and_trace(request, call_next):
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        if key and request.url.path not in ("/health", "/api/health"):
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {key}"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        response = await call_next(request)
        logging.getLogger("tts.api").info("request_id=%s path=%s status=%d total_ms=%.1f",
            request_id, request.url.path, response.status_code, (time.monotonic()-started)*1000)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health")
    @app.get("/api/health")
    async def health():
        pipeline = app.state.pipeline
        ready = pipeline.healthy()
        return JSONResponse(dict(status="ok" if ready else "unhealthy",
            active_requests=pipeline.active, completed=pipeline.completed,
            acoustic_batches={name: q.last_batch_size for name, q in pipeline.queues.items()}),
            status_code=200 if ready else 503)

    probe_lock = asyncio.Lock()
    probe_cache = None

    @app.get("/health/deep")
    async def deep_health():
        nonlocal probe_cache
        if not app.state.pipeline.healthy():
            return JSONResponse({"status": "unhealthy"}, status_code=503)
        async with probe_lock:
            now = time.monotonic()
            if probe_cache and now - probe_cache[0] < 30:
                return JSONResponse(dict(probe_cache[2], cached=True), status_code=probe_cache[1])
            try:
                await asyncio.wait_for(app.state.pipeline.deep_check(), 30)
                code, result = 200, {"status": "ok"}
            except Exception as error:
                code, result = 503, {"status": "unhealthy", "reason": str(error)}
            probe_cache = (time.monotonic(), code, result)
            return JSONResponse(result, status_code=code)

    async def read_payload(request):
        if request.headers.get("content-type", "").startswith("application/json"):
            return await request.json()
        form = await request.form()
        payload = dict(form)
        audio = payload.pop("wav", None) or payload.pop("ref_audio", None)
        if audio is not None:
            data = await audio.read(20*1024*1024+1)
            if len(data) > 20*1024*1024:
                raise ValueError("Reference audio exceeds 20 MiB")
            payload["wav_base64"] = base64.b64encode(data).decode()
        if "text" in payload:
            payload["synthesis_text"] = payload.pop("text")
        return payload

    async def run(item):
        try:
            return await asyncio.wait_for(app.state.pipeline.synthesize(item), args.request_timeout)
        except asyncio.QueueFull:
            raise HTTPException(429, "Inference capacity reached")
        except asyncio.TimeoutError:
            raise HTTPException(504, "Inference deadline exceeded")
        except ValueError as error:
            raise HTTPException(400, str(error))
        except RuntimeError as error:
            raise HTTPException(503, str(error))

    @app.post("/tts")
    @app.post("/api/tts")
    @app.post("/api/generate")
    async def tts(request: Request):
        try:
            item = SynthesisRequest.model_validate(await read_payload(request))
        except (ValueError, ValidationError) as error:
            raise HTTPException(400, str(error))
        audio, info = await run(item)
        return Response(audio, media_type="audio/wav", headers={
            "X-Sample-Rate": str(info["sample_rate"]), "X-Inference-Ms": str(round(info["total_ms"], 2))})

    @app.post("/tts_batch")
    async def batch(request: Request):
        try:
            payload = await read_payload(request)
            if "items" in payload:
                if set(payload) != {"items"}:
                    raise ValueError("items mode cannot be mixed with repeat parameters")
                items = payload["items"]
            else:
                count = int(payload.pop("repeat_num", 3))
                if not 1 <= count <= 16:
                    raise ValueError("repeat_num must be between 1 and 16")
                base_seed = int(payload.get("seed", -1))
                items = [dict(payload, seed=-1 if base_seed < 0 else (base_seed+i) % 2**32) for i in range(count)]
            if not isinstance(items, list) or not 1 <= len(items) <= 16:
                raise ValueError("Batch must contain 1–16 items")
            validated = [SynthesisRequest.model_validate(item) for item in items]
        except (ValueError, TypeError, ValidationError) as error:
            raise HTTPException(400, str(error))
        async def one(index, item):
            try:
                audio, info = await run(item)
                return dict(index=index, audio_base64=base64.b64encode(audio).decode(), **info)
            except HTTPException as error:
                return dict(index=index, error=error.detail, status_code=error.status_code)
        return {"results": await asyncio.gather(*(one(i, item) for i, item in enumerate(validated)))}

    return app


if __name__ == "__main__":
    import uvicorn
    args = webui.parse_args(api=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1)
