"""Length-aware acoustic microbatches with independent request noise."""
from __future__ import annotations

from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence


@torch.inference_mode()
def flow_batch(cfm, items, *, steps, cfg_rate, temperature=1.0, drop_style=False):
    if not items or steps < 1:
        raise ValueError("A nonempty batch and positive steps are required")
    mu = pad_sequence([item["mu"][0] for item in items], batch_first=True)
    device, dtype = mu.device, mu.dtype
    lengths = torch.tensor([item["mu"].shape[1] for item in items], device=device)
    prompt_lengths = torch.tensor([item["prompt"].shape[-1] for item in items], device=device)
    positions = torch.arange(mu.shape[1], device=device)[None, :]
    prompt_mask = positions < prompt_lengths[:, None]
    valid = positions < lengths[:, None]
    prompt = mu.new_zeros(len(items), cfm.in_channels, mu.shape[1])
    x = torch.zeros_like(prompt)
    scale = float(getattr(cfm, "feat_scale", 1.0))
    for i, item in enumerate(items):
        n = item["mu"].shape[1]
        p = item["prompt"].shape[-1]
        if p >= n:
            raise ValueError("Acoustic prompt must be shorter than total length")
        prompt[i, :, :p] = item["prompt"][0] * scale
        generator = torch.Generator(device=device).manual_seed(item.get("seed", 0))
        x[i, :, :n] = torch.randn(cfm.in_channels, n, generator=generator,
                                 device=device, dtype=dtype) * temperature
    x.masked_fill_(prompt_mask[:, None, :], 0)
    if cfm.zero_prompt_speech_token:
        mu = mu.masked_fill(prompt_mask[:, :, None], 0)
    style = torch.cat([item["style"] for item in items])
    kwargs = {"drop_style": True} if drop_style else {}
    times = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    for step in range(steps):
        t = times[step].expand(len(items))
        if cfg_rate > 0:
            prediction = cfm.estimator(
                torch.cat([x, x]), torch.cat([prompt, torch.zeros_like(prompt)]),
                torch.cat([lengths, lengths]), torch.cat([t, t]),
                torch.cat([style, torch.zeros_like(style)]),
                torch.cat([mu, torch.zeros_like(mu)]), **kwargs)
            conditioned, unconditioned = prediction.chunk(2)
            prediction = (1 + cfg_rate) * conditioned - cfg_rate * unconditioned
        else:
            prediction = cfm.estimator(x, prompt, lengths, t, style, mu, **kwargs)
        x = x + (times[step + 1] - times[step]) * prediction
        x.masked_fill_(prompt_mask[:, None, :] | ~valid[:, None, :], 0)
    return [x[i:i+1, :, int(prompt_lengths[i]):int(lengths[i])] / scale
            for i in range(len(items))]


@torch.inference_mode()
def decode_equal_lengths(tensors, decoder):
    """Batch equal lengths: padding changes boundary convolutions in vocoders."""
    groups = defaultdict(list)
    for i, tensor in enumerate(tensors):
        groups[tuple(tensor.shape[1:])].append(i)
    outputs = [None] * len(tensors)
    for indices in groups.values():
        batch = torch.cat([tensors[i] for i in indices])
        decoded = decoder(batch.float())
        if decoded.shape[0] != len(indices):
            raise RuntimeError("Decoder changed batch dimension")
        for row, index in enumerate(indices):
            outputs[index] = decoded[row].reshape(1, -1).clamp(-1, 1).cpu()
    return outputs


def vocode_batch(vocoder, requests, *, backend):
    if not requests:
        return []
    if len(requests) > vocoder.max_batch_size:
        raise ValueError("Batch exceeds configured acoustic cache capacity")
    items = []
    for request in requests:
        kwargs = {}
        if backend == "s2vae":
            kwargs = dict(prompt_features=request["prompt_features"],
                          prompt_feature_length=request["prompt_feature_length"])
        item = vocoder.prepare(request["codes"], request["ref_audio"], **kwargs)
        item["seed"] = int(request.get("seed", 0))
        limit = getattr(vocoder, "max_sequence_length", 8192)
        if item["mu"].shape[1] > limit:
            raise ValueError("Acoustic sequence exceeds estimator cache length")
        items.append(item)
    cfm = vocoder.model.models["cfm"] if backend == "s2vae" else vocoder.s2mel.models["cfm"]
    outputs = flow_batch(cfm, items, steps=vocoder.diffusion_steps,
        cfg_rate=vocoder.cfg_rate, temperature=vocoder.temperature if backend == "s2vae" else 1,
        drop_style=backend == "s2vae")
    decoder = (lambda x: vocoder.audio_vae.decode(x, normalized=True)) if backend == "s2vae" else vocoder.bigvgan
    waves = decode_equal_lengths(outputs, decoder)
    results = []
    for wave, item in zip(waves, items):
        info = dict(item["info"], backend=backend, acoustic_batch_size=len(items))
        info["wav_seconds"] = wave.shape[-1] / info["sample_rate"]
        results.append((wave, info))
    return results
