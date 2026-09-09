import asyncio
from types import SimpleNamespace
import torch
import pytest
from qwen_tts.inference.acoustic_batch import flow_batch, decode_equal_lengths
from qwen_tts.inference.batch_queue import MicrobatchQueue


class Estimator:
    def __init__(self):
        self.batches = []
    def __call__(self, x, prompt, lens, t, style, mu, **kwargs):
        self.batches.append(len(x))
        return x * 0.1 + prompt * 0.2 + mu.transpose(1, 2) * 0.3


def item(n, p, seed):
    return dict(mu=torch.ones(1, n, 2), prompt=torch.ones(1, 2, p),
                style=torch.zeros(1, 192), seed=seed)


@pytest.mark.parametrize('cfg', [0, .7])
def test_batch_matches_single_and_preserves_lengths(cfg):
    estimator = Estimator()
    cfm = SimpleNamespace(in_channels=2, estimator=estimator,
                          zero_prompt_speech_token=True, feat_scale=2.)
    items = [item(7, 2, 9), item(11, 4, 5), item(8, 1, 3)]
    batched = flow_batch(cfm, items, steps=4, cfg_rate=cfg)
    assert estimator.batches == [3 * (2 if cfg else 1)] * 4
    for row, value in zip(items, batched):
        single = flow_batch(cfm, [row], steps=4, cfg_rate=cfg)[0]
        torch.testing.assert_close(value, single, rtol=0, atol=0)
        assert value.shape[-1] == row['mu'].shape[1] - row['prompt'].shape[-1]


def test_decoder_groups_without_padding_and_preserves_order():
    shapes = []
    def decode(x):
        shapes.append(tuple(x.shape))
        return x[:, :1].repeat_interleave(2, dim=-1)
    outputs = decode_equal_lengths([torch.ones(1, 2, n)*v for n,v in [(4,.1),(7,.2),(4,.3)]], decode)
    assert shapes == [(2,2,4),(1,2,7)]
    assert [o.shape[-1] for o in outputs] == [8,14,8]
    assert outputs[2][0,0].item() == pytest.approx(.3)


def test_queue_batches_and_flushes_underfilled():
    async def run():
        sizes = []
        def synth(items):
            sizes.append(len(items))
            return [(i['codes'], {}) for i in items]
        queue = MicrobatchQueue(synth, max_batch_size=4, max_wait_ms=10)
        result = await asyncio.gather(*(queue.submit({'codes':torch.ones(3)*i}) for i in range(3)))
        assert sizes == [3]
        assert result[2][0][0] == 2
        await queue.close()
    asyncio.run(run())


def test_queue_budget_and_failure_recovery():
    async def run():
        sizes=[]
        def synth(items):
            sizes.append(len(items))
            if items[0].get('fail'):
                raise ValueError('bad item')
            return [(i['codes'], {}) for i in items]
        queue=MicrobatchQueue(synth, max_wait_ms=1, frame_budget=4)
        results=await asyncio.gather(queue.submit({'codes':torch.ones(3),'fail':True}),
                                     queue.submit({'codes':torch.ones(3)}), return_exceptions=True)
        assert isinstance(results[0], ValueError)
        assert isinstance(results[1], tuple)
        assert sizes == [1,1]
        await queue.close()
    asyncio.run(run())


def test_bad_reference_does_not_fail_other_batch_members():
    async def run():
        def synth(items):
            if any(i.get('bad') for i in items): raise ValueError('invalid reference')
            return [(i['codes'], {}) for i in items]
        queue=MicrobatchQueue(synth, max_wait_ms=10)
        values=await asyncio.gather(queue.submit({'codes':torch.ones(3),'bad':True}),
                                    queue.submit({'codes':torch.ones(3)}), return_exceptions=True)
        assert isinstance(values[0], ValueError)
        assert isinstance(values[1], tuple)
        await queue.close()
    asyncio.run(run())
