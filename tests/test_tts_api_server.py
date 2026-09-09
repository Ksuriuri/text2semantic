import sys
from pathlib import Path
from types import SimpleNamespace
import asyncio
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from api_server import create_app


class FakePipeline:
    def __init__(self, args):
        self.active=0
        self.completed=0
        self.queues={}
        self.ready=True
    def healthy(self): return self.ready
    async def close(self): pass
    async def synthesize(self, item):
        if item.synthesis_text == 'bad': raise ValueError('bad text')
        await asyncio.sleep(0)
        return b'RIFFtest', dict(sample_rate=48000,total_ms=2)


def client():
    return TestClient(create_app(SimpleNamespace(request_timeout=1), FakePipeline))


def test_single_json_and_multipart_alias():
    with client() as c:
        assert c.get('/health').status_code==200
        result=c.post('/tts', json={'synthesis_text':'hello','wav_base64':'YQ=='})
        assert result.content==b'RIFFtest'
        assert result.headers['x-sample-rate']=='48000'
        result=c.post('/api/tts', data={'text':'hello'}, files={'ref_audio':('a.wav',b'a','audio/wav')})
        assert result.status_code==200


def test_batch_order_partial_failure_and_validation():
    with client() as c:
        result=c.post('/tts_batch',json={'items':[
            {'synthesis_text':text,'wav_base64':'YQ=='} for text in ['hello','bad','world']]})
        values=result.json()['results']
        assert [r['index'] for r in values]==[0,1,2]
        assert values[1]['status_code']==400
        assert 'audio_base64' in values[2]
        assert c.post('/tts_batch',json={'repeat_num':100000000,'synthesis_text':'hello','wav_base64':'YQ=='}).status_code==400
        assert c.post('/tts',json={'synthesis_text':'hello','wav_base64':'YQ==','duration':5}).status_code==400


def test_auth(monkeypatch):
    monkeypatch.setenv('TTS_API_KEY','test-key')
    with client() as c:
        assert c.get('/health').status_code==200
        assert c.post('/tts',json={}).status_code==401
        assert c.post('/tts',json={'synthesis_text':'hello','wav_base64':'YQ=='},headers={'Authorization':'Bearer test-key'}).status_code==200
