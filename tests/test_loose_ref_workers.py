import numpy as np
import torch
from torch.utils.data import DataLoader
from finetuning.dataset import Text2SemanticDataset
from finetuning.train import add_speaker_features

class Tok:
    pad_token_id=0
    eos_token_id=1
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[2,3]}

class Mel:
    def __call__(self,waveforms,**kwargs):
        assert all(w.shape==(160,) for w in waveforms)
        return torch.ones(len(waveforms),2,8),torch.ones(len(waveforms),2,dtype=torch.long)

class Encoder:
    def encode_features(self,features,mask):
        return features,mask.sum(-1)
    def encode_files(self,*a,**kw):
        raise AssertionError('GPU rank must not open files')


def test_loose_refs_decode_and_preprocess_inside_workers(monkeypatch):
    rows=[dict(id=str(i),text='hello',duration=1.0,language='en',speaker_id='s',audio=f'{i}.opus',semantic_codes=[1,2]) for i in range(2)]
    def decode(self,path,name):
        assert torch.utils.data.get_worker_info() is not None
        assert path.endswith('.opus')
        return np.zeros(160,dtype=np.float32)
    monkeypatch.setattr(Text2SemanticDataset,'_decode_audio',decode)
    ds=Text2SemanticDataset(rows,Tok(),speaker_mel_extractor=Mel())
    dl=DataLoader(ds,batch_size=2,num_workers=1,collate_fn=ds.collate_fn)
    batch=next(iter(dl))
    assert 'speaker_audio_paths' not in batch
    assert 'speaker_input_features' in batch
    result=add_speaker_features(batch,Encoder(),20)
    assert result['speaker_features'].shape==(2,2,8)


def test_loose_explicit_ref_also_uses_worker_features(monkeypatch):
    rows=[dict(text='hello',duration=1.0,ref_audio='ref.opus',audio='target.opus',semantic_codes=[1,2])]
    monkeypatch.setattr(Text2SemanticDataset,'_decode_audio',lambda self,p,n:np.zeros(160,dtype=np.float32))
    ds=Text2SemanticDataset(rows,Tok(),speaker_mel_extractor=Mel(),min_speaker_records=1)
    batch=ds.collate_fn([ds[0]])
    assert 'speaker_input_features' in batch
    assert 'speaker_audio_paths' not in batch
