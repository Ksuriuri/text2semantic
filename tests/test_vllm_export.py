import json
import torch
import pytest
from safetensors.torch import load_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from test_text2semantic import tiny_model
from qwen_tts.inference.vllm_export import export_checkpoint
from qwen_tts.inference.vllm_backend import PromptEncoder


def test_export_preserves_prefill_and_logits(tmp_path, monkeypatch):
    model = tiny_model().eval()
    source, target = tmp_path/'source', tmp_path/'export'
    model.save_pretrained(source)
    class Tokenizer:
        def save_pretrained(self, path):
            path.mkdir()
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *a: Tokenizer())
    export_checkpoint(source, target)
    config = Qwen3_5TextConfig.from_dict(json.loads((target/'config.json').read_text()))
    backbone = Qwen3_5ForCausalLM(config).eval()
    weights = {}
    for shard in target.glob('model-*.safetensors'):
        weights.update(load_file(shard))
    backbone.load_state_dict(weights, strict=True)
    encoder = PromptEncoder(target, device='cpu', dtype=torch.float32)
    features, lengths = torch.randn(1,5,8), torch.tensor([5])
    ids, prefix = [1,2,3], [4,5]
    expected, mask, positions = model._build_generation_prompt(torch.tensor([ids]),
        torch.ones(1,3,dtype=torch.long), features, lengths, torch.tensor([[16,*prefix]]))
    actual = encoder(ids, features, lengths, prefix)
    torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
    with torch.no_grad():
        reference = model.speech_head(model.backbone(inputs_embeds=expected,
            attention_mask=mask, position_ids=positions).last_hidden_state)
        result = backbone(inputs_embeds=actual[None], attention_mask=mask,
            position_ids=positions).logits
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(backbone.model.embed_tokens.weight, model.speech_embedding.weight)
    with pytest.raises(FileExistsError):
        export_checkpoint(source, target)
