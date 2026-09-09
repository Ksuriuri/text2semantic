#!/usr/bin/env python3
"""Checkpoint smoke/parity/latency probe. Run HF and vLLM in separate processes."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import torch

import infer


def paths(root):
    codec=root/'indextts-2.5/indextts-2.5/checkpoints'
    return codec, dict(w2v_bert_path=str(codec/'w2v-bert-2.0'), stats_path=str(codec/'wav2vec2bert_stats.pt'))


async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/babysor00'))
    p.add_argument('--mode',choices=['hf','vllm','s2mel','s2vae'],required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    codec, feature_args=paths(args.root)
    texts=['你好，欢迎使用语音合成。','今天天气很好，我们一起出去走走吧。']
    second = Path(args.reference).with_name('ref_02.wav')
    references = [args.reference, str(second) if second.exists() else args.reference]
    if args.mode=='hf':
        model=infer.load_t2s(str(args.root/'t2s/checkpoint-step-16510-ts2-emo'),
                            **feature_args,device='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
        records=[]
        timing=[]
        for text, reference in zip(texts, references):
            infer.generate_codes(model,text,reference, max_new_tokens=8, do_sample=False)
            torch.cuda.synchronize()
            torch.manual_seed(123)
            start=time.perf_counter()
            codes,features,length=infer.generate_codes(model,text,reference,
                return_prompt_features=True,do_sample=False,temperature=1.,top_k=0,
                repetition_penalty=1.,max_new_tokens=200)
            torch.cuda.synchronize()
            records.append(dict(text=text,codes=codes.cpu(),prompt_features=features.cpu(),
                                prompt_feature_length=length,ref_audio=reference,seed=123))
            timing.append(dict(mode="hf", seconds=time.perf_counter()-start, codes=len(codes)))
            print(json.dumps(timing[-1]),flush=True)
        torch.save(records,args.out/'records.pt')
        (args.out/'hf.json').write_text(json.dumps(timing,indent=2))
    elif args.mode=='vllm':
        from qwen_tts.inference.vllm_backend import VLLMSemanticBackend
        backend=VLLMSemanticBackend(args.root/'vllm-speech',**feature_args,gpu_memory_utilization=.55)
        records=torch.load(args.out/'records.pt',weights_only=True)
        async def run(record):
            start=time.perf_counter()
            codes,_,_=await backend.generate(record['text'],record['ref_audio'],temperature=0,
                top_k=0,repetition_penalty=1.,max_new_tokens=200,seed=123)
            return dict(seconds=time.perf_counter()-start,codes=codes.tolist(),
                        exact=bool(torch.equal(codes,record['codes'])))
        await asyncio.gather(*(run(r) for r in records))
        result=await asyncio.gather(*(run(r) for r in records))
        (args.out/'vllm.json').write_text(json.dumps(result,indent=2))
        backend.close()
    else:
        records=torch.load(args.out/'records.pt',weights_only=True)
        common=dict(indextts_root=str(codec.parent),codec_dir=str(codec),device=torch.device('cuda:0'),
                    max_batch_size=8,diffusion_steps=25,cfg_rate=.7)
        if args.mode=='s2mel':
            voc=infer.IndexTTS25Vocoder(**common,bigvgan_dir=str(codec/'bigvgan'))
        else:
            semantic=args.root/'t2s/semantic2any'
            exp=semantic/'exp/s2vae_dit_indextts25_feature_finetune_step462000'
            voc=infer.S2VAEVocoder(**common,semantic2any_root=str(semantic),
                config_path=str(exp/'config.yaml'),checkpoint_path=str(exp/'s2mel_step462000.pth'),
                dots_tts_dir=str(semantic/'checkpoints/dots-tts'),prompt_min_seconds=2)
        singles=[]
        timings={}
        for record in records:
            kwargs={} if args.mode=='s2mel' else {k:record[k] for k in ['prompt_features','prompt_feature_length']}
            voc.vocode(record['codes'],record['ref_audio'],**kwargs)
            torch.cuda.synchronize()
            torch.manual_seed(record['seed']);start=time.perf_counter()
            wav,info=voc.vocode(record['codes'],record['ref_audio'],**kwargs)
            torch.cuda.synchronize(); singles.append(wav)
            timings.setdefault('original_seconds',[]).append(time.perf_counter()-start)
        from qwen_tts.inference.acoustic_batch import flow_batch
        cfm = voc.s2mel.models['cfm'] if args.mode == 's2mel' else voc.model.models['cfm']
        prepared=[]
        for r in records:
            kwargs={} if args.mode=='s2mel' else {k:r[k] for k in ['prompt_features','prompt_feature_length']}
            prepared.append(dict(voc.prepare(r['codes'],r['ref_audio'],**kwargs),seed=r['seed']))
        flow_args=dict(steps=25,cfg_rate=.7,temperature=1 if args.mode=='s2mel' else voc.temperature,
                       drop_style=args.mode=='s2vae')
        flow_multi=flow_batch(cfm,prepared,**flow_args)
        timings['flow_comparisons']=[]
        for r,multi in zip(prepared,flow_multi):
            one=flow_batch(cfm,[r],**flow_args)[0]
            torch.manual_seed(r['seed'])
            kwargs={} if args.mode=='s2mel' else {'drop_style':True}
            original=cfm.inference(r['mu'].clone(),torch.tensor([r['mu'].shape[1]],device='cuda'),
                r['prompt'],r['style'],None,25,temperature=flow_args['temperature'],
                inference_cfg_rate=.7,**kwargs)[:,:,r['prompt'].shape[-1]:]
            timings['flow_comparisons'].append(dict(
                single_original_relative_l2=float((one-original).norm()/original.norm()),
                batch_single_relative_l2=float((multi-one).norm()/one.norm())))
        # Warm new path before measuring steady state.
        voc.vocode_batch(records)
        runs=[]
        for _ in range(3):
            start=time.perf_counter();result=voc.vocode_batch(records)
            torch.cuda.synchronize();runs.append(time.perf_counter()-start)
        timings['batch_seconds']=runs
        timings['comparisons']=[]
        for i,((wave,info),single) in enumerate(zip(result,singles)):
            infer.save_wav(str(args.out/f'{args.mode}-batch-{i}.wav'),wave,info['sample_rate'])
            infer.save_wav(str(args.out/f'{args.mode}-original-{i}.wav'),single,info['sample_rate'])
            comparison=dict(shape=list(wave.shape),original_shape=list(single.shape),finite=bool(torch.isfinite(wave).all()))
            if single.shape==wave.shape:
                comparison.update(max_abs=float((wave-single).abs().max()),
                                  relative_l2=float((wave-single).norm()/single.norm().clamp_min(1e-8)))
            timings['comparisons'].append(comparison)
        timings['throughput_batches'] = []
        for size in (4, 8):
            for mixed in (False, True):
                batch = [dict(records[i % len(records) if mixed else 0], seed=123+i)
                         for i in range(size)]
                voc.vocode_batch(batch)
                elapsed=[]
                for _ in range(2):
                    start=time.perf_counter(); batch_result=voc.vocode_batch(batch)
                    torch.cuda.synchronize(); elapsed.append(time.perf_counter()-start)
                timings['throughput_batches'].append(dict(size=size, mixed=mixed,
                    seconds=elapsed, finite=all(bool(torch.isfinite(w).all()) for w,_ in batch_result),
                    samples=[w.shape[-1] for w,_ in batch_result]))
        (args.out/f'{args.mode}.json').write_text(json.dumps(timings,indent=2))
        print(json.dumps(timings),flush=True)


if __name__=='__main__': asyncio.run(main())
