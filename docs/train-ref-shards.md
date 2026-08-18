# 训练 ref shard 布局（默认读取方式）

训练盘不要再放散落 flac。GCS 上先整理成下面这套目录，再整包拉到
Jiuzhang `/hunshan/...`。`finetuning/train.py` 会自动发现
`refs/speaker_index.jsonl` 并按包读参考音频。

## 目录

```
trainsets/t2s-vN/
  manifests/train.jsonl
  manifests/eval.jsonl
  codes/<dataset>/<shard>.u2.bin
  codes/<dataset>/<shard>.jsonl
  refs/speaker_index.jsonl
  refs/shards/refs-000000.tar
  refs/shards/refs-000001.tar
```

- 原 `preprocessed/<dataset>/audio/*.tar` 只当原料，不进训练盘。
- 一个 speaker 的全部 ref **必须**落在同一个 `refs-*.tar` 里，不要拆包。
- 多个 speaker 可以进同一个 tar，按体积封顶（建议 0.5–1 GB），不要按固定条数。
- 打包时写入 `k_max` 条（或该人全部）；训练 `--refs_per_speaker N` 再抽，
  `N=0`（默认）表示用包里全部。

## speaker_index.jsonl

一行一个 `(language, speaker_id)`：

```json
{"language":"zh","speaker_id":"laion_emolia__ZH_B00039_S01897","shard":"refs-000123.tar","members":["laion_emolia__ZH_B00039_S01897/000.flac","laion_emolia__ZH_B00039_S01897/001.flac"]}
```

也支持 `speaker_index.parquet`（需要 pyarrow）。`members` 是 tar 内路径。

## manifest 行

每行仍要 `text`、`speaker_id`、`language`，以及 compact codes：
`semantic_code_path` / `semantic_code_offset` / `semantic_code_length`。
**不必**再写目标 flac 路径。`id` 用来避免拿自己当 ref：member 去掉音频扩展名就是
row id（`ears__p045_emo_adoration.flac` -> `ears__p045_emo_adoration`），两种布局都
一样，前面的 dataset 前缀本来就属于 id，不要去掉。

## 启动

```bash
# 默认：发现 manifests/../refs/speaker_index.jsonl 就走 packed refs
uv run accelerate launch finetuning/train.py \
  --train_jsonl /hunshan/.../manifests/train.jsonl \
  --eval_jsonl /hunshan/.../manifests/eval.jsonl \
  ...

# 显式指定
--ref_index /hunshan/.../refs/speaker_index.jsonl \
--refs_per_speaker 16

# 旧散文件（回退）
--loose_refs
```

没有 index 时会告警并回退到原来的 `audio` / `audio_path` 散文件逻辑。


## 另一种后端：直接读原始 tar（`--ref_backend source_tar`）

打包的 ref 是建 trainset 时定死的那几条。原始 `preprocessed/<dataset>/audio/*.tar`
是未压缩 ustar，按 member 的数据偏移做一次 ranged GET 就只拿到那一条（350 MB 的包里
取 200 KB），所以可以不打包、不复制，直接把**一个 speaker 的全部片段**都当候选：
录得好的人是 300 条候选而不是 8 条。

### 索引

```
trainsets/t2s-vN/refs/source_speaker_index.jsonl
```

```json
{"dataset":"ears","language":"en","speaker_id":"ears__p001","refs":[["preprocessed/ears/audio/ears-000000.tar","ears__p001_emo_adoration_freeform.flac",512,382599]]}
```

key 是 `(dataset, language, speaker_id)` 三段。**不能**省掉 dataset：`p001` 这种
speaker_id 在好几个 dataset 里都有，且不是同一个人。

偏移来自 header 扫描写的 per-tar sidecar：

```
index/source-tar-members/<dataset>/<tar>.json
{"format_version":1,"shard":"ears-000000.tar","shard_size":...,"members":{"ears__p001_a.flac":[512,382599]}}
```

索引由 sidecar 和 manifest join 出来，不读音频：

```bash
uv run python scripts/build_source_ref_index.py \
  --work_dir ~/t2s-source-refs \
  --max_refs_per_speaker 0
# 完成后上传
gcloud storage cp ~/t2s-source-refs/source_speaker_index.jsonl \
  gs://<bucket>/trainsets/t2s-v1/refs/source_speaker_index.jsonl
```

只有 manifest 里的行会成为候选 ref，且只取 `[3, 30]` 秒（和打包时的 ref 窗口一致）：
0.5 秒的片段可以当目标，但没法当条件。tar 的路径按 sidecar 所在的 dataset 目录拼，
不能从 tar 名字猜——`laion_emolia-000869.tar` 在 `preprocessed/laion_emolia_zh/audio/`
下面。

### 启动

```bash
uv run accelerate launch finetuning/train.py \
  --ref_backend source_tar \
  --ref_source_bucket <bucket> \
  --ref_index /path/to/refs/source_speaker_index.jsonl \
  ...
# 已经有本地镜像时用 --ref_source_root /mnt/... 代替 --ref_source_bucket
```

注意：换后端会改变 speaker key 和 ref 集合，manifest row index 会重建一次。
