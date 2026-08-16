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
**不必**再写目标 flac 路径。可选 `id`：若与某个 member 同名，抽 ref 时会避开。

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
