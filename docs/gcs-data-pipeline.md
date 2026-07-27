# GCS → text2semantic 数据处理流程

本文档描述如何从 `gs://noiz-taiwan-audio-data/preprocessed/` 的原始语料，产出
`finetuning/dataset.py::Text2SemanticDataset` 可以直接训练的 manifest。

代码在 `data_pipeline/`，权重下载脚本在 `scripts/download_models.py`，
RepCodec 架构配置在 `configs/repcodec_semantic.yaml`。全部落盘在本项目内，
不依赖任何外部项目的代码或配置。

---

## 1. 数据源

### 1.1 访问方式

- Bucket：`gs://noiz-taiwan-audio-data`，前缀 `preprocessed/`
- Project：`noiz-430406`，Region：`asia-east1`
- 认证：Service Account `taiwan-audio-rw@noiz-430406.iam.gserviceaccount.com`
- Bucket 开启了 **Uniform Bucket-Level Access**，所以必须用 SA 凭据，
  签名 URL / ACL 那一套不可用。

凭据通过两种方式之一提供（**永远不要把 key 文件提交进仓库**，
`.gitignore` 已经忽略 `gcs-key.json` / `key.json`）：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json
# 或者给每个命令加 --gcs-key /path/to/gcs-key.json
```

依赖：`google-cloud-storage`（已加入 `pyproject.toml`）。

### 1.2 目录结构

```
preprocessed/<dataset>/
    metadata/<dataset>-NNNNNN.jsonl        每条一个 utterance
    audio/<dataset>-NNNNNN.tar             flac 音频，未压缩打包
    asr/<engine>/<dataset>-NNNNNN.jsonl    ASR 识别结果
```

`metadata` / `asr` / `audio` 用**同一个 `NNNNNN` 分片号**，并且共享同一套 `id`，
所以按 shard 号 join 即可，不需要建全局索引。这是整个流程能并行的基础。

`metadata` 行的字段（**注意：所有值都是字符串**，缺失值被导出成字面量 `"None"`）：

```json
{
  "id": "expresso__ex01_confused_00001",
  "audio_path": "audio/expresso-000000.tar/expresso__ex01_confused_00001.flac",
  "text": "Why are you beating up my jukebox?",
  "speaker_id": "expresso__ex01",
  "language": "en",
  "source": "read/train-00000-of-00012.parquet#ex01_confused_00001.wav",
  "duration": "2.7479583333333335",
  "sample_rate": "48000"
}
```

`audio_path` 是一个伪路径，指向 shard tar **内部**的成员名，
`.tar/` 之后的部分就是 tar member name（见 `gcs.tar_member_name`）。

`asr` 行的字段：`completed_at, dataset, id, language, metadata_path, status, text`，
`status` 正常值为 `transcribed`。

### 1.3 ASR 引擎覆盖情况

| 引擎 | 覆盖的数据集 |
|---|---|
| `cohere-transcribe-03-2026` | **全部 10 个** |
| `granite-speech-4.1-2b-nar` | 仅 ears, expresso, hi_fi_tts, noiz-short, vctk, worldspeech |

Genshin / StarRail / WutheringWaves / laion_emolia **只有 cohere**。
这直接决定了下面 ASR 过滤的分支走向。

两个引擎的输出风格不同（Granite 全小写、无句末标点；Cohere 带大小写和标点），
所以计算错误率前必须归一化，否则完全一致的转写也会算出很高的 WER。

### 1.4 规模

metadata + ASR 的 jsonl 合计 **约 75 GB**（metadata 39.3 + cohere 34.2 + granite 1.5），
音频 tar 合计 **约 20.3 TB**，共 42085 个 shard、约 8420 万条 utterance。
其中 `laion_emolia` 一家占 39484 个 shard、69 GB 文本、18.5 TB 音频。

**这个量级决定了流程设计**：统计阶段只读文本不碰音频，且流式处理不落盘原始文本。

---

## 2. 训练侧的数据契约

`Text2SemanticDataset` 对每行 manifest 的要求（细节见 `finetuning/dataset.py`）：

| 字段 | 说明 |
|---|---|
| `text` | 必需，文本条件 |
| `semantic_code_path` + `semantic_code_offset` + `semantic_code_length` | 语义 token，从打包的 `<u2` (uint16 LE) 文件里 memmap 切片读取。也支持内联 `semantic_codes` 数组，但不适合大规模 |
| `speaker_id`（配合 `language`） | 说话人分组键是 **`(language, speaker_id)` 元组**，不是 `speaker_id` 单独 |
| `audio` / `audio_path` | 目标音频路径 |
| `ref_audio` / `ref_audio_path` | 可选，显式指定 speaker reference |

有三条容易踩的隐性规则：

1. **speaker reference 必须是同一 speaker 的另一条音频**（不能等于目标音频本身）。
   找不到 reference 的行会被 `_is_usable` **静默丢弃**。
2. 默认 `min_speaker_records=2`，即每个 `(language, speaker_id)` 至少要有 2 条记录。
3. 默认 `max_target_seconds=30.0`，超过会被丢。

语义 token 词表：RepCodec 索引范围 `[0, 8191]`，特殊 token
`BOS=8192`、`EOS=8193`、`PAD=8194`。

**为什么必须保留一部分音频**：speaker 特征是训练时从 reference 音频**在线**提取的，
而且 speaker encoder 是可训练的，所以不能离线把 latent 冻结下来。
因此 encode 阶段用 `--reference-clips-per-speaker` 每个说话人保留少量（默认 2 条）音频，
而不是保留全部 20 TB。

---

## 3. 过滤条件

### 3.1 采样率 ≥ 22.05 kHz

`sample_rate < 22050` 的行丢弃。实测各数据集：Genshin/expresso/vctk/ears 48k，
WutheringWaves/hi_fi_tts 44.1k，laion_emolia/worldspeech 24k，
StarRail 混合（44.1k 为主，含 12k/24k/32k/36k/48k），
noiz-short 混合（含 16k，会被这一条砍掉）。

### 3.2 时长与说话人支撑

保留满足全部三条的行：

1. `3s < duration < 30s`（开区间，两端都不含）
2. 该 speaker 还有**至少一条其他**通过过滤的音频（即 `>= 2` 条）
3. 该 speaker 的音频中**至少有一条 > 6s**

第 1 条是逐行判断，在流式扫描时完成；第 2、3 条需要全语料的统计，
所以作为第二遍（`scan.speaker_gate`）在索引上跑。
两遍都以 `(language, speaker_id)` 为键，和训练侧保持一致。

> 注意：第 2、3 条判断的是**过滤后**的集合。也就是说采样率或时长被砍掉的音频
> 不能用来充当"其他音频"或"那条 >6s 的音频"。这是有意的 —— 训练时真正能被
> 当作 reference 的只有留在 manifest 里的行。

### 3.3 ASR 一致性（WER/CER < 0.5）

逐行判断，分三种情况：

| 情况 | 参考 (reference) | 假设 (hypothesis) |
|---|---|---|
| metadata 有原始文本 | 原始文本 | cohere |
| 无原始文本，有 granite | granite | cohere |
| 无原始文本，也无 granite | — | **丢弃** |

第三种情况无法判定质量，默认丢弃（`--keep-unscored` 可以改为保留并标记
`asr_scored=false`）。实测受影响的主要是 ears（约 28.5% 的行 `text` 为空），
但 ears 有 granite，所以 fallback 能生效。

**指标选择**：空格分词的语言用 WER，CJK（zh/ja/ko/yue 等）用 CER。
不使用 `min(WER, CER)` —— 取二者中更宽松的那个会放进对齐很差的样本。

**归一化**：NFKC → 转小写 → 去标点（含全角）→ 折叠空白。
这样 `"Why are you beating up my jukebox?"` 与 `"why are you beating up my jukebox"`
的错误率为 0，而不是被标点和大小写误判。

编辑距离用纯 Python DP 实现（`filters._levenshtein`），
不引入 `jiwer` / `rapidfuzz` 依赖。

---

## 4. 运行流程

### 4.1 下载模型权重

```bash
python scripts/download_models.py --out-dir checkpoints/maskgct
# 国内网络：
HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --out-dir checkpoints/maskgct
```

产出布局：

```
checkpoints/maskgct/
    w2v-bert-2.0/{config.json,preprocessor_config.json,model.safetensors}   ~2.3 GB
    wav2vec2bert_stats.pt                                                   ~9 KB
    semantic_codec/{config.yaml,model.safetensors}                          ~177 MB
```

来源（均为 HF Hub 公开仓库）：

| 资产 | 来源 |
|---|---|
| W2V-BERT 2.0 speech encoder | <https://huggingface.co/facebook/w2v-bert-2.0>（`config.json` / `preprocessor_config.json` / `model.safetensors`） |
| RepCodec 语义量化器 | <https://huggingface.co/amphion/MaskGCT>（`semantic_codec/model.safetensors`） |
| 特征均值/方差 `wav2vec2bert_stats.pt` | <https://huggingface.co/IndexTeam/IndexTTS-2>（仓库根目录） |

> `wav2vec2bert_stats.pt` **不在 amphion/MaskGCT 仓库里**（已核对该仓库只有 9 个文件，
> 无此文件），它由 IndexTTS-2 分发。也可以用 `--stats-from <本地路径>` 从已有的
> IndexTTS-2 checkpoint 目录复制。

`semantic_codec/config.yaml` **上游没有发布**（amphion/MaskGCT 把架构写死在代码里），
所以由本项目的 `configs/repcodec_semantic.yaml` 提供。其中每个数值都对着
checkpoint 的张量形状核对过（见该文件注释），且 `load_model(..., strict=True)`
会在任何字段漂移时直接报错，不会静默加载错模型。

### 4.2 扫描 + 统计

```bash
python -m data_pipeline.scan \
    --out-dir runs/full \
    --gcs-key /path/to/gcs-key.json \
    --workers 16

python -m data_pipeline.stats --run-dir runs/full --gcs-key /path/to/gcs-key.json
```

`scan` 做的事：

- 列出所有 shard，按 shard 号把 metadata / cohere / granite 三个 jsonl join 起来
- 每个 shard 交给一个 worker 进程：下载到内存 → 解析 → 应用 3.1/3.2.1/3.3 的逐行门槛
  → 把存活行写成一个 gzip 精简索引（每 shard 一个文件）
- 全部 shard 完成后跑 `speaker_gate`，即 3.2 的第 2、3 条
- 产出 `scan_report.json`（各数据集的逐项丢弃计数、采样率分布）、
  `gate_report.json`、`filtered_index.jsonl.gz`

设计取舍：

- **不下载音频**。统计只需要 metadata 里的 `duration`，20 TB 完全不碰。
- **不落盘原始 jsonl**。75 GB 文本在内存里过一遍就丢，只留精简索引。
  这一点很关键 —— 如果把原始文本写到磁盘，会把同机训练任务的 page cache 冲掉。
- **一次一个 shard**，峰值内存就是一个 shard（约 2000 行）× worker 数。
- 索引里**不带文本**。带上就等于把 75 GB 搬进索引，失去意义。
  encode 阶段按 shard 重新读回文本（`encode.shard_texts`）。
- `--skip-existing` 支持断点续跑；`--gate-only` 可以只重跑说话人门槛
  （改 `--speaker-long-seconds` 之类的参数时不必重扫）。
- `--max-shards-per-dataset 2` 用来做小规模 dry-run。

`stats` 做的事：把两份 report 合成"多少小时 / 占多少空间"。空间分三块：

- **音频**：用各数据集 `audio/` tar 的总字节数除以该数据集音频总时长，
  得到平均 flac 码率，再乘存活时长。这样不必读 20 TB 就能估出字节数。
- **语义 token**：MaskGCT 帧率 50 Hz（20 ms 一帧）、每帧 1 个 `uint16`，
  即 **每音频秒 100 字节**。
- **manifest**：直接量。

`lang_stats` 回答的是另一个问题：**给定要用的数据集子集，覆盖哪些语言、每语言多少小时、
要占多少盘**。它也只读索引：

```bash
python -m data_pipeline.lang_stats \
    --run-dir runs/full \
    --audio-sizes runs/audio_sizes.json \
    --datasets Genshin StarRail WutheringWaves ears expresso hi_fi_tts vctk \
    --out runs/full/lang_stats_trusted7.json
```

两个不能省的点：

- **说话人门槛要按子集重跑，不能复用全量的 `gate_report.json`。** 门槛按
  `(language, speaker_id)` 计数，一个说话人可能是靠跨数据集凑够 2 条才留下的；
  排掉某些数据集后它就该掉。（对这 7 个数据集实测复跑与全量同为 11,484 个说话人，
  即没有这种跨集凑数的情况 —— 但这是量出来的结论，不是默认成立的。）
- **区分"必须常驻"与"只是流过"**。训练把 codes mmap 进来、并在线从 reference 音频
  抽 speaker 特征，所以常驻成本是 `codes + reference`；"全部音频"那一列是 encode
  阶段下载后即丢的量，属一次性网络成本，不是存储成本。

### 4.3 编码落盘

```bash
python -m data_pipeline.encode \
    --index runs/full/filtered_index.jsonl.gz \
    --out-dir /mnt/data_3t_1/t2s_train \
    --model-dir checkpoints/maskgct \
    --gcs-key /path/to/gcs-key.json \
    --device cuda:0
```

每个 shard：流式拉一次 audio tar 只解出存活成员 → 跑 MaskGCT tokenizer →
codes 追加进打包的 `<u2` 文件并记录 `offset` / `length` →
只有被选为 speaker reference 的音频才复制留存。

reference 的选法是取每个 speaker **最长**的若干条（3.2 的第 3 条已经保证
每个 speaker 至少有一条 > 6s，所以最长的那条一定够长），**再加一个相邻性约束**
（`--min-w-distance`，默认 4，`data_pipeline/pairing.py`）：

只按"最长"挑会挑到**同一段连续语音的相邻切块** —— 一段长语音被切成若干条同样很长的
相邻窗口，于是最长的两条往往紧挨着。laion 的 `B_S` 半区 **85.5% 的组本身就是一整段
连续切片**（见 6.1.2），此时 reference 与 target 几乎是同一句话的前后两块，
**训练出来是拷贝 prompt，而不是把音色迁移到新内容上**。

utterance id 尾部的 `_W<n>` 沿源档案单调递增，据此可以免费拿到相邻性：

- **`--min-w-distance`（结构约束，免费）**：同一 speaker 的两条必须在源窗口序号上相距
  ≥ 该值。`0` 关闭约束、退回纯最长优先。
- **`max_cosine`（语义兜底，默认 0.98）**：给**没有 `_W`** 的数据集（其余 7 个）以及
  上面约束满足不了时用 —— 相似度过高即视为同一段录音而非同一说话人。
  它需要 embedding，所以以 `filter_pairs_by_similarity(pairs, cosine_of)` 的形式
  交给持有 embedding 的一侧（S2mel 配对）调用，不在 encode 内部跑模型。

三个刻意的设计决定：

1. **没有 `_W` 返回 `None`，不是 0、也不是无穷。** 当成 0 会静默拒掉那 7 个数据集的
   全部配对；当成无穷会静默放过全部。`None` 表示"无相邻性信息"，由调用方转用 `max_cosine`。
2. **约束满足不了时退让，而不是少给 reference —— 但这条会反号。** 全连续的 3 条组无法满足
   距离要求，若因此只返回 1 条 reference，训练侧 `_is_usable` 会把该说话人**静默丢弃**，
   所以 `spread_reference_clips` 默认兜底补齐条数（`backfill=True`），代价交给 `max_cosine`。
   **但对 laion 的连续 slice 组，退让可能比丢弃更差**：补上来的那条是 target 的近邻复读，
   低价值这一点由 6.1.2 的相邻性结构（85.5% 连续、密度 1.0、4177 组）**独立支撑**，
   与 `max_drop` 的迁移比值无关 —— 后者在 12 组下判不出方向（见 6.1.5），
   所以"纯度测不出来"目前只是合理猜测，不是结论。低价值这一条本身就足以支持"排除而非退让"。
   所以提供 `--drop-consecutive-groups`（`backfill=False` + `is_consecutive_run`）：
   连续组**排除而非退让**。**默认关闭**，因为它会丢数据，只在保留 slice 时才该开。
   `is_consecutive_run` 对没有 `_W` 的 id 返回 False，所以那 7 个可靠数据集不会被误删。
   代价打印在 `speakers_dropped_consecutive`。
3. **约束的代价必须打印出来**（`[encode] pairing constraint: {...}`）：
   `speakers_without_usable_pair` 表示该说话人只剩 reference、贡献不了配对。
   这正是 slice 半区"低价值"而非"有噪声"的量化体现。

两个半区都加这条约束 —— **diar 半区不豁免**，它也有 26.2% 的组是连续段。

产出：

```
<out-dir>/
    codes/<dataset>/<dataset>-NNNNNN.u2.bin
    manifest/<dataset>/<dataset>-NNNNNN.jsonl
    reference_audio/<dataset>/*.flac
```

关于 tokenizer 的两个注意点：

- W2V-BERT 取**第 17 层** hidden state，且必须 **FP32**
  （`semantic_codec.py` 里显式 `autocast(enabled=False)`）。混精度会让 code 漂移。
- 音频以 **16 kHz 单声道**载入（`librosa.load(sr=16000, mono=True)`）。
  所以 3.1 的 22.05 kHz 门槛不是给 tokenizer 用的，
  而是保证源录音质量 —— 上采样上来的低采样率音频不会因此变好。
- `MaskGCTFeatureExtractor.encode_files` 默认 `max_audio_seconds=15.0` 会截断，
  但 `encode_file` 传的是 `None`，不截断。30 秒的音频会完整编码成约 1500 个 token。

---

## 5. 已知的数据问题

1. **worldspeech 和 noiz-short 的 `speaker_id` 100% 是 `"None"`**（无说话人标注）。
   在 3.2 的规则下这两个数据集**一条都留不下**。即使放宽过滤，训练时也会因为
   配不出 speaker reference 而被 `_is_usable` 静默丢弃。
   worldspeech 有约 390 万行 / 1.48 TB，是第二大的数据集，需要业务上决策
   （放弃，还是从 `source` 字段的 parquet 分组推断说话人，或跑说话人聚类）。
2. **ears 约 28.5% 的行没有原始文本**，走 granite-vs-cohere 分支。
3. **noiz-short 含 16 kHz 音频**，被 3.1 砍掉。
4. **StarRail 采样率混杂**，含少量 12k/24k/32k/36k，会被 3.1 砍掉一小部分。
5. 上游导出把缺失值写成**字符串 `"None"`**，不是 `null`，也不是缺字段。
   `filters.is_nullish` 统一处理这一点；自己写解析代码时务必注意。
6. 所有 metadata 字段值都是**字符串**，包括 `duration` 和 `sample_rate`。

## 6. 全量实测结果（2026-07-26）

全量扫描 42085 个 shard、0 错误，过滤后：

| 项 | 数值 |
|---|---|
| 条数 | 76,866,229 |
| 总时长 | **197,053.6 小时** |
| 说话人 `(language, speaker_id)` | 2,658,747 |
| 语义 token（打包 uint16） | 70.9 GB |
| reference 音频（每说话人 2 条） | 约 1.25 TB |
| 全部音频（若整套落盘） | 18.14 TB |

分数据集过滤后时长：laion_emolia 195419.5、Genshin 887.8、StarRail 466.9、
WutheringWaves 113.1、hi_fi_tts 68.8、ears 62.4、vctk 28.0、expresso 7.1，
**noiz-short 和 worldspeech 为 0**。

各条规则实际砍掉的量：采样率 7,395 条；时长不在 (3s,30s) **727,279 条**（最主要的一刀）；
无 speaker_id 3,666,446 条（全部是 worldspeech）；ASR 错误率 ≥0.5 共 1,691,954 条；
无法评分 1,256 条；说话人门槛再砍单条说话人 693,863 条 + 无 >6s 长句 504,521 条。

### 6.0 只用 7 个可靠数据集时的语言与空间（2026-07-26，`lang_stats`）

排除 `laion_emolia` 后，剩下 7 个 `speaker_id` 为真实角色/人物标注的数据集
（`data_pipeline.lang_stats` 默认的 `TRUSTED_DATASETS`）：

| 语言 | 条数 | 时长 | 说话人 | 来自 |
|---|---|---|---|---|
| en | 284,213 | **550.6 h** | 3,108 | 全部 7 个 |
| ja | 185,536 | **424.6 h** | 2,911 | Genshin, StarRail, WutheringWaves |
| zh | 175,587 | **330.6 h** | 2,743 | Genshin, StarRail, WutheringWaves |
| ko | 178,267 | **328.4 h** | 2,722 | Genshin, StarRail, WutheringWaves |
| **合计** | **823,603** | **1,634.2 h** | **11,484** | |

**语言覆盖只有 4 种：en / ja / zh / ko。** ja/zh/ko 全部来自三个游戏数据集
（同一批角色的多语配音轨），en 另有 hi_fi_tts / ears / vctk / expresso 的真人录音。
说话人数 11,484 之所以远大于角色数，是因为 key 是 `(language, speaker_id)` ——
同一角色的 4 条语言轨算 4 个说话人（约 2,900 角色 × 4 语言 + 230 个真人）。

空间（每说话人 2 条 reference）：

| 项 | 大小 |
|---|---|
| 语义 codes（打包 uint16） | **0.59 GB** |
| reference 音频 | **7.71 GB** |
| **训练常驻 = codes + reference** | **≈ 8.3 GB** |
| 过滤后全部音频（encode 阶段流过，不落盘） | 268.5 GB |

即这一档**磁盘完全不是约束**（全量 laion 那档常驻 1.32 TB，差两个数量级）。
分数据集：Genshin 887.8 h / StarRail 466.9 h / WutheringWaves 113.1 h /
hi_fi_tts 68.8 h / ears 62.4 h / vctk 28.0 h / expresso 7.1 h。

### 6.1 laion_emolia 的 speaker_id 有两套体系，各约一半（重要）

`laion_emolia` 的 `speaker_id` **不是一种命名，而是两种**，抽 60 个 shard /
117,340 行实测各占约一半，且**语言分布完全不重叠**，说明是两条不同的预处理管线
灌进同一个数据集：

| 体系 | 形如 | 占比 | 语言 | 含义 |
|---|---|---|---|---|
| `slice` | `laion_emolia__EN_B00008_S03560` | 53.4% | zh, en | `语言_批次B_片段S`，**切片分组**，不含说话人主张 |
| `diar` | `laion_emolia__EN_C7eC3wB9HKU_SPEAKER_01` | 46.6% | en, fr, ja | `<youtube视频id>_SPEAKER_NN`，**pyannote 说话人分离标注** |

> 早期文档只描述了 `B_S` 一种，那是从命名推断的，对 `SPEAKER_NN` 那 46.6% 是**错的** ——
> 它确实是说话人标注。另外这些 shard 的 `source` 字段是 `None`，所以
> `worker_4/DE_B00000_S00000_W000000.mp3` 那条证据只覆盖 slice 半区，不能外推。

**两套体系必须分开统计和分开决策**，混在一起平均等于把"真实说话人标注"和
"切片产物"取平均，得到的数字两者都不描述。

#### 6.1.2 两个半区的行为完全相反（`_W<n>` 相邻性实测）

utterance id 尾部的 `_W<n>` 沿源档案单调递增，据此可以看每个分组的成员
在源里的位置分布（纯索引，不碰音频）。60 shard / 4177 组：

| 体系 | 组数 | 完全连续的组 | 最长连续段/组大小 | 密度 | 占该源全部片段 |
|---|---|---|---|---|---|
| `slice` | 2160 | **85.5%** | **1.000** | **1.000** | 0.128 |
| `diar` | 2017 | 26.2% | 0.500 | 0.808 | **0.974** |

- **`slice`**：85.5% 的组是一段**毫无间隔的连续切片**（密度 1.0），只占源档案 12.8%。
  即"把一条长音频顺序切块，每块一个 id"。**组内同人是平凡成立的**，但组内
  音色/内容多样性≈0：拿它做 speaker reference / 同说话人配对，prompt 与 target
  近乎同一段话的相邻两块，会**教模型拷贝而不是迁移音色**。
- **`diar`**：只有 26.2% 连续，却覆盖源档案 **97.4%** 的片段 → **跨整个视频追踪
  同一个人**，中间断开、与他人交错。同人跨多样内容，是真正可用的配对来源。

> **重要的度量选择教训**：speaker embedding（CAMPPlus 等）设计上就对内容/韵律**不变**，
> 只保留说话人身份。所以它**在结构上无法区分**"同人跨多样内容"（好）与
> "同人近乎复读"（差）。`slice` 半区的组内相似度**更高**、跨组相似度**更低**，
> 恰恰是低多样性的伪影，不能读作"配对质量更好"。判断多样性必须用相邻性这类
> 与 embedding 正交的指标 —— 样本量再大也纠正不了度量选错。

结论：**降权/暂不使用 `slice` 半区，原因是组内多样性≈0，而不是它的相似度数字差。**
若要救 `slice`：同一源档案内不同 `S` 组之间相似度中位数仅 0.10–0.12，说明不同 `S`
确实是不同人，即说话人信息没丢、只是**分组粒度太细**（每组仅占源 12.8%），
可在源档案内做聚类把同人的多个 `S` 合并 —— 属于独立工程。

`diar` 半区也有 26.2% 的连续组，同样低多样性，建议配对采样时加"组内最小时间距离"约束。

#### 6.1.3 原先的担心（保留，仍适用于 slice 半区）

这是**切片分组**，既不保证同一分组内是同一个人，也不保证不同分组不是同一个人。

它占了过滤后总时长的 99.2%，所以这一点直接决定数据可用性：

- 分组内混人 → speaker reference 音色给错，训练信号被污染。
- 同一人散在多组 → 按说话人切 train/eval 会漏，零样本评估虚高。

其余 7 个数据集（Genshin / StarRail / WutheringWaves / ears / expresso / hi_fi_tts / vctk）
的 `speaker_id` 是真实角色/人物标注，没有这个问题，但合计只有 **1634 小时 /
11,484 个说话人 / codes 0.6 GB / reference 音频 7.7 GB**。

**数据量选择因此有三个档，不是两个**：

| 方案 | 量级 | 说话人标签可靠性 |
|---|---|---|
| 只用 7 个可靠数据集 | 1,634 h | 真实角色/人物标注 |
| **+ laion 的 `diar` 半区** | **约 9 万 h 量级** | pyannote 说话人分离，跨视频追踪同一人 |
| 全量 laion | 195,419 h | 其中约一半是连续切片，组内多样性≈0 |

建议：先用 7 个可靠数据集跑通 encode 与首版训练，同时用
`hhy_probes/speaker_purity_probe.py`（CAMPPlus embedding，**按体系分层**）
把 `diar` 半区的纯度测实，再决定纳入范围。

#### 6.1.4 抽样探针的设计要点

单看"组内相似度高"无法下结论 —— 同源档案的片段共享录音信道、房间与底噪，
speaker embedding 会把这些一起编码，"同信道"会伪装成"同音色"。所以探针同时给出
上限（真实同说话人）与下限（真实不同说话人）基线，并加**同源桶、不同分组**的对照组
（信道相同、说话人大概率不同）。对 `diar` 半区，同源桶就是**同一个视频**，
而同一视频里的两个 `SPEAKER_NN` 正是 pyannote 在断言"这是两个不同的人"，
所以那里分数偏高等于 **over-cluster（一个人被拆成多个 label）** 的直接证据。

三个必须记住的方法论陷阱：

1. **均匀抽样测不到对照组。** 一个视频只有 2–3 个 label，而全库有约 10^5 个视频，
   随机抽十几组几乎不可能抽到同一视频的两个 label —— 实测 `diar` 对照组
   **0 对**。`--pair-frac`（默认 0.4）预留部分预算在同源桶内**成对抽**，
   只改"抽哪些组"，不改"测什么"。
2. **最近邻相似度随候选池大小单调升高**（10 组时每组 9 个候选，400 组时 399 个），
   所以方向 B 的主判断用对池大小不敏感的**跨组分布分位数**，最近邻率仅作参考。
3. **embedding 对内容/韵律不变**，故不能用它判断组内多样性（见 6.1.2）。

两个独立的失效方向，危害不同：

- **方向 A —— 组内是否混人**（speaker reference 是否给错音色 / 配对是否同人）：
  由对照组直接测。两次分层 smoke 都显示两个半区的 gap 都很大（+0.45 / +0.53），
  组内换人率 0。
- **方向 B —— 同一个真人是否散落在多个分组**（train/eval 能否真按说话人切开，
  直接决定零样本评估是否虚高）：用组质心测**跨源桶**的组间相似度与每组最近邻，
  阈值取**实测**的真实同说话人 p10。输出 `group_merge_candidates.merge_pairs`
  可直接消费：切分前把 `speaker_a`/`speaker_b` 当同一个人即可（对称、未做传递闭包，
  需要簇请自行 union-find —— 闭包会随阈值膨胀成巨簇，属于切分策略而非测量的决定）。

> `diar` 半区的 `src_share` 中位数 0.974，说明**多数视频只产出一个主说话人分组**
> （以单人讲话视频为主）。因此"同视频不同 label"的对照对**天然稀少**，
> 该数字即使在大样本下 pairs 也不多，报告时必须并排给出 pairs 数。

4. **尾巴占比必须带置信区间，否则会给出反向结论。** 尾巴占比 = "超过阈值的对数比例"，
   而阈值取自真实同说话人的 p10，所以它同时受两种抖动影响，探针把两个都报出来：
   - **阈值抖动**（`thr-CI`）：p10 的 bootstrap 置信区间。注意**必须按说话人重采样，
     不能按对重采样** —— 同一说话人的 C(clips,2) 对共享同一嗓音/房间/麦克风，不是独立样本，
     按对做会把 60 人×10 对当成 600 个独立观测，CI 窄好几倍。
   - **对照组抽样抖动**（`sampling-CI`）：固定阈值、对对照组重采样。对照组只有 16–38 对，
     这个区间宽是**结论本身**，不是缺陷。

   实测代价：真实基线从 48 对（12 说话人）扩到 240 对（24 说话人）后，p10 从 **0.4725
   升到 0.6467**，两个尾巴判定**全部反向**：

   | 基线规模 | 阈值 p10 | diar 尾巴 | slice 尾巴 |
   |---|---|---|---|
   | 48 对 / 12 人 | 0.4725 | 8/16 = **50.0%** → "over-cluster likely" | 0/38 = **0.0%** |
   | 240 对 / 24 人 | 0.6467（CI [0.5907, 0.6718]） | 0/16 = **0.0%** | 13/38 = **34.2%**（sampling-CI 18.4%–50.0%） |

   即"48 对基线得出的 diar over-cluster"是**阈值过低造成的伪影**。因此探针现在在两个
   CI 的**并集**跨过 10% 判定线时直接输出 `UNDETERMINED`，不再给结论。

**结论：smoke 规模太小（对照组仅 16–38 对）不足以定论，必须跑
`--groups 400 --real-groups 60 --clips 5` 的正式版本再改写 6.1 的结论。
正式版基线约 60 人 × C(5,2) = 600 对，阈值才够稳。**

#### 6.1.5 预先钉死的判据（pre-registered，@algo-dev msg `5c33ca5e`）

因为 smoke 的数字反复翻转，判据在**看到结果之前**就定好，结果一落地直接执行，不再回头争：

| # | 指标 | 判据 | 落到哪个字段 |
|---|---|---|---|
| 1 | 组内混人率（方向 A，唯一"难修"的红线） | 两个半区的 **CI 上界 < 5%** 才认为配对纯度可用 | `laion_strata[s].intra_group_mixing.rate_ci95` + `mixing_red_line_pass` |
| 2 | diar over-cluster 尾巴 | UNDETERMINED 或低位 → 不合并，按 label 直接用；CI 下界显著 → 用 `merge_pairs` 合并。**两种都不阻塞 diar 可用性** | `within_source_tail_share` + 两个 CI |
| 3 | slice 源内聚类 | CI 下界 **> 15%** → 值得排聚类工程；CI 上界 **< 5%** → 放弃 slice 于配对；中间 → 不做聚类，仅在数据量不够时用连续对兜底 | `within_source_merge_potential.per_scheme.slice` |
| 4 | 采样约束（组内最小 `_W` 距离 + 最小 1-cosine） | 与数据量方向无关，两个半区都加，**不等结果** | 已实现，见 `data_pipeline/pairing.py` |

**判据 1 的统计功效必须先算，否则红线不可判。** 混人率是 0/1 比例，点估计不能判上界：
0/12 组的 95% 上界约 **24%**，不是"< 5%"。用 Wilson 区间（**不能用 bootstrap** ——
对全 0 样本重采样得到的全是 0，会报 [0,0] 假装通过）算出达到 5% 上界所需组数：

| 实测混人组数 | 达到 CI 上界 <5% 所需总组数 |
|---|---|
| 0 | 74 |
| 1 | 110 |
| 2 | 142 |
| 4 | 202 |
| 5 | 230 |

`--groups 400` 的功效：0 混人 → 上界 0.95%；4 → 2.5%；8 → 3.9%；**12 → 5.2%（刚好不过）**。
即 400 组下红线在 **≤11 个混人组**时通过。这就是选 400 而不是更小的量化理由。

**`switch_drop` 用已知纯的真实说话人组标定，不能手拍。** 混人率对这个切点极敏感
（diar 从 0.10 的 16.7% 掉到 0.20 的 0%）。vctk/ears 的组是真实同说话人，
所以它们被 flag 就是**误报**，据此给每个候选切点一个实测误报率（FPR）：

| `switch_drop` | 已知纯组的 FPR（24 组） |
|---|---|
| 0.05 | 50.0% |
| 0.10 | 29.2% |
| 0.15 | **8.3%** ← 原先手拍的默认值，超标 |
| 0.20 | **4.2%** ← 标定选中 |
| 0.25 | 0.0% |

探针取**满足 FPR 目标（`--max-false-positive`，默认 5%）的最小切点** —— 越小的切点对真实混人
越敏感，所以要"仍可信的最灵敏点"，不是"最安全点"。laion 的混人率在这个切点上读出
（`mixing_at_calibrated_cut`），而不是在我手拍的阈值上读。

实测结果：**原默认 0.15 的 FPR 是 8.3%，超标**；标定后选 0.20（FPR 4.2%）。
在 0.20 上 diar 的那 1/12 个 flag 消失 → 它**大概率本来就是误报**。
若连最宽的切点都达不到 FPR 目标，探针输出 `warning` 并拒绝选点 ——
那说明该 clip 数下这个指标本身太吵，应该加 `--clips` 而不是放宽目标。

探针同时保留 `rate_at_drop`（四个切点的混人率）以展示敏感度。

##### 标定能不能跨域迁移 —— 实测方向与直觉相反

在 vctk/ears 上标定、拿到 laion 上用，隐含假设**两边组内声学方差可比**。这个假设不必争，
可以直接测：`max_drop` 量的就是组内离散度，比一下两个域的分布即可。方向决定它是否 fail-safe：

- laion 方差 **>** real → 同一切点在 laion 上更容易触发 → 混人率**高估** → 对红线 #1 **fail-safe**
- laion 方差 **<** real → 同一切点在 laion 上更难触发 → 混人率**低估** → **不 fail-safe**，
  红线可能靠"测不出来"通过

三次 smoke（各 12 组/半区、5 clips，第三次换 seed）：

| | real | diar | 相对 real | slice | 相对 real |
|---|---|---|---|---|---|
| smoke12 | 0.0493 | 0.0303 | 0.615 | 0.0274 | 0.556 |
| smoke13 | 0.0602 | 0.0598 | 0.993 | 0.0274 | 0.455 |
| smoke14 (seed 99) | 0.0440 | 0.0352 | 0.800 | **0.0443** | **1.007** |

**这张表最重要的一行是最后一行：换个 seed，slice 的比值从 0.46 跳到 1.01。**
也就是说"slice 组内方差稳定地只有一半"这个结论 —— 我自己在上一轮报出去的 ——
**在第三次抽样中不复现，必须撤回**。三级修正，全是被数据逼出来的：

1. **裸 `>` 不行。** real 自己的中位数在 smoke 间从 0.0493 动到 0.0602，
   比较两个中位数会把 0.0004 的差当成"方向"。→ **±15% 相对容差**内输出 `comparable`。
2. **点比值也不行。** 同样 12 个 diar 组，比值 **0.615 → 0.993 → 0.800**。
   单看 0.993 会读成"切点干净地迁移到 diar 了"—— 而这正是纯度红线的承重结论。
   → `bootstrap_ratio`（**以组为重采样单元，两个中位数都重采样**，
   区间同时覆盖参照物和被测物的噪声），跨过 ±15% 带就输出 `unresolved`。
3. **"两次 smoke 一致"也不算证据**（seed 相同则相关，不是独立复现）。

smoke14 的比值区间：

| 半区 | 点比值 | 比值 95% CI | 判定 |
|---|---|---|---|
| diar | 0.80 | **[0.29, 2.00]** | `unresolved` |
| slice | 1.01 | **[0.21, 2.12]** | `unresolved` |

**结论：12 组下这个迁移方向根本判不了，两个半区都是 `unresolved`。**
区间宽到同时容纳"窄 5 倍"和"宽 2 倍"，所以任何基于它的推论 ——
无论是"diar fail-safe 所以放心用"，还是我报的"slice 欠触发所以更该弃用"——
**都没有数据支撑**。正式版 400 组会把区间收紧到可判，届时才有资格下结论。

> 这不是说迁移偏差不存在。6.1.2 的相邻性结构（slice 85.5% 连续切片）**仍然**强烈暗示
> slice 组内变化更小，那条证据独立于本节且样本量大得多（4177 组）。
> 只是**这个特定的 `max_drop` 比值统计量在 12 组下测不出它**。用结构证据下结论，
> 不要用一个测不出方向的统计量去给它背书。

##### 这个比值统计量在 400 组下也测不出来（功效模拟）

在把"等 400 组结果"当成预案之前，先算了它的功效。用实测形状（对数正态，
中位数 0.05，p90/中位数 = 4.5）模拟，`--real-groups 60` 固定：

| 真实比值 | laion 组数 | 比值 CI 中位宽度 | 判定成功率 |
|---|---|---|---|
| 1.0（真 comparable） | 60 | 1.15 | **0/60** |
| 1.0 | 400 | **0.84** | **0/60** |
| 0.7 | 400 | 0.55 | 11/60 |
| 0.5 | 400 | 0.38 | 44/60 |

**关键：真实比值 = 1.0 时，"comparable" 这一支在 400 组下也几乎永不成立**（模拟 0/60）。
再加 laion 组也没用 —— 瓶颈是 **real 那 60 组**，而它不随 laion 组数变化：
把 real 加到 1200 组才勉强 2/50。换成秩统计量（AUC）也一样。

结论：**"400 组收敛到 comparable → diar 红线成立"这一支不是"等结果"，是等一个不会来的结果。**
可判的只有单边强信号（真实比值 ≤0.5 时约 70% 能检出"更窄"）。

##### 因此改成在域内直接测灵敏度（`spike_sensitivity`）

问题问错了。"切点在 diar 上还灵不灵"不必通过"两域方差是否可比"来**推断**，可以直接**测**：
拿该半区**自己的**片段，把一个组里的一条换成**另一个源**的另一组的片段 ——
这样构造出的组**按定义是混人的** —— 然后数 `flag_switches` 在标定切点上抓到多少。

这个 recall 就是红线真正需要的东西，而且：**域内测量，不含任何迁移假设**；
不需要额外音频（embedding 已经算好）；样本量由 trials 决定，不受 real 组数瓶颈限制。

**红线判定因此改成合取**（`mixing_at_calibrated_cut`）：

- `rate_upper_bound_ok`：混人率 CI 上界 < 5%
- `detector_sensitivity_ok`：域内 recall 的 CI **下界 ≥ 50%**
- `red_line_pass` = 两者**同时**成立

理由：**"没测出混人" + "抓不到植入的混人" = 什么都没证明。** 上界低而 recall 低时探针明确输出
"NOT a pass: the low rate may be blindness. Fix sensitivity, not the threshold."

> 为什么直觉会反：laion 的组是**同一段源档案的切片**，共享信道、房间与说话状态；
> 而 vctk/ears 的一个说话人跨的是多次录音 session。"YouTube 噪声大"说的是
> **录音之间**的方差，而切点关心的是**组内**方差。

##### 因此再加一个尺度无关的切点

`z_drop = max_drop / 组内 stdev`：问的是"这条离**它自己所在组**有多远"，
没有任何关于该域绝对方差的假设，因此不受上面那个偏差影响。用同一批已知纯组按同样的
FPR 目标标定（`z_calibration`），两个切点的混人率**并排输出**：

- 两者一致 → 上面的偏差在此规模下无实际影响
- z 的率**更高** → 绝对切点确实在 laion 上欠触发，此时以 z 为准（探针会打印 `!!` 提示）

z **不能取代**绝对切点：一个"整体都很松"的组（每条都彼此远）z 值不高，只有绝对切点抓得到。
两者是互补的，**两个率不一致本身就是结论**。

**实测：5 clips 下 z 标定不出来**（已知纯组的 FPR：z≥1.0 70.8%、z≥2.0 33.3%、z≥2.5 8.3%，
全部超 5% 目标）。原因很直白 —— 组内 stdev 是用 C(5,2)=10 个对估的，
**分母本身比分子还吵**，于是这个本来要去偏的统计量比它要修的绝对切点更不稳。
探针如实输出"没有可用 z 切点"并提示加 `--clips`，**不降目标凑一个切点**。
正式版跑 `--clips 5` 时这项预计仍然标定不出来，届时以绝对切点 + slice 的欠触发警告为准。

## 7. 训练集切分注意

**train / eval 必须按说话人切分，不能按行切分。** 按行随机切会让 eval 的
speaker reference 落到 train 里见过的说话人身上，零样本能力的评估结果会虚高。
`filtered_index.jsonl.gz` 里带了 `speaker_id` 和 `language`，据此分组即可。
