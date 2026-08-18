# 双机 Spot 训练（asia-east1，2 × a3-highgpu-8g）

16 张 H100 跑 2B 模型，两台机器都是 Spot。这份文档只讲和单机不同的三件事：
网络、被抢占之后怎么继续、以及启动顺序。

## 1. 网络：TCPX 不是可选项

`a3-highgpu-8g` 每台 8 张 H100，挂在 5 个 vNIC 后面：`eth0` 走普通流量，
`eth1`-`eth4` 专门给 GPU 流量。NCCL 只有通过 GPUDirect-TCPX 插件才会用到后四个，
**没有插件时它不会报错**，而是安静地退回 `eth0` 一张 gVNIC —— 每 step 约 8 GB 的
all-reduce 全挤在上面，本来算力受限的任务会变成网络受限。

所以 `scripts/train_multinode.sh` 默认要求插件存在：找不到
`/var/lib/tcpx/lib64` 或 `/usr/local/tcpx/lib64` 就直接退出，除非显式设
`T2S_TCPX=0`。同样，flow-steering 用的 receive-datapath-manager
（`tcpgpudmarxd`，socket 在 `/run/tcpx`）必须先起来，否则 NCCL 会在第一次跨机
collective 上卡住，而不是报缺依赖。

`NCCL_GPUDIRECTTCPX_TX_BINDINGS` / `RX_BINDINGS` 用的是 a3-highgpu-8g 的标准
core 映射（每个数据网卡绑到自己 GPU 所在的 NUMA 节点，tx 和 rx 分开）。换机型或
换镜像时要用 `lscpu` 重新推导。

## 2. 被抢占之后：checkpoint 必须在对象存储上

抢占带走整台机器，包括本地盘和只存在盘上的 checkpoint；两台机器里任何一台丢了，
任务都活不下去。所以训练必须带 `--checkpoint_remote_dir gs://...`：

- 每台机器只上传自己写的文件（`save_state` 的 model/optimizer 只有主进程写，
  `random_states_<rank>.pkl` 每个 rank 都写，本地盘各自独立时谁也没有完整一份）；
- 两台都传完之后才由 rank 0 写 `_UPLOAD_COMPLETE` 标记，恢复时只认带标记的
  目录 —— 被抢占打断的上传在 listing 里和完整的长得一模一样；
- 配了远端前缀后 `--resume_from_checkpoint auto` **只看远端**，忽略本地目录：
  存活的那台机器上可能留着一个上传没完成的更新的 checkpoint，两台从不同权重恢复
  DDP 是不报错的。

GCP 的抢占通知只有约 30 秒，一个 2B 模型的 checkpoint 十几 GB，**存不完**。
SIGTERM 触发的那次保存是尽力而为，真正决定损失上限的是
`--checkpointing_min_interval_minutes`（默认 30 分钟）。

## 3. 启动顺序

两台机器跑同一个脚本，只有 `T2S_MACHINE_RANK` 不同：

```bash
# rank 0（其他机器 dial 进来的那台）
T2S_MACHINE_RANK=0 T2S_MAIN_IP=<rank0 内网 IP> \
  scripts/train_multinode.sh \
  --train_jsonl /mnt/data/train.jsonl \
  --eval_jsonl /mnt/data/eval.jsonl \
  --output_model_path /mnt/data/output \
  --checkpoint_remote_dir gs://noiz-taiwan-audio-data/runs/t2s-v1/checkpoints \
  --resume_from_checkpoint auto \
  ...

# rank 1，同样的参数，只改 rank
T2S_MACHINE_RANK=1 T2S_MAIN_IP=<rank0 内网 IP> scripts/train_multinode.sh ...
```

索引（manifest / speaker index）的预构建现在是**每台机器**做一次
（`is_local_main_process`），不再只在全局 rank 0 上做 —— 本地盘各自独立时
全局 rank 0 那种写法会让 node 1 的 8 个 rank 同时去建同一个索引，就地写
`offsets.u64` 加上最后才写 `meta.json`，结果是一个悄悄坏掉的索引。
现在有目录锁 + 暂存后原子改名（`finetuning/index_build.py`）。

## 4. 抢占看护

`scripts/spot_supervisor.py` 跑在训练机器**之外**（机器都没了，机器上的守护进程
也就没了）。循环很小：

    所有节点 RUNNING 且没有任务   -> 每台都拉起来
    所有节点 RUNNING 且任务在跑   -> 继续看着
    任何节点不是 RUNNING          -> 先停任务，重建那台，再重新拉起

它只操作命令行里给出的实例名，不会去列项目里的机器再自作主张 —— 没给它的名字
很可能是别人的机器。注意第三条里“先停任务”是必要的：半个任务比没有任务更糟，
活着的那台会卡在 collective 上等已经不存在的 rank。

```bash
python3 scripts/spot_supervisor.py \
  --node t2s-train-0 --node t2s-train-1 \
  --zone asia-east1-c \
  --instance-template t2s-a3-highgpu-8g-spot \
  --launch-command '/opt/t2s/train_multinode_wrapper.sh' \
  --unit t2s-train
```

先用 `--once --dry-run` 看一遍它打算做什么。

## 还没验证的部分

TCPX 的环境变量、5 网卡的 NCCL 路径和双机 all-reduce 带宽，都要等真机起来才能
测。第一件事应该是 NCCL 的 all-reduce benchmark（不是直接开训）：如果跨机带宽
掉到 gVNIC 的水平，s/step 会比单机 8 卡差，那么先修网络再谈训练吞吐。
