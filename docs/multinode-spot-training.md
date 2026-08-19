# 多机 Spot 训练（GCP a3-highgpu-8g / AWS p5.48xlarge）

多台 8 卡 H100 跑 2B 模型，机器全是 Spot。这份文档只讲和单机不同的三件事：
网络、被抢占之后怎么继续、以及启动顺序。第 1-4 节以 GCP 双机为例，第 5 节是
AWS 上的对应做法（两边共用同一个 `train_multinode.sh` 和同一个 supervisor）。

## 1. 网络：TCPX 不是可选项

`a3-highgpu-8g` 每台 8 张 H100，挂在 5 个 vNIC 后面：`eth0` 走普通流量，
`eth1`-`eth4` 专门给 GPU 流量。NCCL 只有通过 GPUDirect-TCPX 插件才会用到后四个，
**没有插件时它不会报错**，而是安静地退回 `eth0` 一张 gVNIC —— 每 step 约 8 GB 的
all-reduce 全挤在上面，本来算力受限的任务会变成网络受限。

所以 `scripts/train_multinode.sh` 默认要求插件存在（`T2S_FABRIC=tcpx`）：找不到
`/var/lib/tcpx/lib64` 或 `/usr/local/tcpx/lib64` 就直接退出，除非显式设
`T2S_FABRIC=none`（旧的 `T2S_TCPX=0` 仍然等价）。同样，flow-steering 用的 receive-datapath-manager
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

## 5. 换到 AWS：EFA 和 EC2

同一套脚本也能跑 AWS 的 `p5.48xlarge`（8 × H100，32 个 EFA 设备，3200 Gbps），
但两处必须显式切过去，因为两边的默认值互不通用。

**网络：`T2S_FABRIC=efa`。** 对应 TCPX 的是 libfabric 加 aws-ofi-nccl 插件，
失败方式一模一样：插件不在，NCCL 不报错，直接退回 TCP socket。所以启动前有三道
硬检查，任何一条不过就退出，而不是让一台 $400/h 的机器慢着跑：

- `libnccl-net*.so` 是否存在（新版叫 `libnccl-net-ofi.so`，NCCL 只按
  `libnccl-net.so` 这个名字自动找，所以脚本会在需要时自己设 `NCCL_NET_PLUGIN`）；
- `fi_info -p efa` 是否真的列出设备 —— launch template 没挂 EFA 接口、内核模块
  没加载、安全组少了“允许自己到自己全部流量”那条规则，都会在这里暴露；
- `ulimit -l` 是否 unlimited。EFA 直接注册 pinned memory，限额会在跑起来之后才以
  `ibv_reg_mr failed` 的形式炸出来。systemd 要 `LimitMEMLOCK=infinity`，
  docker 要 `--ulimit memlock=-1`。

另外 `NCCL_SOCKET_IFNAME` 只在存在 `eth0` 时才默认成 `eth0`：AWS 上网卡叫
`ens*`/`enp*`，写死 eth0 会让 rendezvous 直接失败。EFA 分支里刻意**没有**抄
TCPX 那一堆 `NCCL_ALGO` / `NCCL_PROTO` / channel 数 —— p5 上插件自己的默认值才是
AWS 调过的，锁成 Simple 反而掉带宽。要调就先测。

**抢占看护：`--cloud aws`。** 三处和 GCE 不同，都在 `Ec2Cloud` 里：

- EC2 的实例“名字”只是一个 tag，所以每次查状态都是带 filter 的 describe，
  且服务端就把 terminated 过滤掉（跑一周的 Spot 会攒下一堆同名的死实例）；
- 没有 `gcloud compute ssh` 这种帮你管密钥和防火墙的东西，所以是普通 ssh 连当时
  的地址，要 `--ssh-user` / `--ssh-key`；重建出来的节点是新机器、可能拿到回收的
  内网 IP，host key 变化是正常的，因此不 pin host key；
- **Spot 中断是 terminate，不是 stop**，所以 MISSING（用 launch template 重建）
  才是常态，`start` 只对 persistent spot request 有意义。

一个名字下面查到两台活着的实例时状态是 `AMBIGUOUS`：这说明上一次 create 只成功了
一半，正确处理是停手报出来，而不是再建第三台。

```bash
python3 scripts/spot_supervisor.py \
  --cloud aws --region us-east-2 \
  --node t2s-train-0 --node t2s-train-1 ... --node t2s-train-7 \
  --instance-template t2s-p5-spot \
  --ssh-user ubuntu --ssh-key ~/.ssh/t2s.pem \
  --setenv T2S_FABRIC=efa \
  --setenv WANDB_RUN_ID=t2s-v1-8node \
  --launch-command '/opt/t2s/train_multinode_wrapper.sh' \
  --unit t2s-train
```

`--setenv` 是重要的一环：这些值每次重新拉起都要重新给一遍，写在第一次启动的 shell
里没用。`WANDB_RUN_ID` 固定住之后，被抢占再恢复才会接回同一条曲线，而不是每次抢占
新开一条。checkpoint 前缀在 AWS 上直接写 `s3://`（`finetuning/checkpoint_remote.py`
按 scheme 选 `aws s3` 还是 `gcloud storage`），跨云镜像每半小时十几 GB 的出网费和
延迟都在保存的关键路径上，不值得。

## 还没验证的部分

TCPX 的环境变量、5 网卡的 NCCL 路径和双机 all-reduce 带宽，都要等真机起来才能
测。第一件事应该是 NCCL 的 all-reduce benchmark（不是直接开训）：如果跨机带宽
掉到 gVNIC 的水平，s/step 会比单机 8 卡差，那么先修网络再谈训练吞吐。

AWS 一侧同理，而且多一件事要量：`p5.48xlarge` 上 per-GPU batch 32 能不能装进 80 GB
（B200 上实测 64.8 GiB），以及真实的 s/step。这两个数字现在都是推算，不是测量。
跨机 all-reduce 之后应该 grep 一次 NCCL 的 `NET/OFI`（`NCCL_DEBUG=INFO`
`NCCL_DEBUG_SUBSYS=INIT,NET`）确认插件真的被加载了 —— 上面三道检查能证明插件文件
和 EFA 设备都在，但不能证明 NCCL 选了它。
