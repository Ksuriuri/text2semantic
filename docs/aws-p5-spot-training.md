# AWS p5.48xlarge 8 机 Spot 训练

这份文档是 t2s-v1（2026-08-19 到 08-21）在 `us-east-2a` 上真正跑通的做法。
通用的多机 Spot / EFA / supervisor 机制见 [multinode-spot-training.md](multinode-spot-training.md)。
这里只写 AWS 上多出来的部分：FSx、launch template、一次拉满 8 台、以及结束时怎么放。

脚本在 `scripts/aws/`。节点上真正开训的还是仓库里的 `scripts/train_multinode.sh`
和 `scripts/spot_supervisor.py`。

## 已经量过的形状

| 项 | 值 |
|---|---|
| 机型 | `p5.48xlarge`，每台 8 × H100 80GB，32 个 EFA，3200 Gbps |
| 台数 | **永远 8 台一起**（MinCount=MaxCount=`8:8`） |
| 形状 | batch 48 × 8 GPU × 8 机 × accum 1 = **3072 rows/step** |
| 一轮 | 102,254,841 行 / 3072 = **33,287 steps**（只设 `--num_epochs 1`，不要再设 `--max_train_steps`） |
| 稳态 | 约 **3.3 s/step**（FSx 热、cuDNN SDP 关掉之后） |
| 权重 | `s3://noiz-t2s-us-east-2/runs/t2s-v1/checkpoints/` |

不要用「先开 1 台再补」的方式凑 8 台：supervisor 默认每次 `RunInstances --count=1`，
会留下付了钱却凑不齐 world size 的半舰队。全舰队重建时先停 supervisor，再一次 `8:8`。

## 常驻资源（训练节点可以放，这些先不要拆）

| 资源 | 标识 | 备注 |
|---|---|---|
| FSx for Lustre | `fs-00132cb2828dac10e`（mount `syvahb4v`） | 12 TiB PERSISTENT_2，250 MB/s/TiB，`us-east-2a`。约 $0.21/GB-月 ≈ **$3.39/h** |
| S3 | `s3://noiz-t2s-us-east-2` | 数据集、checkpoint、`_staging/` |
| S3 gateway | `vpce-04dcfb32ed72287cf` | 数据集和 ckpt **不走 NAT** |
| bastion / supervisor | `noiz-t2s-supervisor` / `t2s-bastion`，`t3.small` | 公网约 `16.59.44.18`，实例 `i-0fcea7f1f068fbc39` |
| NAT | `noiz-t2s-p5-nat` | 只给 PyPI / W&B。数据集走 gateway |
| launch template | `t2s-p5-spot` **v8**（默认） | userdata 里关了 apt 定时器和 needrestart |
| 放置 | `us-east-2a` / `use2-az1`，placement group `noiz-t2s-p5` | FSx 单 AZ，舰队没有别的 AZ 可选 |
| Spot 价 | 约 $19.98（2a） | 2b/2c 更贵，而且没有这份 FSx |

W&B token 放在 SSM `/noiz-t2s/wandb-api-key`，**不要**写进 `node-config.sh`（bootstrap 带 `set -x`，日志会进 S3）。

## 为什么必须开 FSx

每台 p5 的本地 NVMe 更快（单机热读可到 2.6 GB/s），但训练稳态只要大约 **20 MB/s/机**，
8 机合计约 160 MB/s，远低于 FSx 的共享带宽。真正贵的是 Spot 换机：

- 不挂 FSx：每台新节点从 S3 拉约 9.3 TB，READY 大约 **40 分钟**
- 挂了 FSx：数据在共享盘上，READY 大约 **4 分钟**

2026-08-21 整舰队被抢光后重建：8 台全部 READY 在 09:52:53–09:53:25，
`FSX-HYDRATION: 0 of 16599 still released`，`FSX-MEMBER-INDEX: 16599 of 16599`。
hydration 和 member-index 是**舰队一次性**成本，换光所有节点也不用重付。

绑定方式（load-bearing）：

```
mount -t lustre -o relatime,flock ${T2S_FSX_DNS}@tcp:/${T2S_FSX_MOUNTNAME} /fsx
mount --bind /fsx/trainsets /data/trainsets
mount --bind /fsx/models   /data/models
```

训练参数仍写 `/data/trainsets/...`。预构建的 manifest index 记的是**绝对路径**，
如果改成 `/fsx/...`，`_stale()` 会认为 filter 变了，每台重扫 51 GiB。

`_staging/`（repo tarball、`node-config.sh`、index）留在 S3。DRA 的 AutoImport 是 None，
FSx 上看不到后来上传的小文件。

### 第一次必须付的两笔

1. **HSM hydration**。import-linked DRA 上的文件是 stub（`released`）。读 1 字节会拉**整文件**
   （539 MB shard 上 1-byte `dd` 实测拉了 524 MB、堵 1.78 s）。64 个 dataloader 随机打
   16,599 个 tar，等于第一遍把 9.26 TB 当场 hydrate，GPU 0% / 123 W。
   用 `scripts/aws/fsx_preload.sh` 在空舰队上先做完，约 3.5 GB/s、45 分钟。
2. **`refs/.member-index` sidecar**。不在 bucket 里。缺了就会在第一次采样时扫完整 539 MB tar。
   同样由 `fsx_preload.sh` 建一次。

bootstrap 会打两行，重建后先看这两行再开训：

```
FSX-HYDRATION: N of 16599 still released
FSX-MEMBER-INDEX: N sidecars of 16599 shards
```

`N` 不是 0 / 16599 时不要开 8 机，否则看起来像 NCCL 卡住。

## 网络

`scripts/aws/setup_network.sh` 建这些（可重入）：

- 私有 subnet `172.31.64.0/20`（`us-east-2a`），自己的 route table，不动别人的 default VPC
- NAT 在**公网** subnet（NAT 自己要能出网）
- S3 **gateway** endpoint 挂在那张 route table 上。8 机 × 9 TB 走 NAT 按 $0.045/GB 是三千刀；走 gateway 是 $0
- EFA 安全组：允许自己到自己的全部流量
- placement group `noiz-t2s-p5`

p5 要 32 张网卡才能吃满 3200 Gbps：card 0 一张 `efa`（唯一带 IP 的），其余 31 张 `efa-only`。
EC2 对多网卡实例不给公网 IPv4，所以训练节点**只有内网**。supervisor 必须跑在同一 VPC 里。

节点上：

```
export T2S_FABRIC=efa
```

`train_multinode.sh` 会硬检查 `libnccl-net*.so`、`fi_info -p efa`、`ulimit -l unlimited`。
不要手写 TCPX 那套 `NCCL_ALGO` / channel 数。`NCCL_SOCKET_IFNAME` 取默认路由那张网卡
（这台 AMI 上是 `ens*`，写死 `eth0` rendezvous 会直接失败）。

## Launch template

`scripts/aws/make_launch_template.py` + `scripts/aws/node_userdata.sh`。

当前默认 **v8**。相对 v7 只多了 userdata 里的两段保护（2026-08-21 06:03 事故之后）：

- `systemctl disable --now apt-daily.timer apt-daily-upgrade.timer`
- `/etc/needrestart/conf.d/99-t2s-train.conf`：`$nrconf{override_rc}{q(^t2s-train)} = 0;`

`apt-daily-upgrade.timer` 是 Persistent 的。新节点启动后大约 1 小时内会跑
`unattended-upgrades`，然后 needrestart 会 `systemctl restart t2s-train`。
重启**一台** rank 就会干掉整个 64-rank job。那次从 step 16,570 回到 16,310，
大约 260 step + 11 分钟。Spot 换机正好落在这个窗口。

其它固定项：Ubuntu 24.04 OSS PyTorch DLAMI、root EBS 200 GB gp3、
instance profile `NoizT2sStagingProfile`、密钥 `noiz-t2s-20260819`、
Spot one-time。run-specific 的东西**不要**写进 template，写进
`s3://noiz-t2s-us-east-2/_staging/node-config.sh`。

只有**新节点**会重新拉 S3 上的 config。活着的节点只读本地 `/opt/t2s/node-config.sh`。
改 config 必须上传 **并且** 刷新每台还活着的节点，否则舰队会 silently 分裂。

## 怎么一次拉起 8 台

容量分数是预测，不是判决。`use2-az1` 对 target=8 打过 1/10，
`run-instances --count 8:8` 仍然瞬间满员。FSx 在 2a，2b 分数再高也用不了。
**不要只凭 SPS 说没容量。**

```bash
# 1) 先停 bastion 上的 supervisor，避免它按 --count=1 往里滴节点
# 2) 一次要 8 台，要么全有要么全没有
aws ec2 run-instances \
  --region us-east-2 \
  --launch-template LaunchTemplateName=t2s-p5-spot,Version='$Default' \
  --count 8:8 \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Project,Value=t2s-v1}]'

# 3) 按 Name 标成 noiz-t2s-p5-0 .. p5-7（supervisor 靠这个名字认节点）
# 4) 等 8 台 running，bastion 上再开 supervisor
```

bastion 上（`scripts/aws/run_supervisor.sh`）：

```bash
# tmux 会话名习惯用 sup8，日志 /home/ec2-user/sup8.log
bash /home/ec2-user/t2s-repo/scripts/aws/run_supervisor.sh 8
```

它会带 `--setenv T2S_FABRIC=efa`。`WANDB_RUN_ID` 必须钉死（`node-config.sh` 里的
`--wandb_run_id`），抢占恢复才接回同一条曲线。

中途 Name 撞上还在 `shutting-down` 的旧实例时，supervisor 会报 `AMBIGUOUS` 并停手，
这是对的。等旧实例变成 `terminated` 就会自己继续。不要为此再建第三台。

## 进节点

没有 SSM（实例角色没有 `ssm:UpdateInstanceInformation`）。

```bash
ssh -i secrets/noiz-t2s-20260819.pem ec2-user@16.59.44.18
ssh -i /home/ec2-user/.ssh/noiz-t2s.pem ubuntu@<private-ip>
```

换机之后内网 IP 会变，以 `describe-instances` 为准。不要 pin host key。

## 高效训练真正靠的几件事

写在 `scripts/aws/node-config.sh` 和仓库代码里，不是调 NCCL knobs：

1. **`T2S_FABRIC=efa`**，缺插件就退出，不要默默退回 TCP。
2. **`torch.backends.cuda.enable_cudnn_sdp(False)`**（`74ad3e0`）。cuDNN 的 fused attention
   会在某些 batch 的 backward 里 `mha_graph.execute ... got false`，干掉整个 job。
   留给 flash / mem-efficient SDPA。这一行必须在**启动前**就在磁盘上；热补丁改文件
   碰不到已经起来的进程。
3. **fused CE**（`ddad06e`）。speech head 的 B×L×V fp32 logits 不再物化，
   batch 48 才站得住。`T2S_FUSED_CE` 默认开。
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**。64 rank 时 NCCL/EFA 大约占 8 GiB，
   backward 会要一块约 11 GiB 的连续显存。
5. **`--require_fused_linear_attention --liger --attn_implementation sdpa`**。
   没有 flash-linear-attention 时 18/24 层会静默退回 float32 chunk loop（7.47 vs 5.03 s/step）。
6. **`--resume_from_checkpoint auto` + `--checkpoint_remote_dir s3://...`**。
   只认带 `_UPLOAD_COMPLETE` 的远端目录。一个完整 ckpt 是 **18.53 GB / 75 个对象**。
   非递归 `aws s3 ls` 大约 8 条，不要把那个数当成没传完。
7. **keep 每 5000 step**（`checkpoint-keep-step-*`）。**永远不要删。**
   rolling save 大约 30 分钟一次（`--checkpointing_steps 5 --checkpointing_min_interval_minutes 30`），
   `checkpoint_total_limit 2`。
8. **不要让 8 台空转。** 凑不齐 world size 就停 supervisor、放掉节点，等能一次要齐 8 台再开。

`node-config.sh` 里还记了 batch 从 48 掉到 32/24/40/44 再回到 48 的原因。
换 world size 时 resume 按 **step** 不算行，行数会对不齐；能不切就别切。

## Checkpoint

```
s3://noiz-t2s-us-east-2/runs/t2s-v1/checkpoints/
  checkpoint-step-32765/          # 最新完整权重（v1 停在 33,287，没有 33287 那份 save）
  checkpoint-keep-step-5000/
  checkpoint-keep-step-10000/
  ...
  checkpoint-keep-step-30000/
```

推理只要大约 **4.4G**：`model.safetensors` + tokenizer/config。不要把
`accelerator_state` / optimizer 拷到听音盒。

## 结束一轮

1. 确认步数到了，而且最新 `checkpoint-step-*`（或 keep）有 75 个对象 + `_UPLOAD_COMPLETE`。
2. **先停 supervisor**（tmux `sup8`）。它还在就会立刻再买 Spot。
3. `terminate-instances` 这 8 台。等到全部 `terminated`。
4. FSx / bastion / NAT / IAM **先留着**，除非有人明确说拆。
5. 通知部署的人 S3 前缀和该用哪份权重。

## 容量、分数、半舰队

- supervisor 的 `--count=1` 只适合「已经在跑、补一台」。
- 全舰队重建：停 supervisor，`8:8`，再开 supervisor。
- `scripts/aws/capacity_probe.sh` 是中途逐台打听容量的历史工具，**不要**拿它做首次 8 机启动。
- 旧实例还在 `shutting-down` 时会出现 `AMBIGUOUS`；等它 terminated，不要手建第三台。

## 成本（同一账户里还有别人的 p5）

账户 `728750562872` 是共享的。8 月起就有别的团队的 p5 在账单上。
**不要用未过滤的 Cost Explorer。** 按 Name tag `noiz-t2s-*` 从 CloudTrail `RunInstances`
加 CloudWatch 最后一次 CPU（+5 min）还原小时数。Cost Explorer 会滞后一天以上。

v1 整轮大约 **$7,760 ±2%**（2026-08-22 01:50Z）：p5 Spot 381 node-h × $19.98 ≈ $7,618，
FSx $115（拆掉节点后仍按约 $81/天在走），S3 / EBS / NAT+bastion 其余。
如果全程 8 台满、3.30 s/step，大约 $4,880；Spot 换机和升温大约多了 $2,740。

## 不要做的事

- 在训练节点上跑 apt，或让管理 agent 重启服务
- 热补丁改代码却不重建进程（下一次 Spot 换机才会读到磁盘上的文件）
- 只看 W&B 的 step / state 就判断 stall。resume 后 W&B 会丢掉小于计数器的点，
  `state=crashed` 可能是上一轮进程留下的。对照 S3 最新 ckpt 和 rank 0 的 `systemctl is-active t2s-train`
- 报速率用太短的窗口。checkpoint / eval 会把几秒打进那个 bucket。引用速率用至少 500 step
- `pkill -f` 一个也会出现在自己 ssh 命令里的字符串
- 拆 FSx / bastion / NAT，除非被要求
