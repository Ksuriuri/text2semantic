# Sourced by both the node bootstrap and the training wrapper. This is the only
# run-specific file: it lives at s3://noiz-t2s-us-east-2/_staging/node-config.sh,
# so switching from the preflight to the real launch is one small upload, and a
# node the Spot supervisor recreates mid-run reads the current version by itself.
#
# "Recreates" is doing real work in that sentence, and it caught me once. Only a
# *new* node re-reads S3, because the download lives in node_userdata.sh at boot.
# train_multinode_wrapper.sh sources the local /opt/t2s/node-config.sh and runs the
# local /data/t2s-repo, so a node that merely had its unit restarted keeps whatever
# it booted with. Changing this file therefore means uploading it AND refreshing
# /opt/t2s/node-config.sh on every surviving node, or the fleet silently splits: at
# the batch 48 relaunch the two survivors would have run batch 44 on 02a8cc3 while
# six new nodes ran batch 48 on ddad06e.
#
# PRODUCTION, one epoch. Now 8 nodes x 8 H100 x batch 48 x accum 1 = 3,072
# rows/step, which is where it started -- but for a different reason, on repo
# ddad06e, whose fused loss deletes the allocation that killed batch 48 twice.
# In between, a Spot reclaim took it down to 2 nodes x batch 32 x accum 6 and back
# up through 6, and the batch ladder went 48 -> 24 -> 40 -> 44 -> 48. Read DEGRADED
# SHAPE, BATCH 32, 6 NODES, 8 NODES, BATCH 24, BATCH 40, BATCH 44 and BATCH 48 AGAIN
# below in that order: that is the history, and the batch paragraphs are why batch
# was the only lever that moved memory until ddad06e removed the reason.
# Everything here that differs from node-config-preflight.sh was measured on that
# preflight first: see notes/text2semantic.md for the 2-node numbers.
#
# Batch 48 rather than the 32 originally approved, on kusuriuri's "if it measures
# clean and 48 fits, go straight to 48". Measured on one node, 30 steps, 20 s
# references, fused kernels on: 5.473 s/step and 70.2 rows/s against batch 32's
# 4.766 s/step and 53.7. Batch 64 was tried too and is worse, 9.254 s/step for
# 512 rows, so 48 is the top of the useful range rather than a step towards more.
#
# Run length: 102,254,841 kept rows / 3072 = 33,287 steps. That comes from
# --num_epochs 1 and nothing else -- train.py accepts exactly one of --num_epochs
# and --max_train_steps, which is what keeps the run length and the LR schedule on
# one source instead of letting a copied max_train_steps decay the LR to zero
# early.
#
# DEGRADED SHAPE, 2026-08-20 10:0xZ. AWS reclaimed 7 of the 8 Spot nodes at
# 09:34Z and us-east-2a has no p5.48xlarge Spot capacity left (placement score
# 1/10 in all three us-east-2 AZs at every target size, so 2b/2c is not a fix).
# Running kusuriuri's standing rule on the 2 nodes we hold instead -- halve the
# nodes, double the accumulation -- so the total stays 3072 rows/step. The run
# length stays 33,287 steps and the W&B curve stays comparable; only wall-clock
# moves, from 3.79 s/step to about 15.2 at batch 48, or ~18.7 at the batch 32 the
# card turned out to require.
#
# This file is global, so a node the supervisor recreates reads whatever is here.
# The supervisor's node count and the accumulation below therefore have to be
# changed together, and the invariant to hold is the product: batch x 8 GPUs x
# nodes x accum = 3072 rows/step.
#
# BATCH 32, 2026-08-20 10:5xZ. Batch 48 would not survive the first step at this
# shape: every rank filled the card (expandable_segments could not map a 20 MiB
# extension with 18.9 MB free of 85.0 GB) and rank 4 died in the backbone's
# full-attention forward with a cuDNN mha_graph failure, reproducibly at 10:18,
# 10:23, 10:27, 10:42 and 10:46. NCCL_NVLS_ENABLE=0 did not help, and the first
# reading -- an NVLS multicast buffer failing -- was the same exhaustion seen one
# allocation earlier. Batch 48 was never comfortable: the 8-node run peaked at
# 72.69 GiB reserved of 79.2, so it had ~6 GiB of margin, and 16 ranks draw a
# different (differently grouped) set of rows than the 64 ranks that measured it.
# Batch 32 was measured at 4.766 s/step against 48's 5.473, so it costs about 23%
# per row -- the shape kusuriuri pre-approved as the fallback.
#
# Batch 32 also widens the usable node counts, which is worth more than the 23%
# right now: nodes x accum must be 12 rather than 8, so 1, 2, 3, 4, 6 and 12 nodes
# all hold 3072. Three nodes is legal at batch 32 and was not at 48, and Spot
# hands capacity back one node at a time.
#   2 nodes -> accum 6          3 nodes -> accum 4
#   4 nodes -> accum 3          6 nodes -> accum 2 (here)
#
# 6 NODES, 2026-08-20 11:4xZ. The probe won all four back inside 31 minutes --
# p5-2 10:42:54, p5-3 11:03:25, p5-4 11:08:18, p5-5 11:13:31 -- and each staged
# 8.3 TB from S3 in about 40 minutes, so the whole reclaim cost roughly two hours
# of training rather than the run. Cut in one move at accum 2 instead of stopping
# at 3 and then 4 nodes: each cutover re-does everything since the last checkpoint,
# and 2 restarts cost more than the 20 minutes the intermediate shapes would win.
#
# It also takes 8 nodes away: 32 x 8 x 8 = 2048 and 3072/2048 is not an integer.
# Eight nodes therefore means going back to batch 48 x accum 1, which is not a
# gamble -- that is the shape that actually ran steps 0..9231 -- or to batch 24 x
# accum 2 if 48 turns out to be as marginal at 64 ranks as it was at 16.
#
# 8 NODES, batch 48 x accum 1, 2026-08-20 12:5xZ. Back to the original shape: the
# probe won p5-6 (11:32) and p5-7 (12:05) and both staged 8.3 TB by ~12:50. This
# is a restore, not a new bet -- steps 0..9,231 ran here at 3.79 s/step and 72.69
# GiB reserved of 79.2 -- and the 6-node measurement is what makes it worth the
# thin margin: 10.0 s/step at accum 2 against 3.79 at accum 1, i.e. the
# per-micro-batch fixed cost, not comms. If 64 ranks OOM anyway, the fallback
# kusuriuri already has in writing is 8 nodes x batch 24 x accum 2.
#
# BATCH 24 x ACCUM 2, 2026-08-20 13:4xZ -- batch 48 is dead at 64 ranks too. It
# resumed at 12:54, ran 195 steps, and at 13:21 rank 49 on p5-6 filled the card
# (free 4,259,840 of 85,019,590,656) with rank 55 raising the same cuDNN
# mha_graph.execute failure in backward at 13:22 that killed the 16-rank attempts
# this morning. So this morning's five failures were not a small-world-size
# artefact: the shape has no margin at any world size, it only wins the sequence-
# length lottery for a while. peak 59.27 GiB with reserved back at 72.67 of 79.2.
# The supervisor did its job -- 13:23 stop, 13:31 relaunch from
# checkpoint-step-9235, ~200 steps lost.
#
# 24 x 8 x 8 x 2 = 3072 unchanged. At 8 nodes the legal batches are the divisors of
# 384, so under 48 the next one down is 24; batch 32 and 40 do not divide it.
#
# NCCL_NVLS_ENABLE=0 is dropped with this relaunch. It was added at 10:18Z as a
# memory precaution, it never helped a single OOM, and it was still set for the
# 7.4 s/step 8-node leg (against 3.79 for the same shape before the reclaim) with
# per-node utilisation 64-94% instead of pinned. It is a suspect for that gap, not
# a proven cause -- my "reserved fell 6 GiB because the NVLS buffers went away"
# reading was wrong, reserved is a high-water mark and it climbed back to 72.67 --
# and batch 24 leaves enough headroom to just test it.
#
# BATCH 40 x ACCUM 1, 2026-08-20 13:5xZ, and the total batch changes with it.
# kusuriuri's direct instruction (`f18a0692` + `737bac94`): "每卡40" and "梯度累积1".
# At 8 nodes accum 1 means the total is batch x 64, so this is 2,560 rows/step, not
# the 3,072 every earlier shape held. Stated to them; it is their call and it is a
# real change, not a rounding:
#   - the epoch becomes 102,254,841 / 2560 = 39,943 steps instead of 33,287, and
#     train.py derives both the length and the cosine from --num_epochs, so the
#     schedule follows by itself;
#   - resume is by step, not by row, so restarting at global_step 9,235 skips
#     9,235 x 2,560 = 23.6M rows where 28.4M were actually consumed. ~4.7M rows
#     (4.7% of the epoch) get seen a second time and the epoch end shifts.
# Throughput is why it is still the right trade: accum 1 pays the per-micro-batch
# fixed cost once, so the estimate is ~3.5 s/step for 2,560 rows (~730 rows/s)
# against batch 24 x accum 2 at ~6.1 s for 3,072 (~500 rows/s).
#
# Memory risk: 40 is 83% of the batch that just OOM'd twice, so this is not a safe
# shape, only a less marginal one. If it fails the same way, the ladder down at 8
# nodes and accum 1 is batch 32 (2,048 rows/step); batch 24 x accum 2 is the shape
# that holds 3,072 if the total batch matters more than the rate.
#
# BATCH 44 x ACCUM 1, 2026-08-20 15:1xZ. kusuriuri's `ab605178`: "我感觉是不是还是
# 将batch提上去就好了，试试44呢". 44 x 64 = 2,816 rows/step, so the epoch is 36,312
# steps. Third total batch of the day, and the third time the W&B x-axis changes
# meaning.
#
# Why it is worth a restart: batch 32 -> 48 measured +50% rows for +15% time, so
# per-row cost falls with batch. 40 -> 44 is +10% rows for an estimated +3-5% time,
# i.e. 5-7% more rows/s. A cutover costs ~20 minutes of stop_job plus startup and
# ~100 redone steps, so it pays back in about 6 hours against a run of several days.
#
# Why it is not safe, only safer: batch 40 measured peak 53.17 / reserved 65.32 and
# batch 48 measured peak 59.27 / reserved 72.67 before dying. Linear in batch, 44 is
# ~56.2 / ~69.0 against 79.2, so ~10 GiB of margin where 48 had 6.5. But 48 ran 195
# steps *before* it died, and it died wanting one 11.67 GiB contiguous block, i.e.
# the trigger is drawing a long-sequence batch, not the average. Told kusuriuri: if
# 44 dies the same way, batch 40 is final and I stop climbing.
#
# BATCH 48 x ACCUM 1 AGAIN, 2026-08-20 17:3xZ, and this time the memory ceiling
# moved instead of the batch. kusuriuri's `c4a1cabb`: "能节省显存的点可以做吗 ... 没有
# 的话就做，然后batch恢复到48". 48 x 64 = 3,072 rows/step, so the epoch is back to
# 33,287 steps and back to the x-axis every shape before batch 40 used.
#
# What changed is repo ddad06e (see T2S_REPO_TARBALL): the speech head's logits are
# fused into the cross entropy, so the B x L x V fp32 logits and their gradient are
# never materialised. That is the specific allocation named in the paragraph below
# and the one both batch-48 deaths asked for. It is not a general "use less memory"
# tweak whose effect has to be guessed at -- it deletes the tensor.
#
# Measured at 2 nodes / 16 ranks, on kusuriuri's instruction to prove the shape on
# two nodes before paying for eight (`a2820820`): 2.65 s/step median over 21
# intervals, peak 58.17 GiB and reserved 65.45 flat for 115 consecutive steps.
# Flat is the result, not the median: the old batch 48 climbed reserved to 72.67 and
# then died on a long-sequence draw, and what a fused loss should do is exactly this,
# stop the tail from existing. b44 was measured on the same harness at 2.69 s/step,
# so 48 is also ~10% more rows/s (289 vs 262 per rank-second).
#
# The honest limit of that evidence: 16 ranks regroup rows into different batches
# than 64 do, so they draw a different max sequence length, and this file has one
# shape in it already that survived 9,231 steps at one world size and died instantly
# at another. Two nodes can show the fused path is correct, fast and non-growing; it
# cannot prove the 64-rank tail. What makes 48 the right bet anyway is the mechanism
# rather than the measurement -- the block that killed it is gone -- and the fallback
# if it dies regardless is unchanged: batch 44, then 40.
#
# Row accounting at this resume, same effect as the batch-40 cutover but the other
# way round: resume is by step, so restarting checkpoint-step-10625 at 3,072
# rows/step skips 10,625 x 3072 = 32.64M rows, while the last ~1,200 steps actually
# ran at 2,560 and 2,816, so fewer than that were really consumed. The difference is
# a slice near the resume point that gets skipped unseen rather than trained twice.
#
# Reconstructed leg by leg from the W&B peak-VRAM trace, because a guess at "a thin
# slice" was wrong by a factor of five. Each leg re-derives its own offset as
# global_step x that leg's rows/step, so the position is not monotonic across a
# batch change:
#     A  b48/3072  step    10 -> 9,530   reached 29.28M
#     B  b40/2560  step 9,540 -> 10,130  offset moved BACKWARDS to 24.41M,
#                                        re-training ~1.51M rows
#     C  b44/2816  step 10,140 -> 10,690 reached 30.10M
# Resuming at 10,625 x 3072 = 32.64M against a high-water mark of 30.10M leaves
# **2.54M rows, ~2.5% of the epoch, never trained** -- not the 0.5% first published
# in 9544b176, which this comment previously repeated. Corrected in the thread.
#
# Shuffling is global and seeded (train.py does train_generator.manual_seed(args.seed)
# for the whole DataLoader), so the permutation is reproducible and a skipped slice is
# a random sample, not a topic or a language. That seeding is also the answer to
# "can the shuffle seed be fixed": it already is, and no code change is needed to
# start. Making the resume row-exact *would* be a code change -- carry the consumed
# row count in the checkpoint instead of re-deriving it from the step number -- and is
# deferred to after this run rather than restarting again for 2.5%.
#
# What this is NOT: a fix for the utilisation question that prompted it. Measured at
# batch 40 -- all 8 nodes 59-68% mean sm (no straggler), rank 0 GPU 0 over 60 s
# averaging 309 W of 700 with 138-614 W swings, disk 12-17 MB/s on every node
# including the 22 h one, CPU 8.7% of 192, iowait 0.1%. sm% reads 100% at 127 W
# because a waiting NCCL kernel counts as utilised, so the honest gauge is power.
# The cards are idle inside the step for a reason not yet located, and a 10% larger
# batch cannot fix that -- it only buys the per-row amortisation above.
#
# Cutover cadence is now fixed by kusuriuri (`684efcc5`): switch only at 2, 4 and
# 8 nodes. 3 and 6 are legal at batch 32 but each restart re-does everything since
# the last checkpoint, and that outweighs the intermediate shape's gain.
#
# What is actually large here is the speech head's logits, B x L x V in fp32 plus
# its gradient: the 8-node OOM wanted one 11.67 GiB contiguous block, so at ~8k
# tokens and this codebook the batch dimension is multiplying something near 11
# GiB per copy. That is why batch is the lever that moved 85 GB down to 55, and
# why --speaker_encode_chunk (11.15 -> 3.24 GiB on the frozen W2V-BERT) did not
# move the step's peak at all. Per-rank memory does not depend on the node count,
# so this ceiling travels with every cutover below.
#
# ddad06e is the fix for that paragraph rather than another lever against it: the
# fused loss never builds the logits, so the 11 GiB-per-copy term is gone and batch
# stops being the only thing that moves memory. Measured directly at the OOM shape
# (48 x 8000 tokens, V=8195, H=2048): 42.59 GiB unfused against 7.59 GiB fused for
# the loss and its backward alone.
#
# Repo 02a8cc3, which ddad06e sits directly on top of, is required by this shape
# rather than incidental to it: a
# checkpoint records step_in_epoch (the saving rank's own micro-batch count) and
# the prepared scheduler's last_epoch (process-scaled), both only meaningful at
# the world size that wrote them. Resuming 64-rank checkpoint-step-8825 on 16
# ranks with the old code would have skipped a quarter of the rows already
# trained -- ~20M rows twice, the epoch's last ~20M never -- and restored an LR
# position past the end of the cosine, i.e. under 1% of peak, silently. 02a8cc3
# recomputes both from global_step and logs the rescale.

# Exported, not passed: the wrapper sources this file before exec'ing the
# launcher, so anything exported here reaches all 8 processes on the node.
#
# expandable_segments is what makes batch 48 fit on *eight* nodes. The one-node
# measurement had 59.3 GiB allocated against a 66.3 GiB reserved high-water and
# looked comfortable; at 64 ranks NCCL and the EFA plugin take about 8 GiB of the
# card outside PyTorch, and the first 8-node attempt died in backward asking for
# one 11.67 GiB block with 10.6 GiB free and 13.0 GiB sitting in reserved-but-
# unallocated segments. That gap is fragmentation, not use: the backward pass
# wants a single large contiguous block (the speech head's logit gradient) and
# the allocator's fixed segments cannot supply it. Expandable segments grow
# instead of being fixed at allocation size, so a large contiguous request can be
# served without a matching free segment.
#
# Measured with it on, at step 210 of the 8-node run: train/vram_peak_gib 59.27,
# train/vram_reserved_gib 72.69. So reserved does *not* collapse onto allocated --
# both are high-water marks and they peak at different moments -- and the earlier
# version of this comment claiming they would stay close was wrong. What changed
# is that the 11.67 GiB block now gets served instead of raising. Watch the two
# numbers for a *trend*: reserved climbing while peak stays flat is the
# fragmentation this setting is meant to absorb running out of room.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T2S_DATA_PREFIX="trainsets/t2s-v1/"
# ddad06e fuses the speech head into the cross entropy (Liger FLCE), which removes
# exactly the allocation that killed batch 48 on eight nodes: the B x L x V logits
# and their gradient. Measured on an H100 at the OOM shape (48 x 8000 tokens,
# V=8195): 42.59 GiB for the loss alone unfused against 7.59 GiB fused, and the
# loss matches the old F.cross_entropy to 1e-7 and its gradients to one bf16 ulp.
# Average peak barely moves (59.27 -> 58.17 GiB) because the average batch was never
# the problem; what changes is the long-sequence tail that used to ask for one 11.67
# GiB contiguous block. It is one commit on top of 02a8cc3 and nothing else.
#
# It carries a second, unrelated fix worth knowing about when reading step times:
# _validate_speech_ids called int() on a CUDA tensor twice per forward, and each of
# those is a hard synchronisation that drains the previous step's queue, so it was
# also destroying the overlap between steps. The bound comes from the manifest and
# cannot vary per batch, so it now runs once.
#
# Both are on by default (T2S_FUSED_CE defaults to 1, T2S_VALIDATE_SPEECH_IDS to
# "once"), so this file needs no new variable -- the T2S_FUSED_CE=1 in the 2-node
# benchmark was for A/B legibility, not because production has to set it. The
# escapes, if a run ever has to be bisected: T2S_FUSED_CE=0 and
# T2S_VALIDATE_SPEECH_IDS=always restore the old behaviour exactly.
#
# ddad06e plus one line, hence the suffix: cuDNN's fused attention kernel fails in
# the BACKWARD pass on some batch shapes ("mha_graph.execute ... got false"), which
# killed global rank 55 twenty steps into this resume and took all 64 ranks with it.
# train.py now calls torch.backends.cuda.enable_cudnn_sdp(False) at import, leaving
# SDPA to pick flash/mem-efficient -- exact attention either way. torch 2.13 reads no
# environment switch for this, so it had to be code. **This is not committed
# upstream yet**: the tarball was built from the patched tree so that a Spot
# replacement comes up with the fix instead of rejoining and crashing the job again.
# It needs a real commit to Ksuriuri/text2semantic via the bundle relay.
T2S_REPO_TARBALL="_staging/t2s-repo-ddad06e-cudnnsdp.tar.gz"

# Two indexes, both single-threaded full scans, both built once on a cheap box by
# build_index_userdata.sh rather than on eight $20/h nodes: the row index over the
# 51.4 GiB train.jsonl (two passes) and the key table over the 5.0 GB speaker
# index (one pass, which SpeakerRefStore would otherwise build on every node).
# They live outside trainsets/ so the staged dataset stays byte-identical to what
# was verified against the copy manifest.
T2S_MANIFEST_INDEX="_staging/manifest-index/"
T2S_SPEAKER_TABLE="_staging/speaker-index-table/"

# ------------------------------------------------------------------------- FSx
# Setting these two switches the data path from "copy 9.26 TB to local NVMe on
# every new node" to "mount the shared filesystem". Unset them and the bootstrap
# falls back to the S3 copy, unchanged -- worth keeping, because the two paths have
# genuinely different failure modes and the copy is the one with no shared ceiling.
#
# Why the change: local NVMe is faster per node (5.2 GB/s against FSx's ~3 GB/s for
# the WHOLE fleet -- one node alone measured 2,614 MB/s warm, i.e. it can consume
# nearly the entire budget). That sounds decisive until you measure what training
# actually needs, which is ~19 MB/step/node, about 56 MB/s aggregate at 8 nodes --
# roughly 50x under the shared ceiling. So the throughput the copy buys is
# throughput nothing asks for, and its real cost is ~40 minutes of $20/h node per
# replacement, paid over and over under Spot.
#
# What it does NOT remove is the byte movement; it relocates it. Files arrive as HSM
# stubs from the import-linked DRAs and hydrate on first read -- WHOLE file per
# touch, so 16,599 random-access tars means the first pass drags all 9.26 TB in
# while the GPUs starve. fsx_preload.sh pays that once for the fleet at a measured
# 3,471 MB/s (~45 min), and every replacement afterwards is free. Read
# fsx_preload.sh before changing anything here.
#
# _staging/ deliberately stays on S3. All three DRAs are AutoImportPolicy None, so
# FSx shows the snapshot from when the association was created: at cutover time its
# view was missing exactly t2s-repo-ddad06e.tar.gz, the current code. Small files
# that change every run do not belong behind a manual import task.
T2S_FSX_DNS="fs-00132cb2828dac10e.fsx.us-east-2.amazonaws.com"
T2S_FSX_MOUNTNAME="syvahb4v"
# Both are only reusable if their source looks unchanged: _stale compares the
# jsonl's size *and* mtime_ns against what the build recorded, and S3 carries no
# mtime, so a download stamps whatever the clock said. Every box -- the builder
# and every node, including one the supervisor recreates mid-run -- therefore pins
# both jsonls to this one constant. verify_index.py is the gate that says so out
# loud instead of letting train.py quietly rescan 51 GiB.
T2S_PIN_MTIME=1755561600

# --attn_implementation sdpa is not a default: train.py asks for flash_attention_2
# and transformers raises ImportError when the wheel is absent, which it is here --
# flash-attn ships no prebuilt wheel for torch 2.13/cu130 on python 3.12 and a
# source build is 30+ minutes on every node. It also matters far less than its
# name suggests: only 6 of Qwen3.5's 24 layers are full attention, the other 18
# are linear attention, and those are handled by flash-linear-attention, which
# the repo now depends on and the gate below asserts. sdpa is what every
# step-time measurement so far was taken with, so the numbers stay comparable.
#
# --liger swaps the Qwen backbone's RMSNorm and SwiGLU for Liger's fused Triton
# versions: 5.034 -> 4.766 s/step, and both modules were checked bit-exact
# against the transformers originals on this config.
#
# --require_fused_linear_attention is the gate that makes the 5.03 s/step real on
# every node, including one the supervisor recreates at hour 30: without
# flash-linear-attention, 18 of 24 layers fall back to a float32 chunk loop and
# transformers only warns. It lives here rather than in node_userdata.sh because
# the launch template cannot be revised from the supervisor box -- its role has
# no ec2:CreateLaunchTemplateVersion -- and this file plus the repo tarball are
# the two objects a run can actually change.
#
# NOT set: --speaker_encode_chunk. It does shrink the frozen W2V-BERT forward
# (11.15 -> 3.24 GiB at batch 48 / 20 s refs) but left the step's peak at 75.8
# GiB, so it buys nothing here.
#
# Also not set, because it is now the default: the speaker encoder is
# checkpointed along with the backbone. That is what makes batch 48 comfortable
# rather than marginal -- 59.3 GiB allocated and 66.3 GiB reserved out of 79.2,
# against a nvidia-smi reading of 75.8 GiB before -- at a cost of about 2% per
# step. --no-speaker_gradient_checkpointing turns just that half off.
#
# --max_ref_seconds stays at its 20 s default. Capping it at 10 was worth 12-15
# GiB before the fused kernels; after them batch 48 measured 5.354 s/step capped
# against 5.382 uncapped, so it is no longer paying for the semantic change.
#
# --speaker_encoder_dtype bfloat16 and --speaker_mel_in_workers are kept from the
# preflight, but honestly: on these H100s they were worth 2% (7.86 -> 7.68 s/step),
# not the 1.45x they bought on B200. The B200 was fast enough for host-side work to
# dominate; this step is memory-bandwidth bound on the GPU instead. Harmless, and
# the losses were shown numerically neutral, so they stay.
#
# --resume_from_checkpoint auto is what makes a Spot preemption cost minutes
# instead of the whole run. Without it a relaunched node starts at step 0 with the
# same --wandb_run_id, which does not just lose the work -- it rewinds the curve
# kusuriuri is watching. Verified both ways on the preflight: from a local
# checkpoint, and by wiping /data/output so rank 0 had to pull step 60 out of S3.
#
# --checkpointing_steps 5 with --checkpointing_min_interval_minutes 30 is
# kusuriuri's cadence: the step multiple is only a gate on when the clock is
# allowed to fire, so this means "about every 30 minutes", ~240 saves over the
# epoch, and at most ~30 minutes of work lost to a preemption.
T2S_TRAIN_ARGS="\
--train_jsonl /data/trainsets/t2s-v1/manifests/train.jsonl \
--eval_jsonl /data/trainsets/t2s-v1/manifests/eval.jsonl \
--base_model_path /data/models/Qwen3.5-2B-Text \
--w2v_bert_path /data/models/w2v-bert-2.0 \
--stats_path /data/models/indextts25_codec/wav2vec2bert_stats.pt \
--ref_backend packed \
--attn_implementation sdpa \
--liger \
--require_fused_linear_attention \
--speaker_encoder_dtype bfloat16 \
--speaker_mel_in_workers \
--resume_from_checkpoint auto \
--ref_index /data/trainsets/t2s-v1/refs/speaker_index.jsonl \
--ref_root /data/trainsets/t2s-v1/refs \
--manifest_index_dir /data/manifest-index \
--output_model_path /data/output \
--checkpoint_remote_dir s3://noiz-t2s-us-east-2/runs/t2s-v1/checkpoints \
--batch_size 48 \
--gradient_accumulation_steps 1 \
--lr 1e-4 \
--new_module_lr 4e-4 \
--num_epochs 1 \
--checkpointing_steps 5 \
--checkpointing_min_interval_minutes 30 \
--checkpoint_total_limit 2 \
--num_workers 8 \
--logging_steps 10 \
--eval_steps 2000 \
--wandb_run_id t2s-v1-20260819 \
--wandb_run_name t2s-v1-8node-h100 \
"
