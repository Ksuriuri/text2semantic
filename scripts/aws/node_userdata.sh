#!/bin/bash
# p5.48xlarge training node bootstrap.
#
# Runs on every node, including the ones the Spot supervisor recreates hours into
# a run, so it has to be idempotent and it must not contain anything specific to
# one run. Everything that changes between the preflight and the real launch lives
# in s3://BUCKET/_staging/node-config.sh, which this sources: flipping that one
# small object is how the run changes, not a new launch template version. A
# replacement node then picks up the current config by itself.
#
# The end state is /opt/t2s/READY plus a wrapper the supervisor can systemd-run.
set -x
LOG=/var/log/t2s-node.log
exec > >(tee -a "$LOG") 2>&1

BUCKET=noiz-t2s-us-east-2
REGION=us-east-2
NAME="$(curl -s -H "X-aws-ec2-metadata-token: $(curl -sX PUT http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
  http://169.254.169.254/latest/meta-data/tags/instance/Name || echo unknown)"

# A node whose bootstrap failed at minute 3 is indistinguishable from one still
# copying 8 TB unless the log leaves the box. $40/h each is too much to diagnose
# one ssh at a time.
( while sleep 60; do
    aws s3 cp "$LOG" "s3://$BUCKET/_staging/logs/node-$NAME.log" \
      --region "$REGION" >/dev/null 2>&1
  done ) &

# ------------------------------------------------------- no unattended restarts
# A fresh Ubuntu node runs apt-daily-upgrade within the first hour, because the
# timer is Persistent and has no stamp file yet, and then needrestart restarts
# every service whose libraries were replaced. On 2026-08-21 06:03:50 that ran
# `systemctl restart t2s-train.service` on noiz-t2s-p5-3 after a libcurl update,
# and restarting ONE rank kills the whole 64-rank torchrun job: the run resumed
# from step 16,310 having reached 16,570, so three security packages cost 260
# steps plus 11 minutes of stop/relaunch. A replacement node is the dangerous
# case -- it boots into that first-hour window while 63 other ranks are training.
# Belt and braces: the timers cannot fire, and if anyone runs apt by hand
# needrestart still leaves the training unit alone. Never fatal to the bootstrap.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer || true
mkdir -p /etc/needrestart/conf.d
cat > /etc/needrestart/conf.d/99-t2s-train.conf <<'NRCONF'
$nrconf{override_rc}{q(^t2s-train)} = 0;
NRCONF

mkdir -p /opt/t2s
aws s3 cp "s3://$BUCKET/_staging/node-config.sh" /opt/t2s/node-config.sh \
  --region "$REGION" --only-show-errors
# shellcheck disable=SC1091
. /opt/t2s/node-config.sh
: "${T2S_DATA_PREFIX:?node-config.sh must set T2S_DATA_PREFIX}"
: "${T2S_REPO_TARBALL:?node-config.sh must set T2S_REPO_TARBALL}"
: "${T2S_TRAIN_ARGS:?node-config.sh must set T2S_TRAIN_ARGS}"

# ---------------------------------------------------------------- local storage
# The dataset is 8.4 TB and one instance-store disk is 3.5 TB, so /data has to be
# all eight of them together.
#
# On this AMI they already are: the Deep Learning AMI stripes the whole instance
# store into an LVM volume at /opt/dlami/nvme (27.6 TB on p5.48xlarge) before
# user-data runs. So the striping is done, and the job here is only to put /data
# where the space is. A bind mount rather than a symlink, so /data stays a real
# directory for anything that resolves paths.
#
# The mdadm branch is the fallback for an image that leaves the disks raw. It
# picks them by "nvme, no partitions, not mounted" rather than by name, because
# the EBS root is also an nvme device and its index is not guaranteed.
mkdir -p /data
if mountpoint -q /data; then
  echo "/data already mounted"
elif mountpoint -q /opt/dlami/nvme; then
  echo "using the AMI's striped instance store at /opt/dlami/nvme"
  mount --bind /opt/dlami/nvme /data
else
  DISKS=()
  for dev in /dev/nvme*n1; do
    [[ -b "$dev" ]] || continue
    lsblk -no MOUNTPOINT "$dev" | grep -q . && continue
    lsblk -no NAME "$dev" | tail -n +2 | grep -q . && continue
    DISKS+=("$dev")
  done
  echo "instance store: ${#DISKS[@]} raw disks: ${DISKS[*]}"
  if [[ "${#DISKS[@]}" -eq 0 ]]; then
    echo "FATAL: no striped instance store and no raw NVMe disks" >&2
    exit 1
  elif [[ "${#DISKS[@]}" -eq 1 ]]; then
    mkfs.xfs -f "${DISKS[0]}"
    mount -o noatime "${DISKS[0]}" /data
  else
    mdadm --create /dev/md0 --level=0 --raid-devices="${#DISKS[@]}" \
      --chunk=512 --force --run "${DISKS[@]}"
    mkfs.xfs -f /dev/md0
    mount -o noatime /dev/md0 /data
  fi
fi
df -h /data

# ------------------------------------------------------------------------- code
cd /data
aws s3 cp "s3://$BUCKET/$T2S_REPO_TARBALL" /data/t2s-repo.tar.gz \
  --region "$REGION" --only-show-errors
rm -rf /data/t2s-repo && mkdir -p /data/t2s-repo
tar xzf /data/t2s-repo.tar.gz -C /data/t2s-repo --strip-components=1 \
  || tar xzf /data/t2s-repo.tar.gz -C /data/t2s-repo

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
cd /data/t2s-repo
# Out of $HOME: the venv is 6 GB of CUDA wheels and the root EBS volume is 200 GB
# shared with the driver, the DLAMI conda trees and the logs.
export UV_CACHE_DIR=/data/uv-cache
uv venv --python 3.12 /data/venv
VIRTUAL_ENV=/data/venv uv pip install -r <(uv export --no-hashes --no-dev 2>/dev/null) \
  || VIRTUAL_ENV=/data/venv uv pip install -e .
/data/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# Assert the fused linear-attention kernel, do not hope for it. 18 of Qwen3.5's
# 24 layers are linear attention, and transformers falls back to a float32 chunk
# loop with nothing but a warning when flash-linear-attention is missing: 7.47
# s/step instead of 5.03, which on this run is $6k and a day and a half. A node
# that came up without it must fail here rather than quietly join the job.
#
# This is the earlier of two gates, not the only one. The deployed launch
# template version predates it and the supervisor box cannot revise the template
# (its role has no ec2:CreateLaunchTemplateVersion), so the gate that is actually
# in force on the current run is --require_fused_linear_attention in
# node-config.sh's train args. Keep both: this one fails a bad node before it
# spends 40 minutes copying 8 TB.
/data/venv/bin/python - <<'FLACHECK'
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
import liger_kernel.transformers.monkey_patch as liger
assert qwen.chunk_gated_delta_rule is not None, "flash-linear-attention missing"
assert hasattr(liger, "apply_liger_kernel_to_qwen3_5"), "liger-kernel too old"
print("fused kernels: flash-linear-attention + liger ok")
FLACHECK

# ---------------------------------------------------------------------- wrapper
# Written before the 8 TB copy on purpose. The wrapper's first job is to block on
# READY, which only helps if it exists while the node is still building: a
# supervisor tick that lands during the copy then keeps one unit active instead of
# failing with status 127 and relaunching every 60 s.
#
# The supervisor re-supplies T2S_MACHINE_RANK / T2S_MAIN_IP / T2S_NUM_MACHINES /
# T2S_FABRIC on every relaunch, so none of them belong here. What does belong is
# everything that must be identical across relaunches -- above all the pinned
# W&B run id, without which each preemption forks a new curve.
cat > /opt/t2s/train_multinode_wrapper.sh <<'WRAPPER'
#!/bin/bash
set -euo pipefail
# A supervisor tick can arrive while the node is still copying 8 TB. Blocking
# here keeps the unit active so the supervisor watches instead of relaunching
# into a half-built node every 60 s.
for _ in $(seq 1 720); do
  [[ -e /opt/t2s/READY ]] && break
  sleep 15
done
[[ -e /opt/t2s/READY ]] || { echo "node never became READY" >&2; exit 1; }

. /opt/t2s/node-config.sh
cd /data/t2s-repo
export TEXT2SEMANTIC_PYTHON=/data/venv/bin/python
export HF_HOME=/data/hf
export WANDB_DIR=/data/wandb
# Fetched here rather than baked into node-config.sh on purpose: the bootstrap
# runs under `set -x` and ships its log to S3, so a key placed in the config would
# be echoed into that log. This script has no `set -x`, and the value stays in the
# process environment only.
WANDB_API_KEY="$(aws ssm get-parameter --region us-east-2 \
  --name /noiz-t2s/wandb-api-key --with-decryption \
  --query 'Parameter.Value' --output text)"
export WANDB_API_KEY
# NCCL_SOCKET_IFNAME only defaults to eth0 when eth0 exists; on this AMI the
# interfaces are ens*, and with 32 of them NCCL should not be left guessing which
# one carries the rendezvous. Take the one holding the default route.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(ip -o route get 1.1.1.1 | awk '{print $5}')}"
exec bash scripts/train_multinode.sh $T2S_TRAIN_ARGS
WRAPPER
chmod +x /opt/t2s/train_multinode_wrapper.sh

# ------------------------------------------------------------------------- data
# S3 and ENA ride network card 0 only (the other 31 interfaces are efa-only and
# carry no IP traffic), so this leg is capped at 100 Gbps no matter how many EFA
# devices the instance has. Concurrency is what gets close to that cap.
aws configure set default.s3.max_concurrent_requests 128
aws configure set default.s3.max_queue_size 20000
aws configure set default.s3.multipart_chunksize 64MB
mkdir -p /data/trainsets /data/models

if [[ -n "${T2S_FSX_DNS:-}" ]]; then
  # FSx path. The data is not copied at all: it lives in the shared filesystem, so
  # a node is READY in about a minute instead of ~40 for 9.26 TB. That is the whole
  # point -- under Spot the fleet is partially replaced constantly, and the copy was
  # being paid per replacement.
  #
  # Bind-mounting onto /data/trainsets and /data/models rather than pointing the
  # training args at /fsx is deliberate and load-bearing: the prebuilt manifest
  # index records the ABSOLUTE ref_index path it was built with, so running with
  # /fsx/... paths makes _stale() report "filter parameters changed" and rescan
  # 51.4 GiB on every node. Identical paths, different backing store, index reused.
  mkdir -p /fsx
  mount -t lustre -o relatime,flock \
    "${T2S_FSX_DNS}@tcp:/${T2S_FSX_MOUNTNAME}" /fsx
  mount --bind /fsx/trainsets /data/trainsets
  mount --bind /fsx/models   /data/models

  # Cheap assertions instead of `du -sh`: du on FSx walks 157k files of metadata
  # for no benefit, and what actually matters is that the bind mounts point at
  # lustre (not at an empty local dir, which would silently train on nothing) and
  # that the two manifests are visible at their expected sizes.
  for p in /data/trainsets /data/models; do
    mount | grep -q " $p type lustre" \
      || { echo "FATAL: $p is not on FSx" >&2; exit 1; }
  done
  ls -la /data/trainsets/t2s-v1/manifests/ /data/models/
  df -h /fsx

  # The dataset must be hydrated before training, not during it. Files arrive from
  # an import-linked DRA as HSM stubs ("released"): reading one byte restores the
  # WHOLE file -- measured 524 MB pulled and 1.78 s blocked for a 1-byte dd on a
  # 539 MB shard. With 64 dataloader workers sampling randomly across 16,599 tars
  # that turns the first pass into 9.26 TB of hydration, during which the GPUs sit
  # at 0% util / 123 W and no step completes. fsx_preload.sh pays it once for the
  # fleet at ~3.5 GB/s; this is only the gate that says whether it was paid, because
  # the failure mode otherwise looks like a comms stall rather than cold storage.
  released=$(find /data/trainsets/t2s-v1/refs/shards -type f \
    | xargs -n 200 lfs hsm_state 2>/dev/null | grep -c released || true)
  echo "FSX-HYDRATION: ${released:-unknown} of 16599 ref shards still released"
  if [[ "${released:-0}" -gt 100 ]]; then
    echo "WARNING: refs are cold; expect starved GPUs until preload finishes" >&2
  fi

  # Second first-touch cost, and the one that hid for longer. Reading a ref is a
  # pread at an offset that comes from a per-shard sidecar in refs/.member-index;
  # that directory is not in the bucket, so a missing sidecar is built by scanning
  # the entire 539 MB tar, lazily, by whichever worker first wants a speaker in it.
  # Starved GPUs look identical to the cold-storage case above, which is why both
  # get their own line here. fsx_preload.sh builds these once for the fleet.
  # `|| true` is load-bearing: ls exits non-zero on a missing directory and this
  # bootstrap runs under `set -euo pipefail`, so without it an absent sidecar dir
  # would abort the boot of every node instead of printing the warning below.
  sidecars=$(ls /data/trainsets/t2s-v1/refs/.member-index 2>/dev/null | wc -l || true)
  echo "FSX-MEMBER-INDEX: ${sidecars} sidecars of 16599 shards"
  if [[ "${sidecars:-0}" -lt 16599 ]]; then
    echo "WARNING: $((16599 - sidecars)) shards lack a member index; the first" \
         "sample from each will scan a whole tar and starve the GPUs" >&2
  fi
else
  time aws s3 sync "s3://$BUCKET/models/" /data/models/ \
    --region "$REGION" --only-show-errors
  time aws s3 sync "s3://$BUCKET/$T2S_DATA_PREFIX" /data/trainsets/t2s-v1/ \
    --region "$REGION" --only-show-errors
  du -sh /data/trainsets /data/models
fi

# ---------------------------------------------------------------- shared indexes
# Prebuilt elsewhere: the row index over train.jsonl is two single-threaded passes
# over 51.4 GiB and the speaker key table is one over 5.0 GB, and both produce the
# same bytes on every node. The speaker table has to land next to its jsonl --
# speaker_index.index_dir_for() is "<path>.index" and is not configurable.
if [[ -n "${T2S_MANIFEST_INDEX:-}" ]]; then
  mkdir -p /data/manifest-index
  aws s3 sync "s3://$BUCKET/$T2S_MANIFEST_INDEX" /data/manifest-index/ \
    --region "$REGION" --only-show-errors
fi
# Both of the next two write INTO the dataset tree, which on the FSx path is shared
# by the whole fleet rather than private to this node. Eight nodes writing the same
# bytes to the same file at once is not something to find out about later, so on FSx
# they run only when the work is actually missing. Safe to skip: no auto-export
# policy on the DRAs, so nothing here can propagate back into the bucket either.
if [[ -n "${T2S_SPEAKER_TABLE:-}" ]]; then
  if [[ -n "${T2S_FSX_DNS:-}" ]] \
     && [[ -s /data/trainsets/t2s-v1/refs/speaker_index.jsonl.index/keys.blob ]]; then
    echo "speaker table already present on FSx, skipping sync"
  else
    aws s3 sync "s3://$BUCKET/$T2S_SPEAKER_TABLE" \
      /data/trainsets/t2s-v1/refs/speaker_index.jsonl.index/ \
      --region "$REGION" --only-show-errors
  fi
fi
# The pin both builds recorded. S3 carries no mtime, so on the local-copy path this
# is the only thing that makes a shipped index look unchanged here -- each node's
# download stamps its own clock. On FSx the file has ONE mtime for the whole fleet,
# so the divergence the pin exists to fix cannot occur; pinning once is enough, and
# it is idempotent, so this stays unconditional rather than becoming a special case.
if [[ -n "${T2S_PIN_MTIME:-}" ]]; then
  touch -d "@$T2S_PIN_MTIME" /data/trainsets/t2s-v1/manifests/train.jsonl \
    /data/trainsets/t2s-v1/manifests/eval.jsonl \
    /data/trainsets/t2s-v1/refs/speaker_index.jsonl
fi
# Reported, not enforced: a stale index still trains correctly, it just rebuilds.
# Worth a line in the log that names the reason, because the symptom otherwise is
# a wandb curve that starts an hour late for no visible cause.
aws s3 cp "s3://$BUCKET/_staging/verify_index.py" /data/t2s-repo/verify_index.py \
  --region "$REGION" --only-show-errors
( cd /data/t2s-repo && PYTHONPATH=/data/t2s-repo /data/venv/bin/python verify_index.py \
    -- $T2S_TRAIN_ARGS ) || true


# ------------------------------------------------------------------- efa checks
# The three gates train_multinode.sh enforces, run once here so a broken image or
# a launch template missing its EFA interfaces shows up in the bootstrap log
# rather than 40 minutes later on eight nodes at once.
/opt/amazon/efa/bin/fi_info -p efa | grep -c "provider: efa" || echo "NO EFA DEVICES"
# /opt/amazon/ofi-nccl/lib on this AMI, which is also the first place
# train_multinode.sh looks; the find is for an image that puts it elsewhere.
ls /opt/amazon/ofi-nccl/lib/libnccl-net*.so 2>/dev/null \
  || find /opt -name 'libnccl-net*.so' 2>/dev/null || echo "NO OFI PLUGIN"

touch /opt/t2s/READY
echo "READY $(date -Is)"
aws s3 cp "$LOG" "s3://$BUCKET/_staging/logs/node-$NAME.log" --region "$REGION" --only-show-errors
