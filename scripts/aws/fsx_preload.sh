#!/bin/bash
# One-time hydration of the FSx working set from S3.
#
# Why this exists: the three DRAs are import-linked with AutoImportPolicy None, so
# every file arrives as an HSM *stub* -- metadata present, data still in S3, state
# "released". Any read restores the WHOLE file, not the requested range: a 1-byte
# dd on a 539 MB shard pulled 524 MB and blocked for 1.78 s. With 64 dataloader
# workers doing random access into 16,599 half-gigabyte tars, that means the first
# pass is pure hydration and the GPUs sit at 0% util / 123 W waiting on it. A
# one-node test read 1.24 TB without completing a single step.
#
# Doing it here instead costs the same bytes at full filesystem rate (measured
# 3,471 MB/s with 24-way parallel restore) and, unlike per-node S3 staging, it is
# paid ONCE for the whole fleet -- the data lives in the shared filesystem, so a
# replacement node afterwards is READY in about a minute instead of ~40 minutes.
#
# Idempotent: only files still in "released" state are queued, so re-running after
# an interruption resumes rather than re-fetching.
set -uo pipefail

ROOT=/fsx/trainsets/t2s-v1
LOG=/var/log/fsx_preload.log
exec >>"$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) fsx preload start ==="

# lfs hsm_restore takes files as argv and dies with "Argument list too long" well
# before xargs' own limit, so the chunk is deliberately small (20) and the
# parallelism (24) is what supplies the throughput.
preload_tree() {
  local dir="$1" label="$2"
  [ -d "$dir" ] || { echo "SKIP $label: $dir absent"; return 0; }
  local list="/tmp/preload_${label}.txt"
  find "$dir" -type f > "$list"
  local total; total=$(wc -l < "$list")
  echo "--- $label: $total files, filtering to released..."

  # hsm_state is also argv-bound, so batch the query the same way. Only released
  # files are worth queueing; already-resident ones would be a no-op that still
  # costs a round trip per file.
  local rel="/tmp/preload_${label}_released.txt"
  : > "$rel"
  xargs -a "$list" -n 200 lfs hsm_state 2>/dev/null \
    | grep released | cut -d: -f1 >> "$rel"
  local n; n=$(wc -l < "$rel")
  echo "--- $label: $n released of $total, restoring at 24-way..."

  local t0; t0=$(date +%s)
  xargs -a "$rel" -n 20 -P 24 lfs hsm_restore
  local t1; t1=$(date +%s)
  echo "--- $label: queued $n files in $((t1-t0))s"
}

# refs first and by far the biggest (9.05 TB in 16,599 tars): it is what the
# dataloader touches randomly, so it is what starves the GPUs if it is cold.
preload_tree "$ROOT/refs/shards"  refs
preload_tree "$ROOT/codes"        codes
preload_tree "$ROOT/manifests"    manifests
preload_tree /fsx/models          models

echo "--- waiting for in-flight restores to settle ---"
# df stops growing when the queue drains. Two consecutive quiet samples, because a
# single one can land between batches.
quiet=0
while [ "$quiet" -lt 2 ]; do
  a=$(df -B1 /fsx | tail -1 | awk '{print $3}')
  sleep 30
  b=$(df -B1 /fsx | tail -1 | awk '{print $3}')
  d=$(( (b-a)/1048576 ))
  echo "$(date -u +%FT%TZ) +${d} MB/30s  used=$((b/1073741824)) GiB"
  if [ "$d" -lt 50 ]; then quiet=$((quiet+1)); else quiet=0; fi
done

# ---------------------------------------------------------- ref member sidecars
# Hydration is only half of the first-touch cost. The dataloader reads a ref by
# pread at a byte offset, and it gets those offsets from a per-shard sidecar in
# refs/.member-index -- which is NOT in the bucket, so it is built lazily by
# whichever rank first samples a speaker in that shard, by scanning the whole
# 539 MB tar. Measured: rank 0 parked in _try_get_data with every GPU at 100%
# util and 145 W, i.e. starved, while workers each pulled ~22 MB/s of tar headers.
#
# This is almost certainly what made a fresh fleet run 10.4 s/step and decay to
# 3.7 over hours: 16,599 shards' worth of scanning, paid during training, once per
# NODE. On FSx the sidecars are shared, so building them here costs ~2.4 TB of
# reads once for the fleet and no node ever scans a tar again.
#
# build_all skips shards that already have a sidecar, so this is resumable.
echo "--- ref member index (sidecars for pread offsets) ---"
if [ -d /data/t2s-repo ]; then
  ( cd /data/t2s-repo && PYTHONPATH=/data/t2s-repo /data/venv/bin/python \
      -m finetuning.ref_member_index "$ROOT/refs/shards" --workers 64 )
else
  echo "SKIP: /data/t2s-repo absent, cannot build member index here"
fi
echo "sidecars: $(ls "$ROOT/refs/.member-index" 2>/dev/null | wc -l) of 16599"

echo "--- residual released count ---"
find "$ROOT" /fsx/models -type f > /tmp/preload_all.txt
xargs -a /tmp/preload_all.txt -n 200 lfs hsm_state 2>/dev/null \
  | grep -c released || echo 0
df -h /fsx | tail -1
echo "=== $(date -u +%FT%TZ) fsx preload done ==="
