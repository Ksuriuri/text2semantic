#!/bin/bash
# Keep asking us-east-2a for the next p5 node the run wants, one RunInstances at
# a time, and log what AWS says.
#
# Why this exists separately from the supervisor: spot_supervisor.py repairs the
# node set it was given, and while any of them is MISSING it refuses to launch
# (half a job is worse than none). So the 8-node supervisor was a perfect
# capacity probe that could never start training, and the 2-node supervisor now
# training can never notice that capacity came back. This is the second half.
#
# Total batch is held at 3072 rows/step as batch x 8 GPUs x nodes x accum, so the
# usable node counts depend on the batch, and batch is now 32 rather than 48
# because 48 could not survive a step at 16 ranks (see node-config-production.sh).
# At 32, nodes x accum must be 12: 1, 2, 3, 4, 6 and 12 nodes all work, so p5-2
# alone is already a cutover rather than dead weight.
#
# The list runs to p5-5 as a deliberate bet, taken at 11:05Z: 5 nodes is not legal
# at batch 32, so a 5th node bills $20/h contributing nothing until a 6th arrives.
# The bet is on the capacity that just reappeared -- p5-2 at 10:42 and p5-3 at
# 11:03, two nodes in 21 minutes after an hour of InsufficientInstanceCapacity --
# holding long enough for the 6th. Trim back to p5-3 if the 5th sits idle for an
# hour or two, because the idle node is pure loss while the run is not using it.
#
# Now chasing p5-6 and p5-7, i.e. 8 nodes, which kusuriuri ordered directly:
# "有4台或8台就等保存最新权重之后立马往上切" (`71d9fdf4`). Eight nodes is not legal at
# batch 32, so it means going back to batch 48 x accum 1 -- and the 6-node
# measurement is what says that is worth the marginal memory rather than a
# regression: **the per-micro-batch fixed cost dominates**. Measured 11.5 s/step at
# 6 nodes x batch 32 x accum 2 against the 8-node run's 3.79 at batch 48 x accum 1,
# and the single-node bench has +50% rows (32 -> 48) costing only +15% time. So
# what a shape really costs is the number of micro-batches per step, and accum 2
# pays that fixed cost twice. If batch 48 OOMs at 64 ranks after all, the safe
# 8-node shape is batch 24 x accum 2.
set -uo pipefail

REGION=us-east-2
TEMPLATE=t2s-p5-spot
INTERVAL="${INTERVAL:-300}"
TARGETS=(noiz-t2s-p5-6 noiz-t2s-p5-7)

exists() {
  local found
  found=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$1" \
              "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null)
  [[ -n "$found" ]]
}

echo "$(date -Is) probing ${TARGETS[*]} every ${INTERVAL}s"
while true; do
  for name in "${TARGETS[@]}"; do
    if exists "$name"; then continue; fi
    # One attempt per cycle, in order: the aim is to notice capacity, not to win
    # a race against other accounts, and a tight retry loop on
    # InsufficientInstanceCapacity buys nothing.
    if out=$(aws ec2 run-instances --region "$REGION" \
        --launch-template "LaunchTemplateName=$TEMPLATE" --count 1 \
        --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
        --query 'Instances[].InstanceId' --output text 2>&1); then
      echo "$(date -Is) GOT $name $out -- staging starts now, about 55 min"
    else
      echo "$(date -Is) no capacity for $name: $(echo "$out" | tail -1 | cut -c1-160)"
    fi
    break
  done
  sleep "$INTERVAL"
done
