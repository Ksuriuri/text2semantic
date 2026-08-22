#!/bin/bash
# Start spot_supervisor.py for a t2s-v1 run. Runs on the supervisor box, inside
# the VPC, because the training nodes have no public address (32 EFA interfaces).
#
#   run_supervisor.sh 2                 # preflight: two nodes
#   run_supervisor.sh 8                 # the real launch
#   run_supervisor.sh 2 --once --dry-run  # show the plan and stop
#
# The supervisor creates the nodes itself when they are MISSING, which is the same
# code path a preemption takes, so launching this way exercises the repair path
# from the first tick instead of only after the first interruption.
set -euo pipefail

COUNT="${1:?usage: run_supervisor.sh <node-count> [extra supervisor args]}"
shift || true

REGION=us-east-2
TEMPLATE=t2s-p5-spot
UNIT=t2s-train
KEY=/home/ec2-user/.ssh/noiz-t2s.pem
REPO=/home/ec2-user/t2s-repo

NODES=()
for i in $(seq 0 $((COUNT - 1))); do NODES+=(--node "noiz-t2s-p5-$i"); done

# -u because this is always read through a pipe (tee to a log, or tmux), and
# block-buffered stdout means the one thing being watched -- what the loop decided
# about which node -- appears in 8 KB batches, hours after the decision.
exec python3 -u "$REPO/scripts/spot_supervisor.py" \
  --cloud aws --region "$REGION" \
  "${NODES[@]}" \
  --instance-template "$TEMPLATE" \
  --ssh-user ubuntu --ssh-key "$KEY" \
  --unit "$UNIT" \
  --poll-seconds 60 \
  --setenv T2S_FABRIC=efa \
  --launch-command 'bash /opt/t2s/train_multinode_wrapper.sh' \
  "$@"
