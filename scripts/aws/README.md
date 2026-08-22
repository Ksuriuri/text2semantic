# AWS p5 Spot helpers

Runbook: [docs/aws-p5-spot-training.md](../../docs/aws-p5-spot-training.md).

| script | what it does |
|---|---|
| `setup_network.sh` | private subnet, NAT, S3 gateway, EFA SG, placement group |
| `make_launch_template.py` | create / version `t2s-p5-spot` from `node_userdata.sh` |
| `node_userdata.sh` | node bootstrap (FSx bind-mounts, apt/needrestart guards, READY) |
| `node-config.sh` | run-specific env + `T2S_TRAIN_ARGS` (upload to `s3://.../_staging/`) |
| `fsx_preload.sh` | one-time HSM hydration + member-index sidecars |
| `setup_supervisor.sh` | bastion IAM + t3.small |
| `run_supervisor.sh` | start `scripts/spot_supervisor.py --cloud aws` |
| `capacity_probe.sh` | mid-run one-node capacity chase; not for the first 8-node launch |

`spot_supervisor.py` and `train_multinode.sh` stay in `scripts/`. This directory is only the AWS box around them.
