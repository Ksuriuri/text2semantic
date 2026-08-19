#!/usr/bin/env bash
# One node of a multi-machine run. The same script runs on every node; only
# T2S_MACHINE_RANK differs.
#
# The interesting part is the network, and it is different on every cloud.
# Inter-node traffic here is ~8 GB of all-reduce per step, so a run that falls
# back to the ordinary VPC NIC stops being compute-bound. Both fabrics below
# degrade *silently* when their plugin is absent -- NCCL just uses TCP sockets
# and prints nothing above WARN -- so each branch hard-fails on its own
# prerequisites instead of letting a $400/h job run at a fraction of its speed.
#
#   T2S_FABRIC=tcpx  a3-highgpu-8g on GCP: 8 H100s behind five vNICs, eth0 for
#                    everything ordinary and eth1-eth4 for GPU traffic, reachable
#                    only through the GPUDirect-TCPX plugin.
#   T2S_FABRIC=efa   p5.48xlarge on AWS: 8 H100s behind 32 EFA devices
#                    (3200 Gbps), reachable through libfabric plus the
#                    aws-ofi-nccl plugin.
#   T2S_FABRIC=none  accept single-NIC inter-node traffic. On purpose only.
#
# The default stays tcpx, and the older T2S_TCPX=0 still means none, so existing
# GCP launches are unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${TEXT2SEMANTIC_PYTHON:-python3}"

# A stray quote inside ${var:?message} starts a new quoting context, so keep
# these messages plain.
MACHINE_RANK="${T2S_MACHINE_RANK:?set T2S_MACHINE_RANK, 0 for the node the others dial into}"
MAIN_IP="${T2S_MAIN_IP:?set T2S_MAIN_IP to the internal IP of rank 0}"
NUM_MACHINES="${T2S_NUM_MACHINES:-2}"
GPUS_PER_NODE="${T2S_GPUS_PER_NODE:-8}"
MAIN_PORT="${T2S_MAIN_PORT:-29500}"
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

FABRIC="${T2S_FABRIC:-}"
if [[ -z "$FABRIC" ]]; then
  if [[ "${T2S_TCPX:-1}" == "1" ]]; then FABRIC="tcpx"; else FABRIC="none"; fi
fi

# The bootstrap interface is per-cloud: GCP names it eth0, AWS names it ens* or
# enp*. Naming eth0 on a p5 would fail the rendezvous, so only default it where
# it exists and let the fabric branch below have the last word.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]] && [[ -d /sys/class/net/eth0 ]]; then
  export NCCL_SOCKET_IFNAME="eth0"
fi

if [[ "$FABRIC" == "tcpx" ]]; then
  TCPX_LIB="${T2S_TCPX_LIB:-}"
  if [[ -z "$TCPX_LIB" ]]; then
    for candidate in /var/lib/tcpx/lib64 /usr/local/tcpx/lib64; do
      [[ -d "$candidate" ]] && TCPX_LIB="$candidate" && break
    done
  fi
  if [[ -z "$TCPX_LIB" ]]; then
    echo "GPUDirect-TCPX plugin not found (looked in /var/lib/tcpx/lib64 and" >&2
    echo "/usr/local/tcpx/lib64). Install it, point T2S_TCPX_LIB at it, or set" >&2
    echo "T2S_FABRIC=none to accept single-NIC inter-node traffic." >&2
    exit 1
  fi
  # The receive-datapath-manager owns the flow-steering sockets the plugin
  # connects to; without it running, NCCL hangs at the first inter-node collective
  # rather than reporting a missing dependency.
  if [[ ! -S "${T2S_TCPX_SOCKET:-/run/tcpx}/rx_rule_manager" ]] &&
     ! ls "${T2S_TCPX_SOCKET:-/run/tcpx}" >/dev/null 2>&1; then
    echo "TCPX receive-datapath-manager socket dir ${T2S_TCPX_SOCKET:-/run/tcpx}" >&2
    echo "is missing. Start tcpgpudmarxd before launching, or set T2S_TCPX=0." >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$TCPX_LIB:${LD_LIBRARY_PATH:-}"
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="${T2S_TCPX_SOCKET:-/run/tcpx}"
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${T2S_TCPX_DATA_IFNAMES:-eth1,eth2,eth3,eth4}"
  export NCCL_GPUDIRECTTCPX_CTRL_DEV="${T2S_TCPX_CTRL_IFNAME:-eth0}"
  # Documented a3-highgpu-8g core sets: each data NIC is pinned to the NUMA node
  # its GPUs hang off, tx and rx on separate cores. Re-derive these from lscpu if
  # the machine type or image ever changes.
  export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
  export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=500000
  export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
  export NCCL_CROSS_NIC=0
  export NCCL_ALGO=Ring
  export NCCL_PROTO=Simple
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_SOCKET_NTHREADS=1
  export NCCL_MAX_NCHANNELS=12
  export NCCL_MIN_NCHANNELS=12
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export NCCL_BUFFSIZE=4194304
  export NCCL_NET_GDR_LEVEL=PIX
elif [[ "$FABRIC" == "efa" ]]; then
  # aws-ofi-nccl ships libnccl-net.so (older) or libnccl-net-ofi.so (newer). NCCL
  # finds the first by name on LD_LIBRARY_PATH but needs NCCL_NET_PLUGIN for the
  # second, so locate the actual file rather than assuming either layout.
  OFI_LIB="${T2S_OFI_NCCL_LIB:-}"
  OFI_PLUGIN=""
  for candidate in "$OFI_LIB" /opt/amazon/ofi-nccl/lib /opt/amazon/ofi-nccl/lib64 \
                   /opt/aws-ofi-nccl/lib /opt/aws-ofi-nccl/lib64 /usr/local/lib; do
    [[ -n "$candidate" ]] || continue
    # Globs are not expanded inside [[ ]], hence the loop.
    for so in "$candidate"/libnccl-net*.so; do
      if [[ -f "$so" ]]; then
        OFI_LIB="$candidate"
        OFI_PLUGIN="$so"
        break 2
      fi
    done
  done
  if [[ -z "$OFI_PLUGIN" ]]; then
    echo "aws-ofi-nccl plugin not found (looked for libnccl-net*.so in" >&2
    echo "/opt/amazon/ofi-nccl/lib{,64}, /opt/aws-ofi-nccl/lib{,64}, /usr/local/lib)." >&2
    echo "Install it, point T2S_OFI_NCCL_LIB at its lib dir, or set T2S_FABRIC=none" >&2
    echo "to accept single-NIC inter-node traffic." >&2
    exit 1
  fi

  # Without a working efa provider libfabric has nothing to offer and NCCL drops
  # to sockets. fi_info exits non-zero when no device matches, which is the one
  # cheap pre-launch check that catches a missing EFA interface in the launch
  # template, a missing kernel module, and a security group that does not allow
  # the all-traffic-to-itself rule EFA needs.
  FI_INFO="${T2S_FI_INFO:-}"
  if [[ -z "$FI_INFO" ]]; then
    for candidate in /opt/amazon/efa/bin/fi_info "$(command -v fi_info || true)"; do
      [[ -n "$candidate" && -x "$candidate" ]] && FI_INFO="$candidate" && break
    done
  fi
  if [[ -z "$FI_INFO" ]]; then
    echo "fi_info not found; install the EFA installer or set T2S_FI_INFO." >&2
    exit 1
  fi
  EFA_DEVICES="$("$FI_INFO" -p efa 2>/dev/null | grep -c '^provider: efa' || true)"
  if [[ "${EFA_DEVICES:-0}" -eq 0 ]]; then
    echo "no EFA device: $FI_INFO -p efa found none. Check that the launch" >&2
    echo "template attaches EFA interfaces, that the efa kernel module is" >&2
    echo "loaded, and that the security group allows all traffic to itself." >&2
    exit 1
  fi

  # EFA registers pinned memory directly; a bounded RLIMIT_MEMLOCK surfaces much
  # later as an ibv_reg_mr failure mid-run. systemd-run needs
  # --property=LimitMEMLOCK=infinity, docker needs --ulimit memlock=-1.
  LOCKED="$(ulimit -l)"
  if [[ "$LOCKED" != "unlimited" && "${T2S_ALLOW_MEMLOCK_LIMIT:-0}" != "1" ]]; then
    echo "RLIMIT_MEMLOCK is ${LOCKED} kB, not unlimited: EFA memory registration" >&2
    echo "will fail. Pass LimitMEMLOCK=infinity (systemd) or memlock=-1 (docker)," >&2
    echo "or set T2S_ALLOW_MEMLOCK_LIMIT=1 to try anyway." >&2
    exit 1
  fi

  export LD_LIBRARY_PATH="$OFI_LIB:${T2S_EFA_LIB:-/opt/amazon/efa/lib}:${LD_LIBRARY_PATH:-}"
  if [[ "$(basename "$OFI_PLUGIN")" != "libnccl-net.so" ]]; then
    export NCCL_NET_PLUGIN="$OFI_PLUGIN"
  fi
  export FI_PROVIDER="${FI_PROVIDER:-efa}"
  # H100 nodes read and write peer GPU memory from the device itself; without
  # this libfabric bounces every message through host memory.
  export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
  # The dataloader forks workers after libfabric has registered memory. Newer
  # libfabric handles that itself and ignores this; older builds corrupt the
  # registration cache without it.
  export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
  # Deliberately no NCCL_ALGO/NCCL_PROTO/channel counts here, unlike the TCPX
  # block above: on p5 the plugin's own defaults are what AWS tunes for, and
  # pinning Simple costs bandwidth. Tune from a measurement, not from a guess.
  echo "EFA: ${EFA_DEVICES} device(s), plugin ${OFI_PLUGIN}" >&2
  echo "confirm NCCL actually used it with NCCL_DEBUG=INFO" \
       "NCCL_DEBUG_SUBSYS=INIT,NET and grep for NET/OFI." >&2
elif [[ "$FABRIC" == "none" ]]; then
  echo "T2S_FABRIC=none: inter-node NCCL will use ${NCCL_SOCKET_IFNAME:-its own choice} only." >&2
else
  echo "unknown T2S_FABRIC=${FABRIC} (expected tcpx, efa or none)" >&2
  exit 1
fi

echo "node ${MACHINE_RANK}/${NUM_MACHINES}, ${NUM_PROCESSES} processes," \
     "fabric ${FABRIC}, main ${MAIN_IP}:${MAIN_PORT}"
exec "$PY" -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_ip "$MAIN_IP" \
  --main_process_port "$MAIN_PORT" \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  "$ROOT/finetuning/train.py" "$@"
