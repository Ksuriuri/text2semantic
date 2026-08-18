#!/usr/bin/env bash
# One node of a multi-machine run. The same script runs on every node; only
# T2S_MACHINE_RANK differs.
#
# On a3-highgpu-8g the interesting part is the network. Each node holds 8 H100s
# behind five vNICs: eth0 for everything ordinary and eth1-eth4 for GPU traffic.
# NCCL only uses the latter through the GPUDirect-TCPX plugin, and without it it
# quietly falls back to eth0 alone -- one gVNIC for what is ~8 GB of all-reduce
# per step, which turns a compute-bound job into a network-bound one. So the
# plugin is required by default and T2S_TCPX=0 has to be set on purpose.
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
# One rendezvous timeout longer than a node takes to come back from a restart,
# so a straggler joins instead of failing the whole group.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

if [[ "${T2S_TCPX:-1}" == "1" ]]; then
  TCPX_LIB="${T2S_TCPX_LIB:-}"
  if [[ -z "$TCPX_LIB" ]]; then
    for candidate in /var/lib/tcpx/lib64 /usr/local/tcpx/lib64; do
      [[ -d "$candidate" ]] && TCPX_LIB="$candidate" && break
    done
  fi
  if [[ -z "$TCPX_LIB" ]]; then
    echo "GPUDirect-TCPX plugin not found (looked in /var/lib/tcpx/lib64 and" >&2
    echo "/usr/local/tcpx/lib64). Install it, point T2S_TCPX_LIB at it, or set" >&2
    echo "T2S_TCPX=0 to accept single-NIC inter-node traffic." >&2
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
else
  echo "T2S_TCPX=0: inter-node NCCL will use ${NCCL_SOCKET_IFNAME} only." >&2
fi

echo "node ${MACHINE_RANK}/${NUM_MACHINES}, ${NUM_PROCESSES} processes, main ${MAIN_IP}:${MAIN_PORT}"
exec "$PY" -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_ip "$MAIN_IP" \
  --main_process_port "$MAIN_PORT" \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  "$ROOT/finetuning/train.py" "$@"
