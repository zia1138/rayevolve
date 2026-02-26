#!/usr/bin/env bash

set -euo pipefail

# Script for launching a ray cluster on lighting AI studios multimachine clusters (MMT).
# See https://lightning.ai/docs/overview/multi-node-training/cli-commands.
# This should work in other similar cluster setups with a head node and worker nodes
# that is compatible with `torchrun`. 
#
# 1. Install lightnigng AI studios SDK within the uv environment:
#    cd rayevolve && uv sync
#    source .venv/bin/activate
#    uv pip install lightning-sdk
# 2. Authenticate logfire and select rayevolve project:
#    logfire auth
#    logfire projects use rayevolve
# 3. Then launch an MMT cluster using the following command:
#   `lightning run mmt --command="/teamspace/studios/this_studio/rayevolve/mmt_cluster.sh"`

: "${MASTER_ADDR:?need MASTER_ADDR}"

cd /teamspace/studios/this_studio/rayevolve

# Fail fast if nc is not installed
if ! command -v nc >/dev/null 2>&1; then
  echo "ERROR: 'nc' (netcat) is required but not installed or not in PATH." >&2
  exit 1
fi

RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
NODE_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
MASTER_IP="$(getent hosts "${MASTER_ADDR}" | awk '{print $1}' || echo "${MASTER_ADDR}")"

is_head=0
if [[ "${NODE_IP}" == "${MASTER_IP}" ]] || [[ "$(hostname)" == "${MASTER_ADDR}" ]]; then
  is_head=1
fi

wait_for_head() {
  until nc -z "${MASTER_IP}" "${RAY_PORT}" >/dev/null 2>&1; do
    echo "Waiting for Ray head at ${MASTER_IP}:${RAY_PORT}..."
    sleep 2
  done
}

if [[ "${is_head}" == "1" ]]; then
  echo "Starting Ray HEAD on ${NODE_IP}:${RAY_PORT}"
  uv run ray start --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}"
else
  wait_for_head
  echo "Starting Ray WORKER on ${NODE_IP} -> ${MASTER_IP}:${RAY_PORT}"
  uv run ray start \
    --node-ip-address="${NODE_IP}" \
    --address="${MASTER_IP}:${RAY_PORT}"
fi

echo "Ray cluster up. (Driver address: ${MASTER_IP}:${RAY_PORT})"
sleep infinity