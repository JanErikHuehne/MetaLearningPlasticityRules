#!/usr/bin/env bash
# launch_sbi_runs.sh
# Usage:
#   ./launch_sbi_runs.sh <TOTAL_MEM> [HOST_NAS_PATH]
# Examples:
#   ./launch_sbi_runs.sh 12G /path/to/nas
#   ./launch_sbi_runs.sh 8000M

set -euo pipefail

IMAGE="jan--sbi-inital-runs:py310"
MEM_PER_CONTAINER_MB=800
CPUS_PER_CONTAINER=5

if [[ ${1:-} == "" ]]; then
  echo "Usage: $0 <TOTAL_MEM (e.g., 12G | 8000M)> [HOST_NAS_PATH]"
  exit 1
fi

TOTAL_MEM_RAW="$1"
HOST_NAS_PATH="${2:-}"



to_mb() {
  # Convert strings like 12G, 12GB, 8000M, 1.5G to integer MB
  local s="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
  if [[ "$s" =~ ^[0-9]+(\.[0-9]+)?g(b)?$ ]]; then
    awk -v v="${s%%g*}" 'BEGIN { printf "%.0f\n", v*1024 }'
  elif [[ "$s" =~ ^[0-9]+(\.[0-9]+)?m(b)?$ ]]; then
    awk -v v="${s%%m*}" 'BEGIN { printf "%.0f\n", v }'
  elif [[ "$s" =~ ^[0-9]+$ ]]; then
    # bare number: interpret as MB
    echo "$s"
  else
    echo "Error: could not parse memory value '$1' (use e.g. 12G or 8000M)" >&2
    exit 2
  fi
}


TOTAL_MEM_MB="$(to_mb "$TOTAL_MEM_RAW")"
if (( TOTAL_MEM_MB < MEM_PER_CONTAINER_MB )); then
  echo "Total memory ($TOTAL_MEM_MB MB) is less than per-container limit ($MEM_PER_CONTAINER_MB MB). Nothing to launch."
  exit 0
fi

AVAILABLE_CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"


MAX_BY_MEM=$(( TOTAL_MEM_MB / MEM_PER_CONTAINER_MB ))
MAX_BY_CPU=$(( AVAILABLE_CORES / CPUS_PER_CONTAINER ))



echo $TOTAL_MEM_RAW
echo $HOST_NAS_PATH
echo $AVAILABLE_CORES
echo $MAX_BY_MEM 
echo $MAX_BY_CPU

NUM_CONTAINERS=$(( MAX_BY_MEM < MAX_BY_CPU ? MAX_BY_MEM : MAX_BY_CPU ))
if (( NUM_CONTAINERS < 1 )); then
  echo "Computed 0 containers to launch."
  exit 0
fi

echo "Pullig image: $IMAGE"
docker_get $IMAGE 
docker tag "cortex:5000/$IMAGE" "$IMAGE"


echo "Planning to launch $NUM_CONTAINERS container(s):"
echo "  - Per container: ${MEM_PER_CONTAINER_MB} MB RAM, ${CPUS_PER_CONTAINER} CPU"
echo "  - Constrained by: memory=$MAX_BY_MEM, cpu=$MAX_BY_CPU"

MOUNT_ARGS=()

MOUNT_ARGS=( -v "${HOST_NAS_PATH}:/nas" )


for i in $(seq 1 "$NUM_CONTAINERS"); do
  NAME=$(printf "sbi-run-%02d" "$i")
  echo "Starting $NAME ..."
  docker run -d --rm \
    --name "$NAME" \
    --cpus="${CPUS_PER_CONTAINER}" \
    --memory="${MEM_PER_CONTAINER_MB}m" \
    --memory-swap="${MEM_PER_CONTAINER_MB}m" \
    "${MOUNT_ARGS[@]}" \
    "$IMAGE" >/dev/null
done

echo "Done. Running containers:"