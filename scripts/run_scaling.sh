#!/usr/bin/env bash
# scripts/run_scaling.sh — drive weak and strong scaling experiments.
#
# Requires: stencil_multigpu built with MPI enabled.
# Usage:    ./scripts/run_scaling.sh [--type weak|strong|both] [--max-gpus N]

set -euo pipefail

BINARY="${BINARY:-./build/bin/stencil_multigpu}"
MAX_GPUS="${MAX_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
SCALING_TYPE="${SCALING_TYPE:-both}"
STEPS=200
BASE_N=256          # base grid size per GPU (weak scaling)
STRONG_N=512        # fixed total grid (strong scaling)
OUTDIR="results"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)    SCALING_TYPE="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --outdir)  OUTDIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_case() {
    local ngpu="$1" nx="$2" ny="$3" nz="$4" label="$5"
    echo "  GPUs=$ngpu  grid=${nx}x${ny}x${nz}  label=$label"
    mpirun -n "$ngpu" "$BINARY" \
        --nx "$nx" --ny "$ny" --nz "$nz" \
        --steps "$STEPS" \
        --output "$OUTDIR/${label}_gpu${ngpu}_${TIMESTAMP}.json" \
        2>&1 | tail -3
}

if [[ "$SCALING_TYPE" == "weak" || "$SCALING_TYPE" == "both" ]]; then
    echo "=== Weak scaling (grid grows with GPU count) ==="
    for ngpu in 1 2 4 8; do
        [[ "$ngpu" -gt "$MAX_GPUS" ]] && break
        # Each GPU owns BASE_N^3 cells; double NZ per doubling of GPUs
        nz=$(( BASE_N * ngpu ))
        run_case "$ngpu" "$BASE_N" "$BASE_N" "$nz" "weak"
    done
fi

if [[ "$SCALING_TYPE" == "strong" || "$SCALING_TYPE" == "both" ]]; then
    echo "=== Strong scaling (fixed ${STRONG_N}^3 grid) ==="
    for ngpu in 1 2 4 8; do
        [[ "$ngpu" -gt "$MAX_GPUS" ]] && break
        run_case "$ngpu" "$STRONG_N" "$STRONG_N" "$STRONG_N" "strong"
    done
fi

echo ""
echo "Results written to: $OUTDIR/"
echo "Plot with: python3 scripts/plot_results.py --scaling $OUTDIR/"
