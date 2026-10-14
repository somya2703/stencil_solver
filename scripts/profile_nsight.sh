#!/usr/bin/env bash
# scripts/profile_nsight.sh
#
# Wrapper for Nsight Compute (ncu) and Nsight Systems (nsys).
#
# Modes:
#   compute  — full Nsight Compute kernel profile
#   systems  — Nsight Systems end-to-end timeline
#   both     — compute + systems back-to-back
#   compare  — profile naive AND tiled kernels, print metric diff table
#
# Usage:
#   ./scripts/profile_nsight.sh compute  ./build/bin/stencil_naive --nx 512 --steps 1
#   ./scripts/profile_nsight.sh systems  ./build/bin/stencil_tiled --nx 512 --steps 50
#   ./scripts/profile_nsight.sh both     ./build/bin/stencil_naive --nx 256 --steps 1
#   ./scripts/profile_nsight.sh compare  ./build/bin/stencil_tiled --nx 512 --steps 1
#
# The 'compare' mode assumes stencil_naive lives in the same directory as stencil_tiled.
#
# Environment variables:
#   NCU_SET=quick|full       metric depth (default quick)
#   LAUNCH_COUNT=N           launches profiled per run (default 1)
#   KERNEL_FILTER=name       pass -k to ncu (compare sets kernel_naive/kernel_tiled)
#   CLOCK_CONTROL=none|base  ncu clock locking (default none; matches docs/CASE_STUDY.md)

set -euo pipefail

MODE="${1:-compute}"
shift

OUTDIR="nsight-output/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# ── Metrics collected by the compute mode ─────────────────────────────────────
# Covers the four Nsight pillars: memory traffic, occupancy, throughput, SMEM.
METRICS=(
    # DRAM traffic
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    # DRAM clock during the launch: dram__throughput % is normalised to it,
    # so record it next to the percentage (see docs/CASE_STUDY.md 3.7)
    "dram__cycles_elapsed.avg.per_second"
    # L1 hit rate
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
    "l1tex__t_sector_hit_rate.pct"
    # Throughput (SM and DRAM)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    # SMEM bank conflicts (tiled kernel)
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active"
    "sm__occupancy.avg.pct_of_peak_sustained_active"
)
METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

# ── Helper: run Nsight Compute ─────────────────────────────────────────────────
# NCU_SET controls the metric collection depth:
#   NCU_SET=quick (default) — only the METRICS array above, few passes
#   NCU_SET=full            — full section set (slow: ~39 passes/launch)
# Use `full` only for a one-off deep dive, not routine before/after comparisons.
#
# LAUNCH_COUNT limits ncu to profiling a single kernel launch instead of every
# launch in the run (warmup + timed). Without this, the default --warmup 10
# means ncu replays the *entire* --set full pass sequence 11 times, which is
# what produced the ~2800 ms/step "results" — a profiling artifact, not real
# timing. For a truly single-launch profile, also pass --warmup 0 --steps 1
# in the binary args.
NCU_SET="${NCU_SET:-quick}"
# ncu locks GPU clocks to base by default, which changes SM-bound kernels
# (tiled) far more than DRAM-bound ones (naive). "none" matches the
# docs/CASE_STUDY.md methodology; set CLOCK_CONTROL=base for locked clocks.
CLOCK_CONTROL="${CLOCK_CONTROL:-none}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"

run_compute() {
    local binary="$1"; shift
    local args=("$@")
    local name
    name=$(basename "$binary")

    local k_args=()
    if [[ -n "${KERNEL_FILTER:-}" ]]; then
        k_args=(-k "$KERNEL_FILTER")
    fi
    local set_args=()
    if [[ "$NCU_SET" == "full" ]]; then
        set_args=(--set full)
        echo "   (NCU_SET=full — this will be slow: ~39 passes per launch)"
    fi

    echo ""
    echo "── Nsight Compute: $name ${args[*]} ──"
    echo "   Output: $OUTDIR/${name}.ncu-rep"

    ncu \
        "${set_args[@]}" \
        "${k_args[@]}" \
        --clock-control "$CLOCK_CONTROL" \
        --metrics "$METRICS_STR" \
        --launch-count "$LAUNCH_COUNT" \
        --target-processes all \
        -o "$OUTDIR/${name}" \
        "$binary" "${args[@]}"

    echo "   Open: ncu-ui $OUTDIR/${name}.ncu-rep"
}

# ── Helper: run Nsight Systems ─────────────────────────────────────────────────
run_systems() {
    local binary="$1"; shift
    local args=("$@")
    local name
    name=$(basename "$binary")

    echo ""
    echo "── Nsight Systems: $name ${args[*]} ──"
    echo "   Output: $OUTDIR/${name}.nsys-rep"

    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=process-tree \
        --backtrace=dwarf \
        --cudabacktrace=all \
        --stats=true \
        --output "$OUTDIR/${name}" \
        "$binary" "${args[@]}"

    echo "   Open: nsys-ui $OUTDIR/${name}.nsys-rep"
}

# ── Helper: print metric diff table ───────────────────────────────────────────
print_diff_table() {
    local rep_naive="$1"
    local rep_tiled="$2"

    # ncu --import reads a .ncu-rep and exports metrics as CSV
    if ! command -v python3 &>/dev/null; then
        echo "python3 not found — skipping diff table"
        return
    fi

    echo ""
    echo "── Metric comparison (naive vs tiled) ────────────────────────────"

    python3 - "$rep_naive" "$rep_tiled" << 'PYEOF'
import subprocess, sys, csv, io

UNIT_SCALE = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6,
              "Gbyte": 1e9, "Tbyte": 1e12}

def _to_float(s):
    return float(s.replace(",", "").strip())

def extract(rep_path):
    """Run ncu --import and pivot its LONG-format CSV into {metric: value}.

    ncu --csv emits one row per (launch, metric) with generic columns
    "ID", "Metric Name", "Metric Unit", "Metric Value".  Metric names only
    appear as row *values*, never as column headers.  We therefore:
      - skip any ==PROF== preamble lines before the header,
      - keep only the last launch ID (the timed one, not a warmup),
      - strip thousands separators and normalise byte units to bytes,
      - return values as strings so the caller's truthiness test is safe
        for a legitimate 0.
    Falls back to the wide layout (metric names as columns) if present.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", rep_path, "--csv",
             "--metrics",
             "dram__bytes_read.sum,"
             "dram__bytes_write.sum,"
             "l1tex__t_sector_hit_rate.pct,"
             "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
             "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
             "sm__warps_active.avg.pct_of_peak_sustained_active"],
            capture_output=True, text=True, timeout=60)
        lines = result.stdout.splitlines()
        start = next((k for k, ln in enumerate(lines)
                      if ln.startswith('"ID"') or ln.startswith("ID,")), None)
        if start is None:
            return {}
        body = [ln for ln in lines[start:] if not ln.startswith("==")]
        rows = [r for r in csv.DictReader(io.StringIO("\n".join(body)))
                if r and r.get("ID") is not None]
        if not rows:
            return {}
        if "Metric Name" in rows[0]:
            last_id = rows[-1]["ID"]
            out = {}
            for r in rows:
                if r["ID"] != last_id:
                    continue
                name = (r.get("Metric Name") or "").strip()
                raw = (r.get("Metric Value") or "").strip()
                if not name or not raw:
                    continue
                try:
                    val = _to_float(raw)
                except ValueError:
                    print(f"  Note: {name} = {raw!r} in {rep_path}")
                    continue
                val *= UNIT_SCALE.get((r.get("Metric Unit") or "").strip(), 1.0)
                out[name] = repr(val)
            return out
        row = rows[-1]   # wide layout fallback
        return {(k.strip() if k else k): (v.strip() if v else v)
                for k, v in row.items()}
    except Exception as e:
        print(f"  Warning: could not extract metrics from {rep_path}: {e}")
        return {}

naive_path, tiled_path = sys.argv[1], sys.argv[2]
n = extract(naive_path)
t = extract(tiled_path)

if not n and not t:
    print("  (could not read .ncu-rep files — open manually in ncu-ui)")
    sys.exit(0)

rows = [
    ("DRAM reads (GB)",       "dram__bytes_read.sum",
                              lambda v: f"{float(v)/1e9:.2f}" if v else "—"),
    ("DRAM writes (GB)",      "dram__bytes_write.sum",
                              lambda v: f"{float(v)/1e9:.2f}" if v else "—"),
    ("L1 hit rate",           "l1tex__t_sector_hit_rate.pct",
                              lambda v: f"{float(v):.1f}%" if v else "—"),
    ("DRAM throughput % peak", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                              lambda v: f"{float(v):.1f}%" if v else "—"),
    ("SM throughput % peak",  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                              lambda v: f"{float(v):.1f}%" if v else "—"),
    ("Achieved occupancy",    "sm__warps_active.avg.pct_of_peak_sustained_active",
                              lambda v: f"{float(v):.1f}%" if v else "—"),
]

print(f"  {'Metric':<30}  {'Naive':>12}  {'Tiled':>12}")
print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
for label, key, fmt in rows:
    nv = fmt(n.get(key, "")) if n else "—"
    tv = fmt(t.get(key, "")) if t else "—"
    print(f"  {label:<30}  {nv:>12}  {tv:>12}")
PYEOF
}

# ── Mode dispatch ──────────────────────────────────────────────────────────────
case "$MODE" in

  compute|ncu)
    BINARY="$1"; shift
    run_compute "$BINARY" "$@"
    ;;

  systems|nsys)
    BINARY="$1"; shift
    run_systems "$BINARY" "$@"
    ;;

  both)
    BINARY="$1"; shift
    run_compute "$BINARY" "$@"
    run_systems "$BINARY" "$@"
    ;;

  compare)
    # Profile naive and tiled, then print diff table.
    # Expects either:
    #   - stencil_tiled binary → derives stencil_naive from same dir
    #   - stencil_naive binary → uses as-is and also profiles tiled
    BINARY="$1"; shift
    BINARY_ARGS=("$@")
    BINARY_DIR=$(dirname "$BINARY")
    BINARY_NAME=$(basename "$BINARY")

    # Derive the paired binary name
    if [[ "$BINARY_NAME" == *"tiled"* ]]; then
        NAIVE_BIN="$BINARY_DIR/stencil_naive"
        TILED_BIN="$BINARY"
    else
        NAIVE_BIN="$BINARY"
        TILED_BIN="$BINARY_DIR/stencil_tiled"
    fi

    echo "Profiling naive:  $NAIVE_BIN"
    echo "Profiling tiled:  $TILED_BIN"
    echo "Output dir:       $OUTDIR"
    echo ""

    # Profile naive
    if [[ -x "$NAIVE_BIN" ]]; then
        KERNEL_FILTER=kernel_naive run_compute "$NAIVE_BIN" "${BINARY_ARGS[@]}"
    else
        echo "Warning: $NAIVE_BIN not found — skipping naive profile"
    fi

    # Profile tiled
    if [[ -x "$TILED_BIN" ]]; then
        KERNEL_FILTER=kernel_tiled run_compute "$TILED_BIN" "${BINARY_ARGS[@]}"
    else
        echo "Warning: $TILED_BIN not found — skipping tiled profile"
    fi

    # Print diff table
    REP_NAIVE="$OUTDIR/stencil_naive.ncu-rep"
    REP_TILED="$OUTDIR/stencil_tiled.ncu-rep"
    if [[ -f "$REP_NAIVE" && -f "$REP_TILED" ]]; then
        print_diff_table "$REP_NAIVE" "$REP_TILED"
    fi

    echo ""
    echo "── Next steps ──────────────────────────────────────────────────"
    echo "  ncu-ui $REP_NAIVE"
    echo "  ncu-ui $REP_TILED"
    echo "  (open the Baselines panel in ncu-ui for a side-by-side diff)"
    ;;

  *)
    echo "Usage: $0 {compute|systems|both|compare} <binary> [args...]"
    echo ""
    echo "  compute  — Nsight Compute kernel profile"
    echo "  systems  — Nsight Systems end-to-end timeline"
    echo "  both     — compute + systems back-to-back"
    echo "  compare  — profile naive AND tiled, print metric diff table"
    exit 1
    ;;
esac
