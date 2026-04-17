#!/usr/bin/env bash
# run_q10_experiment.sh
#
# End-to-end Q10 Nyström topology error experiment.
# Fully reproducible, no manual steps — suitable for arXiv replication.
#
# Pipeline:
#   1. Blender (headless) generates terrain OBJ files + ground-truth JSON sidecar
#   2. kernelcal q10_pipeline.py runs Nyström beta_1 estimation per resolution
#   3. Q10 pass/fail verdict is printed and exit code reflects result
#
# Usage:
#   ./run_q10_experiment.sh [options]
#
# Options:
#   --blender PATH    path to Blender executable  (default: blender)
#   --out_dir DIR     output directory            (default: /tmp/q10_terrains)
#   --n_loops N       number of ring loops        (default: 3)
#   --n_craters N     number of craters           (default: 5)
#   --resolutions     comma-separated resolutions (default: 32,64,128,256)
#   --n_modes N       Nyström modes               (default: 16)
#   --n_coarse N      Nyström coarse points       (default: 300)
#   --seed N          RNG seed                    (default: 42)
#   --all_loops       run n_loops in {3,5,13}     (paper benchmark set)
#
# Exit codes:
#   0 — all Q10 checks passed
#   1 — at least one Q10 check failed
#   2 — dependency error

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
BLENDER="${BLENDER:-blender}"
OUT_DIR="/tmp/q10_terrains"
N_LOOPS=3
N_CRATERS=5
RESOLUTIONS="32,64,128,256"
N_MODES=16
N_COARSE=300
SEED=42
ALL_LOOPS=false

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blender)    BLENDER="$2";      shift 2 ;;
        --out_dir)    OUT_DIR="$2";      shift 2 ;;
        --n_loops)    N_LOOPS="$2";      shift 2 ;;
        --n_craters)  N_CRATERS="$2";    shift 2 ;;
        --resolutions) RESOLUTIONS="$2"; shift 2 ;;
        --n_modes)    N_MODES="$2";      shift 2 ;;
        --n_coarse)   N_COARSE="$2";     shift 2 ;;
        --seed)       SEED="$2";         shift 2 ;;
        --all_loops)  ALL_LOOPS=true;    shift ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

GEN_SCRIPT="$(realpath "$SCRIPT_DIR/terrain_gen.py")"
PIPELINE_SCRIPT="$(realpath "$SCRIPT_DIR/q10_pipeline.py")"

# kernelcal lives two levels above kernelcal/blender/ — always run pipeline
# with PYTHONPATH pointing at the repo root so the local editable install is
# importable without a system-wide pip install.
REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"

# ── Dependency checks ────────────────────────────────────────────────────────
if ! command -v "$BLENDER" &>/dev/null; then
    echo "ERROR: Blender not found at '$BLENDER'."
    echo "       Set the BLENDER env var or pass --blender /path/to/blender"
    exit 2
fi
if ! python3 -c "import kernelcal" &>/dev/null; then
    echo "ERROR: kernelcal not importable. Install it (pip install -e .) first."
    exit 2
fi

mkdir -p "$OUT_DIR"

# ── Loop set ─────────────────────────────────────────────────────────────────
if $ALL_LOOPS; then
    LOOP_SET=(3 5 13)
else
    LOOP_SET=($N_LOOPS)
fi

OVERALL_EXIT=0

for N_L in "${LOOP_SET[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Q10 experiment  n_loops=${N_L}  n_craters=${N_CRATERS}"
    echo "════════════════════════════════════════════════════════"

    # ── Step 1: Blender terrain generation ───────────────────────────────────
    echo "[1/2] Generating terrain meshes via Blender (headless)…"
    "$BLENDER" --background --python "$GEN_SCRIPT" -- \
        --out_dir "$OUT_DIR" \
        --n_loops "$N_L" \
        --n_craters "$N_CRATERS" \
        --resolutions "$RESOLUTIONS" \
        --seed "$SEED"

    SIDECAR="$OUT_DIR/ground_truth_loops${N_L}_craters${N_CRATERS}.json"
    if [[ ! -f "$SIDECAR" ]]; then
        echo "ERROR: Expected sidecar not found: $SIDECAR"
        OVERALL_EXIT=1
        continue
    fi

    # ── Step 2: kernelcal Q10 pipeline ───────────────────────────────────────
    echo "[2/2] Running kernelcal Q10 pipeline…"
    REPORT="$OUT_DIR/q10_report_loops${N_L}.json"
    set +e
    # Prepend the repo root to PYTHONPATH so kernelcal is importable without
    # a system-wide pip install.
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$PIPELINE_SCRIPT" \
        --sidecar "$SIDECAR" \
        --n_modes "$N_MODES" \
        --n_coarse "$N_COARSE" \
        --seed "$SEED" \
        --out "$REPORT"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "  → Q10 PASS  (n_loops=${N_L})"
    else
        echo "  → Q10 FAIL  (n_loops=${N_L})"
        OVERALL_EXIT=1
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $OVERALL_EXIT -eq 0 ]]; then
    echo "  ALL Q10 CHECKS PASSED"
else
    echo "  ONE OR MORE Q10 CHECKS FAILED — see reports in $OUT_DIR"
fi
echo "  Reports: $OUT_DIR/"
echo "════════════════════════════════════════════════════════"

exit $OVERALL_EXIT
