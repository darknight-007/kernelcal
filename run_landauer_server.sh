#!/bin/bash
# ── Run Landauer experiment on 2×Titan RTX server ────────────────────────
# Usage: bash run_landauer_server.sh [--steps N] [--output-dir DIR]
#
# Steps:
#  1. Builds Docker image  (once, ~3 min first time)
#  2. Launches two containers in parallel, one per GPU
#  3. Merges results when both finish
#  4. Generates combined figure
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

STEPS=${STEPS:-2000}
RESULTS_DIR=${RESULTS_DIR:-$HOME/landauer_results}
N_SEEDS=${N_SEEDS:-3}
PRIME=${PRIME:-53}

echo "========================================"
echo " kernelcal Landauer experiment"
echo " STEPS=$STEPS  SEEDS=$N_SEEDS  PRIME=$PRIME"
echo " RESULTS=$RESULTS_DIR"
echo "========================================"

# Ensure results dir exists on host
mkdir -p "$RESULTS_DIR/gpu0" "$RESULTS_DIR/gpu1"

# Check NVIDIA runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Docker runtime not detected. Falling back to CPU power estimate."
fi

# Build image
echo "[1/4] Building Docker image..."
docker build -f Dockerfile.landauer -t kernelcal-landauer . \
    --build-arg BUILDKIT_INLINE_CACHE=1

# Launch both GPUs in parallel
echo "[2/4] Launching containers (GPU 0: d=128,256 | GPU 1: d=512,1024)..."
echo ""
echo ">>> MANUAL STEP: Record wall-plug kWh reading NOW (before experiment)"
echo "    Smart plug / PDU / IPMI reading: _____ kWh"
echo "    Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "    (Record this value; you will be prompted again at the end)"
echo ""
WALL_KWH_BEFORE=""
if [ -t 0 ]; then   # only prompt if interactive terminal
    read -rp "Enter wall-plug kWh before [skip with Enter]: " WALL_KWH_BEFORE
fi
echo "$WALL_KWH_BEFORE" > "$RESULTS_DIR/wall_kwh_before.txt"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$RESULTS_DIR/wall_kwh_before.txt"

docker run --rm --gpus '"device=0"' \
    -v "$RESULTS_DIR":/results/landauer \
    kernelcal-landauer \
    --widths 128 256 \
    --lrs 1e-2 1e-3 1e-4 1e-5 \
    --steps "$STEPS" --n-seeds "$N_SEEDS" --prime "$PRIME" \
    --device-id 0 --output-dir /results/landauer/gpu0 &
PID0=$!

docker run --rm --gpus '"device=1"' \
    -v "$RESULTS_DIR":/results/landauer \
    kernelcal-landauer \
    --widths 512 1024 \
    --lrs 1e-2 1e-3 1e-4 1e-5 \
    --steps "$STEPS" --n-seeds "$N_SEEDS" --prime "$PRIME" \
    --device-id 0 --output-dir /results/landauer/gpu1 &
PID1=$!

echo "  GPU0 container PID: $PID0"
echo "  GPU1 container PID: $PID1"

wait $PID0 && echo "[GPU0 done]"
wait $PID1 && echo "[GPU1 done]"

echo ""
echo ">>> MANUAL STEP: Record wall-plug kWh reading NOW (after experiment)"
echo "    Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
WALL_KWH_AFTER=""
if [ -t 0 ]; then
    read -rp "Enter wall-plug kWh after [skip with Enter]: " WALL_KWH_AFTER
fi
echo "$WALL_KWH_AFTER" > "$RESULTS_DIR/wall_kwh_after.txt"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$RESULTS_DIR/wall_kwh_after.txt"

# Compute wall delta if both values present
if [ -n "$WALL_KWH_BEFORE" ] && [ -n "$WALL_KWH_AFTER" ]; then
    python3 -c "
before=$WALL_KWH_BEFORE; after=$WALL_KWH_AFTER
delta=after-before
print(f'Wall-plug delta: {delta:.4f} kWh  ({delta*3600:.1f} kJ)')
print(f'GPU-only share: see landauer_results_merged.json for W_gpu totals')
" | tee "$RESULTS_DIR/wall_delta.txt"
fi

# Merge results
echo "[3/4] Merging results..."
python3 - "$RESULTS_DIR" <<'PY'
import json, sys
from pathlib import Path

base = Path(sys.argv[1])
all_results = []
config = None

for part in ['gpu0', 'gpu1']:
    f = base / part / 'landauer_results.json'
    if f.exists():
        d = json.loads(f.read_text())
        all_results.extend(d['results'])
        config = d['config']
        print(f'  Loaded {len(d["results"])} runs from {part}')
    else:
        print(f'  WARNING: {f} not found, skipping')

if not all_results:
    print('ERROR: no results found to merge')
    sys.exit(1)

merged = {'config': config or {}, 'results': all_results}
out = base / 'landauer_results_merged.json'
out.write_text(json.dumps(merged, indent=2))
print(f'Merged {len(all_results)} runs → {out}')
PY

# Generate combined figure
echo "[4/4] Generating combined figure..."
python3 - "$RESULTS_DIR" <<'PY'
import json, sys
sys.path.insert(0, '.')
from kernelcal.attention.landauer import _generate_landauer_figures
from pathlib import Path

base = Path(sys.argv[1])
merged_file = base / 'landauer_results_merged.json'
data = json.loads(merged_file.read_text())
_generate_landauer_figures(data['results'], base)
print(f'Figure saved → {base}/fig_landauer_results.pdf')
PY

echo "========================================"
echo " Experiment complete!"
echo " Results: $RESULTS_DIR/landauer_results_merged.json"
echo " Figure:  $RESULTS_DIR/fig_landauer_results.pdf"
echo "========================================"
