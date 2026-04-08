#!/bin/bash
# Run this on the server to merge + plot results from a completed run.
# Usage: bash merge_landauer_results.sh [results_dir]
# Default results_dir: $HOME/landauer_results

RESULTS_DIR=${1:-$HOME/landauer_results}
echo "Merging from: $RESULTS_DIR"

python3 - "$RESULTS_DIR" <<'PY'
import json, sys
sys.path.insert(0, '.')
from kernelcal.attention.landauer import _generate_landauer_figures
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
        print(f'  WARNING: {f} not found')

if not all_results:
    print('ERROR: no results found'); sys.exit(1)

merged = {'config': config or {}, 'results': all_results}
out = base / 'landauer_results_merged.json'
out.write_text(json.dumps(merged, indent=2))
print(f'Merged {len(all_results)} runs → {out}')

_generate_landauer_figures(all_results, base)
print(f'Figures saved → {base}/fig_landauer_results.pdf')
PY
