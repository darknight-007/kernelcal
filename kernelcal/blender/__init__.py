"""kernelcal.blender — Blender/bpy integration for spectral digital twins.

Strict separation of concerns
------------------------------
This subpackage owns everything that requires Blender's Python runtime (bpy,
mathutils, bmesh).  It does NOT compute eigenvalues, Betti numbers, or kernel
updates — all spectral and topological computation belongs in kernelcal.geo3d.

Modules
-------
terrain_gen     Procedural planetary terrain generation with controlled β₁ ground
                truth; exports OBJ + JSON sidecar for kernelcal Q10 experiments.
                Requires: bpy, bmesh, mathutils (Blender runtime).

twin_receiver   Receiving-side visualiser: loads a SynthesizedTwin NPZ payload,
                applies procedural shader nodes driven by H[h*] and E_curl, adds
                a per-vertex curl heatmap, and optionally exports OBJ / .blend.
                Requires: bpy, bmesh, mathutils (Blender runtime).

q10_pipeline    Nyström β₁ topology error experiment (no bpy dependency — pure
                kernelcal.geo3d).  Reads the terrain_gen JSON sidecar, runs the
                three-stage decoder, and reports Q10 pass/fail.
                Requires: kernelcal.geo3d only.

run_q10_experiment.sh
                End-to-end orchestrator: calls Blender headlessly for terrain_gen,
                then invokes q10_pipeline.py via the system Python with PYTHONPATH
                set to the repo root.  No manual steps; arXiv-replicable.

Usage
-----
Blender scripts must be invoked via the Blender executable, not imported by a
normal Python interpreter:

    blender --background --python \
        $(python3 -c "import kernelcal.blender; import os; \
                      print(os.path.dirname(kernelcal.blender.__file__))") \
        /terrain_gen.py -- --n_loops 3 --out_dir /tmp/q10

Q10 pipeline (pure Python, no Blender):

    python3 -m kernelcal.blender.q10_pipeline \
        --sidecar /tmp/q10/ground_truth_loops3_craters5.json

Full end-to-end experiment:

    BLENDER=/path/to/blender \\
    python3 -c "import kernelcal.blender, os, subprocess; \
                d = os.path.dirname(kernelcal.blender.__file__); \
                subprocess.run([d+'/run_q10_experiment.sh','--all_loops'])"

Or directly:

    BLENDER=/path/to/blender ./kernelcal/blender/run_q10_experiment.sh --all_loops

Important: bpy modules are only importable inside Blender's own Python runtime.
Importing this package from a regular Python interpreter will succeed (the
__init__.py has no bpy import at module level), but attempting to import
terrain_gen or twin_receiver outside Blender will raise ImportError on bpy.
"""

# Nothing is imported at package level: bpy is unavailable outside Blender.
# Use kernelcal.blender.q10_pipeline directly for the pure-Python pipeline.
