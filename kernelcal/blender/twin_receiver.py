"""Blender digital twin receiver — visualization side of the decoder pipeline.

Receives a synthesized twin (OBJ or NPZ payload) and applies:
  1. Base mesh from spectral decoder (skeleton + displacement)
  2. Procedural texture nodes driven by diagnostics (H[h*], E_curl, D_t)
  3. Curl-energy colormap on vertex colors (channel activity heatmap)
  4. Patch-request visual flag (red rim overlay when D_t > threshold)
  5. Optional: MeshLab-compatible PLY export for post-processing

Usage (headless, called by run_q10_experiment.sh or a ROS2 bridge):
  blender --background --python twin_receiver.py -- \
      --mesh_npz /tmp/twin_payload.npz \
      --diagnostics_json /tmp/twin_diag.json \
      --out_blend /tmp/twin_scene.blend \
      --out_obj /tmp/twin_rendered.obj

Usage (interactive, import into running Blender session):
  import importlib, sys
  sys.path.insert(0, "/path/to/kernelcal-repo")
  import blender_terrain_gen.twin_receiver as tr
  tr.load_and_display_from_npz("/tmp/twin_payload.npz", "/tmp/twin_diag.json")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Color, Vector

# Resolve kernelcal repo root: kernelcal/blender/ → two levels up
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Mesh creation from numpy arrays
# ---------------------------------------------------------------------------

def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)


def np_to_blender_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    name: str = "DigitalTwin",
) -> bpy.types.Object:
    """Create a Blender mesh object from numpy vertices and faces."""
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bverts = [bm.verts.new(Vector(v.tolist())) for v in vertices]
    bm.verts.ensure_lookup_table()
    for tri in faces:
        try:
            bm.faces.new([bverts[i] for i in tri])
        except ValueError:
            pass  # duplicate face — skip silently
    bm.to_mesh(mesh)
    bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    return obj


# ---------------------------------------------------------------------------
# Vertex color application (curl-energy heatmap)
# ---------------------------------------------------------------------------

def apply_vertex_colors(
    obj: bpy.types.Object,
    weights: np.ndarray,
    color_name: str = "CurlEnergy",
) -> None:
    """Paint per-vertex colors from a (V,) weight array [0,1].

    0 → blue  (spectral/gradient terrain)
    1 → red   (curl-active channel/flow region)
    """
    mesh = obj.data
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name=color_name)
    vcol = mesh.vertex_colors[color_name]

    # Build a vertex → loop index map
    loop_weights = np.zeros(len(mesh.loops), dtype=float)
    for loop in mesh.loops:
        loop_weights[loop.index] = weights[loop.vertex_index]

    for i, loop_col in enumerate(vcol.data):
        t = float(loop_weights[i])
        # Blue → cyan → green → yellow → red
        h = (1.0 - t) * 0.667   # hue: 0.667=blue, 0=red
        c = Color()
        c.hsv = (h, 0.9, 0.9)
        loop_col.color = (c.r, c.g, c.b, 1.0)


# ---------------------------------------------------------------------------
# Material: diagnostic-driven procedural shader
# ---------------------------------------------------------------------------

def build_diagnostic_material(
    diag: dict,
    name: str = "TwinMaterial",
) -> bpy.types.Material:
    """Create a Cycles/EEVEE material node tree driven by diagnostics.

    Nodes
    -----
    - VertexColor (CurlEnergy)   → Base color tinted by channel activity
    - NoiseTexture (roughness ← H[h*])  → micro-surface detail
    - MixShader (D_t gate)       → patch-request areas tinted red
    - PrincipledBSDF             → final output
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    roughness   = float(diag.get("roughness", 0.5))
    D_t         = float(diag.get("D_t", 0.0))
    patch_flag  = bool(diag.get("request_patch", False))
    curl_active = bool(diag.get("curl_active", False))

    # Output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = min(0.95, 0.3 + roughness * 0.5)
    bsdf.inputs["Base Color"].default_value = (0.55, 0.50, 0.45, 1.0)  # regolith grey-tan

    # Vertex color for curl heatmap
    vcol_node = nodes.new("ShaderNodeVertexColor")
    vcol_node.layer_name = "CurlEnergy"
    vcol_node.location = (-200, 100)

    # Noise texture for micro-roughness detail
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-200, -100)
    noise.inputs["Scale"].default_value     = 80.0 * (0.5 + roughness)
    noise.inputs["Detail"].default_value    = 8.0
    noise.inputs["Roughness"].default_value = 0.6 + roughness * 0.3
    noise.inputs["Distortion"].default_value = 0.2

    # Displacement node for micro-detail
    disp_node = nodes.new("ShaderNodeDisplacement")
    disp_node.location = (300, -250)
    disp_node.inputs["Scale"].default_value = 0.02 * (1.0 + roughness)

    # Mix vertex color and base color based on curl activity
    mix_color = nodes.new("ShaderNodeMixRGB")
    mix_color.blend_type = "MIX"
    mix_color.location = (50, 100)
    mix_color.inputs["Fac"].default_value = 0.4 if curl_active else 0.05

    # Patch request: overlay warning tint
    if patch_flag:
        patch_rgb = nodes.new("ShaderNodeRGB")
        patch_rgb.location = (-200, 300)
        patch_rgb.outputs[0].default_value = (0.8, 0.1, 0.1, 1.0)  # red
        mix_patch = nodes.new("ShaderNodeMixRGB")
        mix_patch.blend_type = "MIX"
        mix_patch.location = (50, 300)
        mix_patch.inputs["Fac"].default_value = min(0.5, D_t / 40.0)
        links.new(patch_rgb.outputs[0], mix_patch.inputs[2])
        links.new(mix_color.outputs[0], mix_patch.inputs[1])
        links.new(mix_patch.outputs[0], bsdf.inputs["Base Color"])
    else:
        links.new(mix_color.outputs[0], bsdf.inputs["Base Color"])

    # Wire up
    links.new(vcol_node.outputs["Color"], mix_color.inputs[2])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    links.new(noise.outputs["Fac"], disp_node.inputs["Height"])
    links.new(disp_node.outputs["Displacement"], out.inputs["Displacement"])

    mat.cycles.displacement_method = "DISPLACEMENT"
    return mat


# ---------------------------------------------------------------------------
# HUD text overlay (diagnostics in Blender viewport)
# ---------------------------------------------------------------------------

def add_diagnostics_text(diag: dict) -> bpy.types.Object:
    """Add a text object in the 3D scene summarising key diagnostics."""
    lines = [
        f"D_t       = {diag.get('D_t', 0):.3f}",
        f"H[h*]     = {diag.get('spectral_entropy', 0):.3f}",
        f"E_curl    = {diag.get('curl_energy', 0):.4f}",
        f"Detail    : {diag.get('synthesis_level', '?')}",
        f"Scene     : {diag.get('scene_class', '?')}",
        f"beta      : {diag.get('beta', '?')}",
        f"Patch req : {diag.get('request_patch', False)}",
    ]
    bpy.ops.object.text_add(location=(0, 0, 60))
    txt_obj = bpy.context.active_object
    txt_obj.name = "TwinDiagnostics"
    txt_obj.data.body = "\n".join(lines)
    txt_obj.data.size = 2.0
    return txt_obj


# ---------------------------------------------------------------------------
# Main loading entry points
# ---------------------------------------------------------------------------

def load_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_weights: np.ndarray,
    diagnostics: dict,
    *,
    out_blend: str | None = None,
    out_obj: str | None = None,
) -> bpy.types.Object:
    """Load decoded twin into Blender, apply material and vertex colors."""
    _clear_scene()

    obj = np_to_blender_mesh(vertices, faces, name="DigitalTwin_Base")
    apply_vertex_colors(obj, texture_weights)
    mat = build_diagnostic_material(diagnostics)
    obj.data.materials.append(mat)
    add_diagnostics_text(diagnostics)

    # Camera and lighting for headless render
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 200))
    bpy.ops.object.camera_add(location=(0, -300, 200))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.camera.rotation_euler = (0.8, 0, 0)

    if out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
        print(f"[receiver] Blend saved: {out_blend}")

    if out_obj:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.wm.obj_export(
            filepath=str(out_obj),
            export_selected_objects=True,
            export_triangulated_mesh=True,
            export_normals=True,
            export_uv=False,
            export_materials=False,
        )
        print(f"[receiver] OBJ exported: {out_obj}")

    return obj


def load_and_display_from_npz(
    npz_path: str,
    diagnostics_json: str | None = None,
    *,
    out_blend: str | None = None,
    out_obj: str | None = None,
) -> bpy.types.Object:
    """Load SynthesizedTwin data from NPZ + JSON diagnostic files."""
    z = np.load(npz_path, allow_pickle=True)
    vertices = np.asarray(z["vertices_detailed"], dtype=float)
    faces    = np.asarray(z["faces"], dtype=np.int32)
    weights  = np.asarray(z["texture_weights"], dtype=float)

    diag: dict = {}
    if diagnostics_json and Path(diagnostics_json).exists():
        diag = json.loads(Path(diagnostics_json).read_text())
    elif "diagnostics" in z.files:
        raw = z["diagnostics"].item()
        diag = raw if isinstance(raw, dict) else {}

    return load_from_arrays(
        vertices, faces, weights, diag,
        out_blend=out_blend,
        out_obj=out_obj,
    )


# ---------------------------------------------------------------------------
# CLI entry point (headless Blender)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Blender digital twin receiver")
    p.add_argument("--mesh_npz",          required=True)
    p.add_argument("--diagnostics_json",  default=None)
    p.add_argument("--out_blend",         default=None)
    p.add_argument("--out_obj",           default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    load_and_display_from_npz(
        args.mesh_npz,
        args.diagnostics_json,
        out_blend=args.out_blend,
        out_obj=args.out_obj,
    )
    print("[receiver] Done.")
