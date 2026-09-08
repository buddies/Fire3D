"""Shared cross-dataset helpers for voxelize_v2 scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from subprocess import DEVNULL

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import o_voxel
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
BLENDER_SCRIPT_DIR = REPO_ROOT / "data_processing" / "blender"
DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
DEFAULT_DEPENDENCY_ROOT = Path(
    os.environ.get("FIRE3D_DEPENDENCY_ROOT", REPO_ROOT / "data/dependencies")
)
DEFAULT_BLENDER_PATH = Path(
    os.environ.get(
        "FIRE3D_BLENDER",
        REPO_ROOT / "blender" / "blender-4.5.1-linux-x64" / "blender",
    )
)

torch.set_grad_enabled(False)


def load_raw_kit_utils(dataset_name: str, kit_dir: Path):
    utils_path = kit_dir / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing raw-kit utils: {utils_path}")
    module_name = f"fire3d_raw_kits_{dataset_name}_utils"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load raw-kit utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(kit_dir))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def blender_export_rotation_matrix() -> np.ndarray:
    """Rotation used by raw-kit GLB export before Blender attribute dumping."""
    return trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def rotation_degrees_label(degrees: float) -> str:
    normalized = float(degrees) % 360.0
    if abs(normalized - round(normalized)) < 1e-6:
        return str(int(round(normalized)))
    return (f"{normalized:.6f}".rstrip("0").rstrip(".")).replace(".", "p")


def local_z_rotation_matrix(degrees: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(np.deg2rad(float(degrees)), [0, 0, 1])


def apply_local_z_rotation(mesh_or_scene, degrees: float):
    out = mesh_or_scene.copy()
    out.apply_transform(local_z_rotation_matrix(degrees))
    return out


def save_glb(mesh_or_scene, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = mesh_or_scene.copy()
    out.apply_transform(blender_export_rotation_matrix())
    if isinstance(out, trimesh.Scene):
        out.export(save_path)
    else:
        trimesh.exchange.export.export_mesh(out, save_path)


def dump_with_blender(blender_path: Path, script_name: str, object_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / output_path.name
        cmd = [
            str(blender_path),
            "-b",
            "-P",
            str(BLENDER_SCRIPT_DIR / script_name),
            "--",
            "--object",
            str(object_path),
            "--output_path",
            str(temp_path),
        ]
        result = subprocess.run(cmd, stdout=DEVNULL, stderr=DEVNULL)
        if result.returncode == 0 and temp_path.exists():
            shutil.move(str(temp_path), str(output_path))
            return True
        return False


def vxz_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        o_voxel.io.read_vxz_info(str(path))
        return True
    except Exception:
        return False


def dual_grid_from_dump(mesh_dump_path: Path, output_path: Path, resolution: int) -> bool:
    if vxz_is_valid(output_path):
        return True
    with mesh_dump_path.open("rb") as f:
        dump = pickle.load(f)
    vertices = []
    faces = []
    start = 0
    for obj in dump["objects"]:
        if obj["vertices"].size == 0 or obj["faces"].size == 0:
            continue
        vertices.append(obj["vertices"])
        faces.append(obj["faces"] + start)
        start += len(obj["vertices"])
    if not vertices:
        return False
    vertices_t = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
    faces_t = torch.from_numpy(np.concatenate(faces, axis=0)).long()
    vmin = vertices_t.min(dim=0)[0]
    vmax = vertices_t.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    vertices_t = (vertices_t - center) * scale
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices_t,
        faces=faces_t,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    dual_vertices = torch.clamp(dual_vertices * resolution - voxel_indices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o_voxel.io.write_vxz(str(output_path), voxel_indices, {"vertices": dual_vertices, "intersected": intersected})
    return vxz_is_valid(output_path)


def pbr_from_dump(pbr_dump_path: Path, output_path: Path, resolution: int) -> bool:
    if vxz_is_valid(output_path):
        return True
    with pbr_dump_path.open("rb") as f:
        dump = pickle.load(f)
    for mat in dump["materials"]:
        if mat["alphaTexture"] is not None and mat["alphaMode"] == "OPAQUE":
            mat["alphaMode"] = "BLEND"
    dump["materials"].append(
        {
            "baseColorFactor": [0.8, 0.8, 0.8],
            "alphaFactor": 1.0,
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "baseColorTexture": None,
            "alphaTexture": None,
            "metallicTexture": None,
            "roughnessTexture": None,
        }
    )
    dump["objects"] = [obj for obj in dump["objects"] if obj["vertices"].size != 0 and obj["faces"].size != 0]
    if not dump["objects"]:
        return False
    vertices = torch.from_numpy(np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)).float()
    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    for obj in dump["objects"]:
        obj["vertices"] = ((torch.from_numpy(obj["vertices"]).float() - center) * scale).numpy()
        obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False,
    )
    attr.pop("normal", None)
    attr.pop("emissive", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o_voxel.io.write_vxz(str(output_path), coord, attr)
    return vxz_is_valid(output_path)


def write_records(records_path: Path, records: list[dict[str, object]]) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
