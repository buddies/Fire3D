"""MansionWorld-specific helpers for voxelize_v2 scripts."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import trimesh

from utils import DEFAULT_BLENDER_PATH, DEFAULT_TRAINING_ROOT, REPO_ROOT, load_raw_kit_utils

RAW_MANSIONWORLD_KIT = REPO_ROOT / "data_processing/datasets/mansionworld"
DEFAULT_MANSIONWORLD_ROOT = Path(
    os.environ.get("FIRE3D_MANSIONWORLD_ROOT", DEFAULT_TRAINING_ROOT / "MansionWorld")
)
DEFAULT_AI2THOR_ASSET_DIR = Path(
    os.environ.get(
        "FIRE3D_AI2THOR_ASSET_DIR",
        DEFAULT_TRAINING_ROOT / "ProcTHOR/ai2thor-hab/assets",
    )
)
OBJATHOR_VERSION = "2023_09_23"


def status_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return path.read_text(errors="ignore").lstrip().startswith("completed")
    except OSError:
        return False


def dependency_state(args: argparse.Namespace) -> dict[str, bool]:
    objathor_assets = args.objathor_root / OBJATHOR_VERSION / "assets"
    objathor_materials = args.objathor_root / "holodeck" / OBJATHOR_VERSION / "materials" / "images"
    ai2thor_objects = args.ai2thor_asset_dir / "objects"
    scene_dir = args.mansionworld_root / "mansionworld"
    return {
        "objathor_status_completed": status_completed(args.objathor_status),
        "objathor_assets_present": objathor_assets.is_dir() and any(objathor_assets.iterdir()),
        "objathor_materials_present": objathor_materials.is_dir() and any(objathor_materials.iterdir()),
        "ai2thor_status_completed": status_completed(args.ai2thor_status),
        "ai2thor_objects_present": ai2thor_objects.is_dir() and any(ai2thor_objects.iterdir()),
        "scene_jsons_present": scene_dir.is_dir() and any(scene_dir.glob("*/floor_*.json")),
        "blender_present": args.blender_path.exists() and os.access(args.blender_path, os.X_OK),
    }


def wait_for_dependencies(args: argparse.Namespace) -> None:
    while True:
        state = dependency_state(args)
        missing = [name for name, ok in state.items() if not ok]
        if not missing:
            print("All MansionWorld voxelization dependencies are ready.", flush=True)
            return
        if not args.wait_for_deps:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")
        print(f"Waiting for dependencies: {', '.join(missing)}", flush=True)
        time.sleep(args.poll_seconds)


def configure_mansionworld_import(args: argparse.Namespace):
    os.environ["MANSIONWORLD_ROOT"] = str(args.mansionworld_root)
    os.environ["OBJATHOR_ROOT"] = str(args.objathor_root)
    os.environ["AI2THOR_ASSET_DIR"] = str(args.ai2thor_asset_dir)
    return load_raw_kit_utils("mansionworld", RAW_MANSIONWORLD_KIT)


def mansionworld_asset_orientation_matrix() -> np.ndarray:
    """Map raw asset coordinates into MansionWorld's canonical object convention."""
    return np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def load_canonical_asset_mesh(asset_id: str, mansion_utils):
    asset_mesh = mansion_utils.get_asset_mesh(asset_id)
    if asset_mesh is None:
        return None

    asset_mesh = asset_mesh.copy()
    asset_mesh.apply_transform(mansionworld_asset_orientation_matrix())
    mesh_for_bounds = trimesh.util.concatenate(asset_mesh.dump()) if isinstance(asset_mesh, trimesh.Scene) else asset_mesh
    if len(mesh_for_bounds.vertices) == 0:
        return None

    center, scale = mansion_utils.normalize_mesh(mesh_for_bounds)
    center_matrix = np.eye(4)
    center_matrix[:3, 3] = -center
    scale_matrix = np.eye(4)
    scale_matrix[np.diag_indices(3)] = scale
    asset_mesh.apply_transform(center_matrix)
    asset_mesh.apply_transform(scale_matrix)
    return asset_mesh
