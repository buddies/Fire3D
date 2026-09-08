"""SceneSmith-specific helpers for voxelize_v2 scripts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
DEFAULT_SCENESMITH_ROOT = Path(
    os.environ.get("FIRE3D_SCENESMITH_ROOT", DEFAULT_TRAINING_ROOT / "Scenesmith")
)
SAVE_GLB_ROTATION = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0])
SAVE_GLB_ROTATION_INV = np.linalg.inv(SAVE_GLB_ROTATION)


def safe_name(name: str) -> str:
    return quote(str(name).replace(" ", "_").replace("|", "_").replace("/", "_"), safe="")


def scene_dir_for(root: Path, room_name: str) -> Path:
    return root / "scenes" / room_name


def transform_path_for(root: Path, room_name: str) -> Path:
    return root / "transforms" / f"{room_name}.pkl"


def render_dir_for(root: Path, room_name: str) -> Path:
    return root / "renders" / room_name


def has_render_payload(render_dir: Path, trajectory_id: str = "0") -> bool:
    return (
        (render_dir / f"{trajectory_id}.json").is_file()
        and (render_dir / f"{trajectory_id}_frames").is_dir()
        and (render_dir / f"{trajectory_id}_depth").is_dir()
        and (render_dir / f"{trajectory_id}_masks").is_dir()
    )


def load_transform_dict(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"SceneSmith transform pkl must contain a dict, got {type(data)!r}: {path}")
    return data


def entry_to_matrix(entry: dict[str, Any]) -> np.ndarray:
    if "matrix" in entry:
        matrix = np.asarray(entry["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix entry must be 4x4, got {matrix.shape}")
        return matrix

    scale = float(entry.get("scale", 1.0))
    angles = np.asarray(entry.get("angles", [0.0, 0.0, 0.0]), dtype=np.float64)
    trans = np.asarray(entry.get("trans", [0.0, 0.0, 0.0]), dtype=np.float64)
    if angles.shape != (3,):
        raise ValueError(f"angles must have shape (3,), got {angles.shape}")
    if trans.shape != (3,):
        raise ValueError(f"trans must have shape (3,), got {trans.shape}")
    return trimesh.transformations.compose_matrix(
        scale=[scale, scale, scale],
        angles=angles,
        translate=trans,
    )


def matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def collect_room_names(args: argparse.Namespace) -> list[str]:
    if args.room_name:
        room_names = [str(room_name) for room_name in args.room_name]
    elif args.room_list:
        room_names = [line.strip() for line in args.room_list.read_text().splitlines() if line.strip()]
    else:
        scenes_dir = args.scenesmith_root / "scenes"
        room_names = sorted(path.name for path in scenes_dir.iterdir() if path.is_dir()) if scenes_dir.is_dir() else []

    if getattr(args, "require_renders", False):
        room_names = [
            room_name
            for room_name in room_names
            if has_render_payload(render_dir_for(args.scenesmith_root, room_name), args.trajectory_id)
        ]
    room_names = sorted(room_names)
    if args.limit_rooms is not None:
        room_names = room_names[: args.limit_rooms]
    return room_names


def split_room_names(room_names: list[str], rank: int, world_size: int) -> list[str]:
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world_size")
    start = len(room_names) * rank // world_size
    end = len(room_names) * (rank + 1) // world_size
    return room_names[start:end]


def layout_latent(room_name: str, transforms: dict[str, dict[str, Any]] | None = None) -> str:
    if transforms:
        for key, value in transforms.items():
            latent = str(value.get("latent", key))
            if latent.startswith("layout_"):
                return latent
    return f"layout_{room_name}"


def object_latents(transforms: dict[str, dict[str, Any]]) -> list[str]:
    latents = []
    for key, value in transforms.items():
        latent = str(value.get("latent", key))
        if not latent.startswith("layout_"):
            latents.append(latent)
    return sorted(latents)


def object_asset_id(room_name: str, latent: str) -> str:
    return safe_name(f"{room_name}__{latent}")


def copy_normalized_glb(src: Path, dst: Path | None, mode: str = "copy") -> str | None:
    if dst is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve() or dst.exists():
        return str(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return str(dst)
        except OSError:
            shutil.copy2(src, dst)
            return str(dst)
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return str(dst)
        except OSError:
            shutil.copy2(src, dst)
            return str(dst)
    if mode != "copy":
        raise ValueError(f"Unsupported normalized GLB dump mode: {mode}")
    shutil.copy2(src, dst)
    return str(dst)


def validator_transform_entries(room_name: str, transforms: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key, value in transforms.items():
        latent = str(value.get("latent", key))
        is_bg = latent.startswith("layout_") or str(value.get("mesh_name", "")) == "bg"
        entry: dict[str, object] = {
            "name": latent,
            "latent": latent,
            "mesh_name": value.get("mesh_name"),
            "matrix": matrix_payload(entry_to_matrix(value)),
        }
        if is_bg:
            entry["asset_id"] = None
        else:
            entry["asset_id"] = object_asset_id(room_name, latent)
            entry["source_name"] = value.get("mesh_name")
        entries.append(entry)
    return entries


def write_validator_transforms(room_name: str, transforms: dict[str, dict[str, Any]], output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_name(room_name)}.json"
    path.write_text(json.dumps(validator_transform_entries(room_name, transforms), indent=2) + "\n")
    return path


def default_start_frame(render_dir: Path, trajectory_id: str = "0") -> int:
    depth_dir = render_dir / f"{trajectory_id}_depth"
    indices = []
    for path in depth_dir.glob("depth_*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return min(indices) if indices else 0
