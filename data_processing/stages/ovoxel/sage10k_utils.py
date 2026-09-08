"""SAGE-10k-specific helpers for voxelize_v2 scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from utils import DEFAULT_TRAINING_ROOT, REPO_ROOT, load_raw_kit_utils

RAW_SAGE10K_KIT = REPO_ROOT / "data_processing/datasets/sage10k"
DEFAULT_SAGE10K_ROOT = Path(
    os.environ.get("FIRE3D_SAGE10K_ROOT", DEFAULT_TRAINING_ROOT / "SAGE-10k")
)
BG_PREFIXES = ("floor_", "wall_", "window_", "door_")


def status_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return path.read_text(errors="ignore").lstrip().startswith("completed")
    except OSError:
        return False


def has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def dependency_state(args: argparse.Namespace) -> dict[str, bool]:
    zips_present = args.scene_zip_dir.is_dir() and any(args.scene_zip_dir.glob("*.zip"))
    extracted_present = args.extracted_dir.is_dir() and any(args.extracted_dir.glob("*/layout_*.json"))
    return {
        "sage10k_status_completed_or_scene_sources_present": status_completed(args.sage10k_status)
        or zips_present
        or extracted_present,
        "sage10k_scene_sources_present": zips_present or extracted_present,
        "blender_present": args.blender_path.exists() and os.access(args.blender_path, os.X_OK),
    }


def wait_for_dependencies(args: argparse.Namespace) -> None:
    while True:
        state = dependency_state(args)
        missing = [name for name, ok in state.items() if not ok]
        if not missing:
            print("All SAGE-10k voxelization dependencies are ready.", flush=True)
            return
        if not args.wait_for_deps:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")
        print(f"Waiting for dependencies: {', '.join(missing)}", flush=True)
        time.sleep(args.poll_seconds)


def normalize_scene_id(scene_id_or_path: str) -> str:
    name = Path(scene_id_or_path.strip()).name
    if name.endswith(".zip"):
        name = name[:-4]
    return name


def layout_id_from_scene_id(scene_id: str) -> str:
    match = re.search(r"(layout_[^/\\]+)$", scene_id)
    if match:
        return match.group(1)
    return scene_id[-len("layout_xxxxxxxx") :]


def collect_scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene_id:
        scene_ids = [normalize_scene_id(scene_id) for scene_id in args.scene_id]
    elif args.scene_list:
        scene_ids = [normalize_scene_id(line) for line in args.scene_list.read_text().splitlines() if line.strip()]
    else:
        scene_ids = []
        if args.scene_zip_dir.is_dir():
            scene_ids.extend(path.stem for path in args.scene_zip_dir.glob("*.zip"))
        if args.extracted_dir.is_dir():
            scene_ids.extend(
                path.name
                for path in args.extracted_dir.iterdir()
                if path.is_dir() and (path / f"{layout_id_from_scene_id(path.name)}.json").exists()
            )
        scene_ids = sorted(set(scene_ids))
    scene_ids = sorted(scene_ids)
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]
    return scene_ids


def scene_dir_has_layout(scene_dir: Path, scene_id: str) -> bool:
    return (scene_dir / f"{layout_id_from_scene_id(scene_id)}.json").exists() or any(scene_dir.glob("layout_*.json"))


def append_jsonl(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def safe_extract_zip(
    zip_path: Path,
    output_dir: Path,
    *,
    skip_bad_members: bool = False,
    bad_member_log: Path | None = None,
) -> None:
    output_root = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = output_dir / member.filename
            resolved_target = target.resolve()
            if output_root != resolved_target and output_root not in resolved_target.parents:
                raise ValueError(f"Unsafe path in archive {zip_path}: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            except zipfile.BadZipFile as exc:
                if target.exists():
                    target.unlink()
                append_jsonl(
                    bad_member_log,
                    {
                        "zip_path": str(zip_path),
                        "output_dir": str(output_dir),
                        "member": member.filename,
                        "error": repr(exc),
                    },
                )
                if skip_bad_members:
                    print(
                        f"WARNING: skipped corrupt zip member {member.filename} in {zip_path}: {exc}",
                        flush=True,
                    )
                    continue
                raise


def resolve_scene_dir(scene_id: str, args: argparse.Namespace) -> Path:
    scene_dir = args.extracted_dir / scene_id
    if scene_dir_has_layout(scene_dir, scene_id):
        return scene_dir

    zip_path = args.scene_zip_dir / f"{scene_id}.zip"
    if zip_path.exists() and args.extract_zips:
        safe_extract_zip(
            zip_path,
            scene_dir,
            skip_bad_members=getattr(args, "skip_bad_zip_members", False),
            bad_member_log=getattr(args, "bad_zip_members_log", None),
        )
        if scene_dir_has_layout(scene_dir, scene_id):
            return scene_dir

    if zip_path.exists() and not args.extract_zips:
        raise FileNotFoundError(f"Scene {scene_id} is zipped but --no-extract-zips was set: {zip_path}")
    raise FileNotFoundError(f"Could not find extracted scene data for {scene_id}; expected {scene_dir} or {zip_path}")


def load_raw_kit_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(RAW_SAGE10K_KIT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def load_sage10k_modules():
    sage_utils = load_raw_kit_utils("sage10k", RAW_SAGE10K_KIT)
    tex_utils = load_raw_kit_module("fire3d_raw_kits_sage10k_tex_utils_local", RAW_SAGE10K_KIT / "tex_utils_local.py")
    glb_utils = load_raw_kit_module("fire3d_raw_kits_sage10k_glb_utils", RAW_SAGE10K_KIT / "glb_utils.py")
    return sage_utils, tex_utils, glb_utils


def load_scene_mesh_dict(scene_dir: Path, scene_id: str):
    sage_utils, tex_utils, _glb_utils = load_sage10k_modules()
    layout_id = layout_id_from_scene_id(scene_id)
    layout_json_path = scene_dir / f"{layout_id}.json"
    if not layout_json_path.exists():
        candidates = sorted(scene_dir.glob("layout_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No layout JSON found under {scene_dir}")
        layout_json_path = candidates[0]
    with layout_json_path.open("r") as f:
        layout_data = json.load(f)
    layout = sage_utils.dict_to_floor_plan(layout_data)
    return tex_utils.export_layout_to_mesh_dict_list(layout, str(scene_dir))


def background_mesh_ids(mesh_dict: dict[str, object]) -> list[str]:
    return sorted(mesh_id for mesh_id in mesh_dict if mesh_id.startswith(BG_PREFIXES))


def build_trimesh_scene(mesh_dict: dict[str, object], mesh_ids: list[str]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for mesh_id in mesh_ids:
        mesh = mesh_dict[mesh_id]["mesh"].copy()
        scene.add_geometry(mesh, geom_name=mesh_id)
    return scene


def save_sage_textured_glb(
    mesh_dict: dict[str, object],
    mesh_ids: list[str],
    save_path: Path,
    transform: np.ndarray | None = None,
) -> None:
    _sage_utils, _tex_utils, glb_utils = load_sage10k_modules()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scene = glb_utils.create_glb_scene()
    for mesh_id in mesh_ids:
        mesh_data = mesh_dict[mesh_id]
        texture = mesh_data["texture"]
        texture_map_path = Path(texture["texture_map_path"])
        if texture_map_path.exists():
            texture_image = np.asarray(Image.open(texture_map_path).convert("RGB"))
        else:
            texture_image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        pbr_parameters = texture.get("pbr_parameters") or {}
        vertices = np.asarray(mesh_data["mesh"].vertices, dtype=np.float32)
        if transform is not None:
            vertices = trimesh.transformations.transform_points(vertices, transform).astype(np.float32)
        mesh_data_dict = {
            "vertices": vertices,
            "faces": np.asarray(mesh_data["mesh"].faces, dtype=np.uint32),
            "vts": np.asarray(texture["vts"], dtype=np.float32).copy(),
            "fts": np.asarray(texture["fts"], dtype=np.uint32),
            "texture_image": texture_image,
            "metallic_factor": texture.get("metallic_factor", pbr_parameters.get("metallic", 0.0)),
            "roughness_factor": texture.get("roughness_factor", pbr_parameters.get("roughness", 1.0)),
        }
        glb_utils.add_textured_mesh_to_glb_scene(
            mesh_data_dict,
            scene,
            material_name=f"material_{mesh_id}",
            mesh_name=f"mesh_{mesh_id}",
            preserve_coordinate_system=True,
        )
    glb_utils.save_glb_scene(str(save_path), scene)
