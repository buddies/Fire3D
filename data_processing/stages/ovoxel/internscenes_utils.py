"""InternScenes helpers for voxelize_v2 scripts.

The raw-kit composer keeps the dataset-specific asset orientation rules.  This
module wraps those rules and adds the v2 conventions: deterministic asset IDs,
normalized GLB dumping, and validator transform JSON entries.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
DEFAULT_OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
DEFAULT_INTERNSCENES_ROOT = Path(
    os.environ.get("FIRE3D_INTERNSCENES_ROOT", DEFAULT_TRAINING_ROOT / "InternScenes")
)
DEFAULT_RAW_KIT_DIR = REPO_ROOT / "data_processing/datasets/internscenes"
DEFAULT_SPLIT_JSON = Path(
    os.environ.get(
        "FIRE3D_SCENE_SPLIT",
        REPO_ROOT / "data/training_manifests/scene_train.json",
    )
)
DEFAULT_POSTPROCESS_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_POSTPROCESS_ROOT", DEFAULT_OBJECT_ROOT / "postprocess")
)
DEFAULT_DATASET_NAME = "internscenes"


def safe_name(value: str) -> str:
    return quote(str(value).replace(" ", "_").replace("|", "_").replace("/", "_"), safe="")


def matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def scene_name_to_safe(scene_name: str) -> str:
    return safe_name(scene_name)


def render_dir_for(root: Path, scene_name: str) -> Path:
    return root / "renders" / scene_name.replace("/", "_")


def layout_dir_for(layout_root: Path, scene_name: str) -> Path:
    return layout_root / scene_name


def scene_name_from_split_name(split_name: str) -> str | None:
    prefix = "InternScenes_"
    if not split_name.startswith(prefix):
        return None
    raw = split_name[len(prefix) :]
    if raw.startswith("3rscan_"):
        return "3rscan/" + raw[len("3rscan_") :]
    if raw.startswith("scannet_"):
        return "scannet/" + raw[len("scannet_") :]
    if raw.startswith("arkitscenes_Training_"):
        return "arkitscenes/Training/" + raw[len("arkitscenes_Training_") :]
    if raw.startswith("arkitscenes_Validation_"):
        return "arkitscenes/Validation/" + raw[len("arkitscenes_Validation_") :]
    if raw.startswith("matterport3d_"):
        parts = raw.split("_")
        if len(parts) >= 3 and parts[-1].startswith("region"):
            scan = parts[1]
            region = parts[-1]
            return f"matterport3d/{scan}/{region}"
    return raw.replace("_", "/")


def split_name_from_scene_name(scene_name: str) -> str:
    return "InternScenes_" + scene_name.replace("/", "_")


def collect_split_scene_names(split_json: Path, limit_scenes: int | None = None) -> list[str]:
    data = json.loads(split_json.read_text())
    scene_names = data.get("scene_names") if isinstance(data, dict) else data
    if not isinstance(scene_names, list):
        raise ValueError(f"Unsupported split JSON format: {split_json}")
    out: list[str] = []
    seen: set[str] = set()
    for item in scene_names:
        if not isinstance(item, str):
            continue
        scene_name = scene_name_from_split_name(item)
        if scene_name is None or scene_name in seen:
            continue
        seen.add(scene_name)
        out.append(scene_name)
        if limit_scenes is not None and len(out) >= limit_scenes:
            break
    return out


def collect_scene_names(args: argparse.Namespace) -> list[str]:
    if getattr(args, "scene_name", None):
        scene_names = [str(x) for x in args.scene_name]
    elif getattr(args, "scene_list", None):
        scene_names = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    elif getattr(args, "split_json", None) is not None and getattr(args, "from_split", False):
        scene_names = collect_split_scene_names(args.split_json, limit_scenes=args.limit_scenes)
    else:
        list_path = args.internscenes_root / "downloaded" / "Scenes_info" / "scene_name_list_final.json"
        scene_names = json.loads(list_path.read_text())
        if not isinstance(scene_names, list):
            raise ValueError(f"Expected a scene-name list in {list_path}")
        scene_names = [str(x) for x in scene_names]
        if args.limit_scenes is not None:
            scene_names = scene_names[: args.limit_scenes]

    if getattr(args, "require_renders", False):
        scene_names = [
            scene_name
            for scene_name in scene_names
            if has_render_payload(render_dir_for(args.internscenes_root, scene_name), args.trajectory_id)
        ]
    return sorted(scene_names)


def has_render_payload(render_dir: Path, trajectory_id: str = "0") -> bool:
    return (
        (render_dir / f"{trajectory_id}.json").is_file()
        and (render_dir / f"{trajectory_id}_frames").is_dir()
        and (render_dir / f"{trajectory_id}_depth").is_dir()
        and (render_dir / f"{trajectory_id}_masks").is_dir()
    )


def split_items(items: list[str], rank: int, world_size: int) -> list[str]:
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world_size")
    start = len(items) * rank // world_size
    end = len(items) * (rank + 1) // world_size
    return items[start:end]


def ensure_layout_extracted(scene_name: str, layout_root: Path, layout_tar: Path, extract_layout: bool) -> Path:
    scene_dir = layout_dir_for(layout_root, scene_name)
    if (scene_dir / "layout.json").exists() and (scene_dir / "StructureMesh").is_dir():
        return scene_dir
    if not extract_layout:
        raise FileNotFoundError(f"Missing extracted InternScenes layout: {scene_dir}")
    if not layout_tar.exists():
        raise FileNotFoundError(f"Missing InternScenes layout archive: {layout_tar}")

    prefix = f"Layout_info/{scene_name}/"
    layout_root.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(layout_tar, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.startswith(prefix)]
        if not members:
            raise FileNotFoundError(f"Scene {scene_name!r} not found in {layout_tar}")
        for member in members:
            target = (layout_root.parent / member.name).resolve()
            if not str(target).startswith(str(layout_root.resolve())):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(path=layout_root.parent, members=members)
    return scene_dir


def load_raw_composer_module(raw_kit_dir: Path, internscenes_root: Path, layout_root: Path):
    module_path = raw_kit_dir / "compose_scenes.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing InternScenes raw-kit composer: {module_path}")
    module_name = "fire3d_raw_kits_internscenes_compose_scenes"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(raw_kit_dir))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path

    downloaded = internscenes_root / "downloaded"
    module.BASE_DIR = str(internscenes_root)
    module.ASSET_LIBRARY_FOLDER = str(downloaded / "asset_library")
    module.SCENE_SAVE_DIR = str(internscenes_root / "composed_scenes")
    module.SCENE_INFO_DIR = str(layout_root)
    return module


def composer_for(raw_kit_dir: Path, internscenes_root: Path, layout_root: Path):
    module = load_raw_composer_module(raw_kit_dir, internscenes_root, layout_root)
    return module.SceneComposer(), module


def scene_or_mesh_to_mesh(mesh_or_scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        dumped = mesh_or_scene.dump(concatenate=False)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
        meshes = [mesh for mesh in dumped if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) and len(mesh.faces)]
        if not meshes:
            raise ValueError("Scene contains no mesh geometry")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported geometry type: {type(mesh_or_scene)!r}")


def normalize_transform_for_geometry(mesh_or_scene) -> tuple[np.ndarray, np.ndarray]:
    mesh = scene_or_mesh_to_mesh(mesh_or_scene)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    extent = float((bounds[1] - bounds[0]).max())
    if extent <= 0:
        raise ValueError("geometry has zero extent and cannot be normalized")
    scale = 0.99999 / extent
    center_matrix = np.eye(4, dtype=np.float64)
    center_matrix[:3, 3] = -center
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[np.diag_indices(3)] = scale
    normalizer = scale_matrix @ center_matrix
    return normalizer, np.linalg.inv(normalizer)


def instance_object_asset_id(scene_name: str, instance_id: int | str) -> str:
    return safe_name(f"{scene_name_to_safe(scene_name)}__object_{int(instance_id):04d}")


def model_asset_id(model_uid: str) -> str:
    return safe_name(model_uid)


def mesh_path_for_model_uid(asset_root: Path, model_uid: str) -> Path:
    if model_uid.startswith("objaverse/"):
        return asset_root / f"{model_uid}.glb"
    if model_uid.startswith("objaverse_old/"):
        return asset_root / f"{model_uid}.glb"
    if model_uid.startswith("partnet_mobility"):
        return asset_root / model_uid / "whole.glb"
    if model_uid.startswith("3D-FUTURE-model"):
        return asset_root / f"{model_uid}.glb"
    if model_uid.startswith("hssd-models"):
        return asset_root / f"{model_uid}.glb"
    if model_uid.startswith("gen_assets"):
        return asset_root / f"{model_uid}.glb"
    if model_uid.startswith("gr100"):
        return asset_root / f"{model_uid}.glb"
    raise ValueError(f"Unsupported InternScenes model_uid: {model_uid}")


def model_asset_exists(internscenes_root: Path, model_uid: str) -> bool:
    return mesh_path_for_model_uid(internscenes_root / "downloaded" / "asset_library", model_uid).exists()


def load_layout(scene_name: str, layout_root: Path) -> list[dict[str, Any]]:
    path = layout_dir_for(layout_root, scene_name) / "layout.json"
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"InternScenes layout must be a list: {path}")
    return data


def object_transform_from_instance(composer, canonical_mesh, instance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    box_data = instance["bbox"]
    mesh_size = canonical_mesh.bounding_box.extents
    scale_matrix = composer.get_scale_transform_from_rules(mesh_size, instance, bbox_data_key="bbox")
    transform = np.eye(4, dtype=np.float64)
    transform = scale_matrix @ transform
    angles = np.asarray(box_data[6:9], dtype=np.float64)
    rotation = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2], axes="rzxy")
    transform = rotation @ transform
    transform[:3, 3] = np.asarray(box_data[0:3], dtype=np.float64)
    return scale_matrix, transform


def normalized_scene_object(composer, instance: dict[str, Any]) -> tuple[trimesh.Scene | trimesh.Trimesh, np.ndarray]:
    model_uid = str(instance.get("model_uid") or "")
    if not model_uid:
        raise ValueError(f"Empty model_uid for InternScenes instance {instance.get('id')}")
    mesh_or_scene = composer.asset_mesh_loader.load_canonical_mesh(model_uid, use_texture=True)
    scale_matrix, scene_transform = object_transform_from_instance(composer, mesh_or_scene, instance)

    normalized = mesh_or_scene.copy()
    normalized.apply_transform(scale_matrix)
    z90 = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    normalized.apply_transform(z90)
    normalizer, normalized_to_scaled = normalize_transform_for_geometry(normalized)
    normalized.apply_transform(normalizer)
    normalized_to_scene = scene_transform @ np.linalg.inv(scale_matrix) @ np.linalg.inv(z90) @ normalized_to_scaled
    return normalized, normalized_to_scene


def normalized_source_object(composer, model_uid: str) -> tuple[trimesh.Scene | trimesh.Trimesh, np.ndarray]:
    mesh_or_scene = composer.asset_mesh_loader.load_canonical_mesh(model_uid, use_texture=True)
    normalizer, canonical_to_source = normalize_transform_for_geometry(mesh_or_scene)
    normalized = mesh_or_scene.copy()
    normalized.apply_transform(normalizer)
    return normalized, canonical_to_source


def normalized_background(scene_name: str, layout_root: Path) -> tuple[trimesh.Scene, np.ndarray]:
    structure_dir = layout_dir_for(layout_root, scene_name) / "StructureMesh"
    scene = trimesh.Scene()
    rot_x = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    for name in ("floor", "wall", "ceiling"):
        path = structure_dir / f"{name}.glb"
        if not path.exists():
            continue
        geom = trimesh.load(path, force="scene", process=False)
        geom.apply_transform(rot_x)
        scene.add_geometry(geom, geom_name=name)
    if not scene.geometry:
        raise FileNotFoundError(f"No StructureMesh GLBs found under {structure_dir}")
    normalizer, normalized_to_scene = normalize_transform_for_geometry(scene)
    scene.apply_transform(normalizer)
    return scene, normalized_to_scene


def transform_entries_for_scene(
    scene_name: str,
    layout: list[dict[str, Any]],
    bg_transform: np.ndarray | None,
    object_transforms: dict[int, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    safe_scene = scene_name_to_safe(scene_name)
    entries: list[dict[str, object]] = []
    if bg_transform is not None:
        entries.append(
            {
                "name": f"layout_{safe_scene}_bg",
                "latent": f"layout_{safe_scene}_bg",
                "asset_id": None,
                "matrix": matrix_payload(bg_transform),
            }
        )
    if object_transforms:
        for instance in sorted(layout, key=lambda x: int(x.get("id", 0))):
            instance_id = int(instance.get("id", 0))
            if instance_id not in object_transforms:
                continue
            model_uid = str(instance.get("model_uid") or "")
            entries.append(
                {
                    "name": f"object_{instance_id:04d}",
                    "latent": model_asset_id(model_uid),
                    "asset_id": instance_object_asset_id(scene_name, instance_id),
                    "source_id": model_uid,
                    "source_name": model_uid,
                    "category": instance.get("category"),
                    "instance_id": instance_id,
                    "matrix": matrix_payload(object_transforms[instance_id]),
                }
            )
    return entries


def write_transform_entries(scene_name: str, entries: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scene_name_to_safe(scene_name)}.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    return path


def collect_unique_model_records(
    split_json: Path,
    internscenes_root: Path,
    layout_root: Path,
    layout_tar: Path,
    extract_layout: bool,
    limit_scenes: int | None = None,
    limit_objects: int | None = None,
    skip_bad_scenes: bool = False,
    skip_missing_assets: bool = True,
    include_prefixes: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    scene_names = collect_split_scene_names(split_json, limit_scenes=limit_scenes)
    prefix_tuple = tuple(include_prefixes or [])
    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for scene_index, scene_name in enumerate(scene_names):
        try:
            ensure_layout_extracted(scene_name, layout_root, layout_tar, extract_layout)
            layout = load_layout(scene_name, layout_root)
        except Exception:
            if skip_bad_scenes:
                continue
            raise
        for instance in layout:
            model_uid = str(instance.get("model_uid") or "")
            if not model_uid or model_uid in records:
                continue
            if prefix_tuple and not model_uid.startswith(prefix_tuple):
                continue
            mesh_path = None
            try:
                mesh_path = mesh_path_for_model_uid(internscenes_root / "downloaded" / "asset_library", model_uid)
            except ValueError:
                if skip_missing_assets:
                    continue
                raise
            if skip_missing_assets and not mesh_path.exists():
                continue
            records[model_uid] = {
                "model_uid": model_uid,
                "asset_id": model_asset_id(model_uid),
                "mesh_path": str(mesh_path),
                "scene_name": scene_name,
                "scene_index": scene_index,
                "instance_id": instance.get("id"),
                "category": instance.get("category"),
            }
            if limit_objects is not None and len(records) >= limit_objects:
                return list(records.values())
    return list(records.values())
