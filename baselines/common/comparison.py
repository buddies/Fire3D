from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


DATASET_DIR_NAMES = {
    "ithor": "ithor",
    "imaginarium": "Imaginarium",
}


def load_assignment_scenes(assignments_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(assignments_path.read_text())
    scenes_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for worker in payload.get("workers", []):
        for raw_scene in worker.get("scenes", []):
            scene = dict(raw_scene)
            dataset = str(scene["dataset"]).lower()
            if dataset not in DATASET_DIR_NAMES:
                raise ValueError(f"Unsupported dataset in assignments: {dataset}")
            scene["dataset"] = dataset
            key = (dataset, str(scene["scene_id"]))
            if key in scenes_by_key and scenes_by_key[key] != scene:
                raise ValueError(f"Conflicting assignment records for {key}")
            scenes_by_key[key] = scene
    if not scenes_by_key:
        raise ValueError(f"No scenes found in {assignments_path}")
    return sorted(scenes_by_key.values(), key=lambda item: (item["dataset"], item["scene_id"]))


def partition_scenes(
    scenes: list[dict[str, Any]], worker_index: int, num_workers: int
) -> list[dict[str, Any]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index must be in [0, {num_workers}), got {worker_index}")

    bins: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    ordered = sorted(
        scenes,
        key=lambda item: (
            -int(item.get("estimated_objects", 1)),
            item["dataset"],
            item["scene_id"],
        ),
    )
    for scene in ordered:
        destination = min(range(num_workers), key=lambda index: (loads[index], index))
        bins[destination].append(scene)
        loads[destination] += int(scene.get("estimated_objects", 1))
    return sorted(bins[worker_index], key=lambda item: (item["dataset"], item["scene_id"]))


def exact_render_dir(source_run: Path, dataset: str, scene_id: str) -> Path:
    return source_run / "dataset" / DATASET_DIR_NAMES[dataset] / "renders" / scene_id


def dataset_root(fire3d_test_root: Path, dataset: str) -> Path:
    return fire3d_test_root / DATASET_DIR_NAMES[dataset]


def load_manifest_scene(repo_root: Path, dataset: str, scene_id: str) -> dict[str, Any]:
    manifest_path = (
        repo_root / "benchmarks" / "scene_reconstruction" / "manifests" / f"{dataset}_v1.json"
    )
    manifest = json.loads(manifest_path.read_text())
    matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {dataset} manifest entry for {scene_id}, found {len(matches)}")
    return matches[0]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
        os.chmod(path, 0o644)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def intrinsics_for_vertical_fov_viewport(
    camera: dict[str, Any], width: int, height: int
) -> list[list[float]]:
    """Resize a calibrated viewport while preserving its vertical field of view."""
    source_width = int(camera["width"])
    source_height = int(camera["height"])
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("Source and target viewport dimensions must be positive")
    source = camera["K"]
    if len(source) != 3 or any(len(row) != 3 for row in source):
        raise ValueError("Camera K must be a 3x3 matrix")
    scale = float(height) / float(source_height)
    return [
        [
            float(source[0][0]) * scale,
            float(source[0][1]) * scale,
            float(width) / 2.0
            + (float(source[0][2]) - source_width / 2.0) * scale,
        ],
        [
            float(source[1][0]) * scale,
            float(source[1][1]) * scale,
            float(height) / 2.0
            + (float(source[1][2]) - source_height / 2.0) * scale,
        ],
        [float(source[2][0]), float(source[2][1]), float(source[2][2])],
    ]


def replace_empty_background(
    beauty_path: Path, mask_path: Path, background_rgb: list[int]
) -> None:
    """Composite an opaque beauty render over a solid color using its AA mask."""
    if len(background_rgb) != 3 or any(not 0 <= value <= 255 for value in background_rgb):
        raise ValueError("background_rgb must contain three values in [0, 255]")
    with Image.open(beauty_path) as beauty_image:
        beauty = beauty_image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if beauty.size != mask.size:
        raise ValueError(
            f"Beauty/mask size mismatch: {beauty_path}={beauty.size}, "
            f"{mask_path}={mask.size}"
        )
    background = Image.new("RGB", beauty.size, tuple(background_rgb))
    Image.composite(beauty, background, mask).save(beauty_path)


def background_exclusion_filters_disagree(
    instance_mesh_names: set[str], named_mesh_names: set[str]
) -> bool:
    """Report whether two background exclusion filters selected different meshes.

    The composed predicted scene names one node ``background_position_0000`` by
    transform index rather than by which prediction is actually the background,
    so that name can point at an ordinary foreground object. When both an
    instance filter and the named-background filter match meshes, they must
    agree; a disjoint match means one of them is deleting foreground geometry.
    """
    if not instance_mesh_names or not named_mesh_names:
        return False
    return instance_mesh_names.isdisjoint(named_mesh_names)
