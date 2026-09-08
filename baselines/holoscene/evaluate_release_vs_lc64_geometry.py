#!/usr/bin/env python3
"""Compare released HoloScene and LC64 foreground geometry to extracted GT.

The scene-level metric is intentionally visibility-conditioned: prediction
surfaces are compared to the complete extracted GT scene, while the reverse
direction uses the foreground points that were actually available to both
reconstruction pipelines. Gibson additionally has exact per-instance GT
meshes, so it receives a standard symmetric object-level CD/F1/NC evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.holoscene.audit_extracted_gt_alignment import (  # noqa: E402
    SCENES,
    load_gt,
    load_scene_points,
)


METHODS = ("holoscene_release", "lc64_ss445k_shape255k_pbr140k")
SURFACE_RE = re.compile(r"surface_(\d+)\.obj$")
SCHEMA = "holoscene_release_vs_lc64_extracted_gt_geometry_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--background-cache", default="cache_background64")
    parser.add_argument("--scene-samples", type=int, default=200_000)
    parser.add_argument("--visible-gt-points", type=int, default=200_000)
    parser.add_argument("--object-samples", type=int, default=200_000)
    parser.add_argument("--f1-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--scene",
        action="append",
        choices=[scene.key for scene in SCENES],
        help="Restrict to one or more scenes; default is all four.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_seed(base: int, *parts: Any) -> int:
    text = ":".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def as_mesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    if isinstance(value, trimesh.Scene):
        meshes = [
            mesh for mesh in value.dump() if isinstance(mesh, trimesh.Trimesh)
        ]
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    raise TypeError(type(value))


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = as_mesh(trimesh.load(path, force="scene", process=False))
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def concatenate(meshes: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = [mesh for mesh in meshes if len(mesh.vertices) and len(mesh.faces)]
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()


def surface_id(path: Path) -> int:
    match = SURFACE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def release_world_contract(release_scene_dir: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(
        (release_scene_dir / "transforms.json").read_text(encoding="utf-8")
    )
    centers = np.asarray(
        [
            np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
            for frame in payload["frames"]
        ]
    )
    center = (centers.min(axis=0) + centers.max(axis=0)) * 0.5
    scale = float(np.max(centers.max(axis=0) - centers.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid released scene scale under {release_scene_dir}")
    return center, scale


def release_scene_data_dir(release_root: Path, scene_key: str) -> Path:
    relative = {
        "replica_room_0": "replica/room_0",
        "gibson_beechwood_0": "gibson/Beechwood_0_int",
        "scannetpp_67d": "scannetpp/67d702f2e8",
        "uiuc_siebel_game_room": "custom/siebelgame",
    }[scene_key]
    return release_root.parent / "data_dir" / relative


def load_release_surfaces(
    release_scene_dir: Path,
    scene_data_dir: Path,
) -> tuple[dict[int, trimesh.Trimesh], dict[str, Any]]:
    center, scale = release_world_contract(scene_data_dir)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = center
    surfaces: dict[int, trimesh.Trimesh] = {}
    for path in sorted(release_scene_dir.glob("surface_*.obj"), key=surface_id):
        identifier = surface_id(path)
        if identifier == 0:
            continue
        mesh = load_mesh(path)
        mesh.apply_transform(transform)
        surfaces[identifier] = mesh
    if not surfaces:
        raise ValueError(f"No released foreground surfaces: {release_scene_dir}")
    return surfaces, {
        "release_scene_dir": str(release_scene_dir.resolve()),
        "scene_data_dir": str(scene_data_dir.resolve()),
        "normalized_to_world_center": center,
        "normalized_to_world_scale": scale,
        "surface_ids": sorted(surfaces),
    }


def load_lc64_surfaces(
    bundle_dir: Path,
) -> tuple[dict[int, trimesh.Trimesh], dict[str, Any]]:
    manifest_path = bundle_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    surfaces: dict[int, trimesh.Trimesh] = {}
    records: dict[int, dict[str, Any]] = {}
    for record in payload["objects"]:
        identifier = int(record["surface_id"])
        if identifier == 0:
            continue
        pose_path = Path(record["pose_json"])
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        matrix = np.asarray(pose["T_world_from_ff_canonical"], dtype=np.float64)
        mesh_path = Path(record["canonical_glb"])
        mesh = load_mesh(mesh_path)
        mesh.apply_transform(matrix)
        surfaces[identifier] = mesh
        records[identifier] = {
            "surface_id": identifier,
            "canonical_glb": str(mesh_path.resolve()),
            "pose_json": str(pose_path.resolve()),
            "T_world_from_ff_canonical": matrix,
        }
    if not surfaces:
        raise ValueError(f"No LC64 foreground surfaces: {manifest_path}")
    return surfaces, {
        "bundle_manifest": str(manifest_path.resolve()),
        "surface_ids": sorted(surfaces),
        "records": records,
    }


def make_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def normalized_normals_or_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = np.isfinite(values).all(axis=1, keepdims=True) & (norms > 1e-12)
    result = np.full_like(values, np.nan)
    np.divide(values, norms, out=result, where=valid)
    return result


def finite_normal_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else math.nan


def closest(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    distances: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for start in range(0, len(points), 250_000):
        tensor = o3d.core.Tensor(points[start : start + 250_000])
        result = scene.compute_closest_points(tensor)
        closest_points = result["points"].numpy()
        distances.append(
            np.linalg.norm(
                points[start : start + 250_000].astype(np.float64)
                - closest_points.astype(np.float64),
                axis=1,
            )
        )
        normals.append(result["primitive_normals"].numpy().astype(np.float64))
    return np.concatenate(distances), normalized_normals_or_nan(
        np.concatenate(normals)
    )


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-16)
    if not valid.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    valid_faces = np.flatnonzero(valid)
    probabilities = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(seed)
    face_positions = rng.choice(len(valid_faces), size=int(count), p=probabilities)
    chosen = valid_faces[face_positions]
    u = rng.random(int(count))
    v = rng.random(int(count))
    square_root = np.sqrt(u)
    weights = np.column_stack(
        [1.0 - square_root, square_root * (1.0 - v), square_root * v]
    )
    points = np.einsum("ni,nij->nj", weights, triangles[chosen])
    normals = unit_rows(cross[chosen])
    return points, normals


def directional_stats(distances: np.ndarray) -> dict[str, float | int]:
    distances = np.asarray(distances, dtype=np.float64)
    quantiles = np.quantile(distances, [0.5, 0.9, 0.95, 0.99])
    return {
        "count": int(len(distances)),
        "mean": float(distances.mean()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(distances.max()),
    }


def metrics_from_directions(
    pred_to_gt: np.ndarray,
    gt_to_pred: np.ndarray,
    pred_to_gt_nc: np.ndarray,
    gt_to_pred_nc: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    precision = float(np.mean(pred_to_gt <= threshold))
    recall = float(np.mean(gt_to_pred <= threshold))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    cd = 0.5 * (float(pred_to_gt.mean()) + float(gt_to_pred.mean()))
    pred_nc = finite_normal_mean(pred_to_gt_nc)
    gt_nc = finite_normal_mean(gt_to_pred_nc)
    available_nc = [value for value in (pred_nc, gt_nc) if math.isfinite(value)]
    nc = float(np.mean(available_nc)) if available_nc else math.nan
    return {
        "CD": cd,
        "CD_x100": cd * 100.0,
        "F1": f1,
        "NC": nc,
        "precision": precision,
        "recall": recall,
        "threshold_m": threshold,
        "pred_to_gt": directional_stats(pred_to_gt),
        "gt_to_pred": directional_stats(gt_to_pred),
        "pred_to_gt_normal_consistency": pred_nc,
        "gt_to_pred_normal_consistency": gt_nc,
        "pred_to_gt_valid_normal_pairs": int(
            np.isfinite(np.asarray(pred_to_gt_nc)).sum()
        ),
        "gt_to_pred_valid_normal_pairs": int(
            np.isfinite(np.asarray(gt_to_pred_nc)).sum()
        ),
    }


def visible_scene_metrics(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    visible_gt_points: np.ndarray,
    *,
    count: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    pred_points, pred_normals = sample_surface(pred_mesh, count, seed)
    gt_scene = make_scene(gt_mesh)
    pred_scene = make_scene(pred_mesh)
    pred_to_gt, gt_normals_at_pred = closest(gt_scene, pred_points)
    visible_to_gt, visible_gt_normals = closest(gt_scene, visible_gt_points)
    gt_to_pred, pred_normals_at_visible = closest(pred_scene, visible_gt_points)
    metrics = metrics_from_directions(
        pred_to_gt,
        gt_to_pred,
        np.abs(np.sum(unit_rows(pred_normals) * gt_normals_at_pred, axis=1)),
        np.abs(np.sum(visible_gt_normals * pred_normals_at_visible, axis=1)),
        threshold,
    )
    metrics["visible_gt_alignment_to_scene"] = directional_stats(visible_to_gt)
    metrics["prediction_surface_area_m2"] = float(pred_mesh.area)
    metrics["prediction_vertices"] = int(len(pred_mesh.vertices))
    metrics["prediction_faces"] = int(len(pred_mesh.faces))
    return metrics


def symmetric_object_metrics(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    pred_points, pred_normals = sample_surface(pred_mesh, count, seed)
    gt_points, gt_normals = sample_surface(gt_mesh, count, seed ^ 0x9E3779B97F4A7C15)
    gt_scene = make_scene(gt_mesh)
    pred_scene = make_scene(pred_mesh)
    pred_to_gt, gt_closest_normals = closest(gt_scene, pred_points)
    gt_to_pred, pred_closest_normals = closest(pred_scene, gt_points)
    metrics = metrics_from_directions(
        pred_to_gt,
        gt_to_pred,
        np.abs(np.sum(unit_rows(pred_normals) * gt_closest_normals, axis=1)),
        np.abs(np.sum(unit_rows(gt_normals) * pred_closest_normals, axis=1)),
        threshold,
    )
    metrics["prediction_surface_area_m2"] = float(pred_mesh.area)
    metrics["gt_surface_area_m2"] = float(gt_mesh.area)
    return metrics


def finite_mean(rows: list[dict[str, Any]], key: str, weights: np.ndarray | None = None) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return math.nan
    if weights is None:
        return float(values[valid].mean())
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def metric_aggregate(rows: list[dict[str, Any]], weights: np.ndarray | None = None) -> dict[str, float]:
    keys = ("CD", "CD_x100", "F1", "NC", "precision", "recall")
    return {key: finite_mean(rows, key, weights) for key in keys}


def bundle_dir(inference_root: Path, scene_key: str) -> Path:
    root = (
        "inferred_glb_pose_bundles_reference_filtered_mid_v3"
        if scene_key in {"scannetpp_67d", "uiuc_siebel_game_room"}
        else "inferred_glb_pose_bundles"
    )
    return inference_root / root / scene_key


def deterministic_cap(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), size=maximum, replace=False))]


def write_scene_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scene",
        "method",
        "CD",
        "CD_x100",
        "F1",
        "NC",
        "precision",
        "recall",
        "prediction_surface_area_m2",
        "prediction_vertices",
        "prediction_faces",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_object_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "gt_instance",
        "surface_ids",
        "method",
        "CD",
        "CD_x100",
        "F1",
        "NC",
        "precision",
        "recall",
        "gt_surface_area_m2",
        "prediction_surface_area_m2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "surface_ids": " ".join(str(value) for value in row["surface_ids"]),
                }
            )


def main() -> None:
    args = parse_args()
    selected = set(args.scene or [scene.key for scene in SCENES])
    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, Any]] = []
    scene_records = []
    gibson_assets: dict[str, Any] | None = None

    for spec in SCENES:
        if spec.key not in selected:
            continue
        print(f"[scene] {spec.key}", flush=True)
        gt_vertices, gt_faces, _, _, gt_conversion = load_gt(spec, args.archive_root)
        gt_mesh = trimesh.Trimesh(
            vertices=gt_vertices, faces=gt_faces, process=False
        )
        cached = [
            record
            for record in load_scene_points(
                spec, args.inference_root, args.background_cache
            )
            if int(record["surface_id"]) > 0
        ]
        visible_points = deterministic_cap(
            np.concatenate([record["points_world"] for record in cached], axis=0),
            args.visible_gt_points,
            stable_seed(args.seed, spec.key, "visible"),
        )
        release_surfaces, release_info = load_release_surfaces(
            args.release_root / spec.key,
            release_scene_data_dir(args.release_root, spec.key),
        )
        lc64_surfaces, lc64_info = load_lc64_surfaces(
            bundle_dir(args.inference_root, spec.key)
        )
        method_surfaces = {
            "holoscene_release": release_surfaces,
            "lc64_ss445k_shape255k_pbr140k": lc64_surfaces,
        }
        method_info = {
            "holoscene_release": release_info,
            "lc64_ss445k_shape255k_pbr140k": lc64_info,
        }
        method_records = {}
        for method, surfaces in method_surfaces.items():
            print(
                f"  {method}: {len(surfaces)} foreground surfaces",
                flush=True,
            )
            pred_mesh = concatenate(surfaces.values())
            metrics = visible_scene_metrics(
                pred_mesh,
                gt_mesh,
                visible_points,
                count=args.scene_samples,
                seed=stable_seed(args.seed, spec.key, method),
                threshold=args.f1_threshold,
            )
            row = {"scene": spec.key, "method": method, **metrics}
            scene_rows.append(row)
            method_records[method] = {
                "assets": method_info[method],
                "metrics": metrics,
            }
        scene_records.append(
            {
                "scene": spec.key,
                "gt_source": str((args.archive_root / spec.gt_path).resolve()),
                "gt_coordinate_conversion": gt_conversion,
                "gt_vertices": int(len(gt_mesh.vertices)),
                "gt_faces": int(len(gt_mesh.faces)),
                "visible_foreground_points": int(len(visible_points)),
                "cached_foreground_objects": int(len(cached)),
                "methods": method_records,
            }
        )
        if spec.key == "gibson_beechwood_0":
            gibson_assets = {
                "gt_mesh": gt_mesh,
                "release_surfaces": release_surfaces,
                "release_info": release_info,
                "lc64_surfaces": lc64_surfaces,
                "lc64_info": lc64_info,
            }

    scene_summary = {
        "schema": SCHEMA,
        "protocol": {
            "name": "visibility_conditioned_foreground_to_extracted_scene",
            "scene_surface_samples": args.scene_samples,
            "visible_gt_points": args.visible_gt_points,
            "f1_threshold_m": args.f1_threshold,
            "seed": args.seed,
            "CD": (
                "0.5*(method foreground surface -> extracted GT scene + "
                "visible cached GT foreground points -> method foreground surface)"
            ),
            "background_policy": "surface_0 excluded from both methods",
        },
        "scenes": scene_records,
        "method_scene_macro": {
            method: metric_aggregate(
                [
                    {"method": row["method"], **row}
                    for row in scene_rows
                    if row["method"] == method
                ]
            )
            for method in METHODS
            if any(row["method"] == method for row in scene_rows)
        },
    }
    atomic_json(args.output_root / "scene_foreground_geometry.json", scene_summary)
    write_scene_csv(args.output_root / "scene_foreground_geometry.csv", scene_rows)

    if gibson_assets is None:
        return
    alignment_path = (
        args.inference_root
        / "extracted_gt_alignment_20260724"
        / "gibson_beechwood_0"
        / "alignment_report.json"
    )
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[int]] = defaultdict(list)
    for match in alignment["gibson_object_diagnostic"]["matches"]:
        grouped[str(match["matched_instance"])].append(int(match["surface_id"]))

    object_rows = []
    object_records = []
    for object_index, (instance, surface_ids) in enumerate(sorted(grouped.items())):
        surface_ids = sorted(surface_ids)
        gt_path = args.archive_root / (
            "gibson/data/Beechwood_0_int/mesh/instances_vc/"
            f"{instance}.ply"
        )
        gt_mesh = load_mesh(gt_path)
        methods = {}
        for method, surface_key in (
            ("holoscene_release", "release_surfaces"),
            ("lc64_ss445k_shape255k_pbr140k", "lc64_surfaces"),
        ):
            available = gibson_assets[surface_key]
            included = [identifier for identifier in surface_ids if identifier in available]
            missing = [identifier for identifier in surface_ids if identifier not in available]
            if not included:
                methods[method] = {
                    "status": "missing_all_surfaces",
                    "included_surface_ids": [],
                    "missing_surface_ids": missing,
                    "metrics": None,
                }
                continue
            mesh = concatenate(available[identifier] for identifier in included)
            metrics = symmetric_object_metrics(
                mesh,
                gt_mesh,
                count=args.object_samples,
                seed=stable_seed(args.seed, "gibson", instance, method),
                threshold=args.f1_threshold,
            )
            methods[method] = {
                "status": "complete" if not missing else "partial",
                "included_surface_ids": included,
                "missing_surface_ids": missing,
                "metrics": metrics,
            }
            object_rows.append(
                {
                    "gt_instance": instance,
                    "surface_ids": surface_ids,
                    "method": method,
                    **metrics,
                }
            )
        object_records.append(
            {
                "object_index": object_index,
                "gt_instance": instance,
                "surface_ids": surface_ids,
                "gt_geometry": str(gt_path.resolve()),
                "gt_textured": str(
                    (
                        args.archive_root
                        / "gibson/data/Beechwood_0_int/mesh/instances_whole"
                        / f"{instance}.obj"
                    ).resolve()
                ),
                "gt_bounds_world": [
                    np.asarray(gt_mesh.bounds[0]),
                    np.asarray(gt_mesh.bounds[1]),
                ],
                "gt_surface_area_m2": float(gt_mesh.area),
                "methods": methods,
            }
        )

    aggregates = {}
    for method in METHODS:
        rows = [row for row in object_rows if row["method"] == method]
        weights = np.asarray(
            [float(row["gt_surface_area_m2"]) for row in rows], dtype=np.float64
        )
        aggregates[method] = {
            "objects": len(rows),
            "equal_object_macro": metric_aggregate(rows),
            "gt_surface_area_weighted": metric_aggregate(rows, weights),
        }
    object_summary = {
        "schema": SCHEMA,
        "protocol": {
            "name": "gibson_extracted_instance_symmetric_surface",
            "surface_samples_per_direction": args.object_samples,
            "f1_threshold_m": args.f1_threshold,
            "seed": args.seed,
            "grouping": (
                "released surface IDs sharing one nearest exact extracted "
                "Gibson instance are unioned before evaluation"
            ),
        },
        "alignment_report": str(alignment_path.resolve()),
        "aggregates": aggregates,
        "objects": object_records,
    }
    atomic_json(args.output_root / "gibson_object_geometry.json", object_summary)
    write_object_csv(args.output_root / "gibson_object_geometry.csv", object_rows)
    atomic_json(
        args.output_root / "gibson_object_groups.json",
        {
            "schema": "holoscene_release_vs_lc64_gibson_object_groups_v1",
            "release_info": gibson_assets["release_info"],
            "lc64_info": gibson_assets["lc64_info"],
            "objects": object_records,
        },
    )
    print(
        json.dumps(
            {
                "scene_metrics": str(
                    args.output_root / "scene_foreground_geometry.json"
                ),
                "gibson_metrics": str(
                    args.output_root / "gibson_object_geometry.json"
                ),
                "gibson_objects": len(object_records),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
