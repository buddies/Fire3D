#!/usr/bin/env python3
"""Audit FF-HoloScene inference points against extracted benchmark GT meshes.

The audit deliberately reads the exact cached conditioning points used by the
four HoloScene release inference runs.  It maps every object's ``points_model``
back to scene/world coordinates with the saved ``T_world_model`` and computes
unsigned point-to-triangle distance to the independently extracted GT scene
mesh.  Gibson additionally has per-instance GT meshes; those are concatenated
for the scene metric and independently nearest-matched to foreground point
sets for a diagnostic object-level audit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData


SCHEMA = "ff_holoscene_extracted_gt_alignment_v1"


@dataclass(frozen=True)
class SceneSpec:
    key: str
    gt_kind: str
    gt_path: str
    foreground_cache: str


SCENES = (
    SceneSpec(
        "replica_room_0",
        "single_mesh",
        "replica/data/room_0/mesh.ply",
        "cache",
    ),
    SceneSpec(
        "gibson_beechwood_0",
        "gibson_instances",
        "gibson/data/Beechwood_0_int/mesh/instances_vc",
        "cache",
    ),
    SceneSpec(
        "scannetpp_67d",
        "single_mesh",
        "scannetpp/67d702f2e8/mesh.ply",
        "cache_reference_filtered_mid_v3",
    ),
    SceneSpec(
        "uiuc_siebel_game_room",
        "single_mesh",
        "siebelgame2/mesh.ply",
        "cache_reference_filtered_mid_v3",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scene",
        action="append",
        choices=[scene.key for scene in SCENES],
        help="Scene(s) to audit; default is all four.",
    )
    parser.add_argument("--background-cache", default="cache_background64")
    parser.add_argument("--max-export-points", type=int, default=1_000_000)
    parser.add_argument("--max-plot-points", type=int, default=30_000)
    parser.add_argument("--max-plot-faces", type=int, default=40_000)
    parser.add_argument("--gibson-match-points", type=int, default=8_192)
    parser.add_argument("--gibson-candidates", type=int, default=10)
    parser.add_argument("--gibson-visualize-objects", type=int, default=16)
    return parser.parse_args()


def deterministic_indices(count: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def stable_color(identifier: int) -> np.ndarray:
    value = max(0, int(identifier)) + 1
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return np.asarray(
        [
            55 + int(value & 0xBF),
            55 + int((value >> 8) & 0xBF),
            55 + int((value >> 16) & 0xBF),
        ],
        dtype=np.uint8,
    )


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(
        str(path),
        enable_post_processing=False,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if (
        path.suffix.lower() == ".ply"
        and len(vertices)
        and len(faces) < max(1_000, len(vertices) // 20)
    ):
        # Open3D aborts on Replica's otherwise-valid all-quad GT PLY.  Read
        # the complete face list with plyfile and triangulate every polygon
        # with a deterministic fan instead of silently accepting a partial
        # mesh.
        ply = PlyData.read(str(path))
        vertex_data = ply["vertex"].data
        vertices = np.column_stack(
            [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
        ).astype(np.float64)
        polygons = ply["face"].data["vertex_indices"]
        triangle_parts: list[np.ndarray] = []
        for polygon in polygons:
            polygon = np.asarray(polygon, dtype=np.int64)
            if len(polygon) < 3:
                continue
            triangle_parts.append(
                np.column_stack(
                    [
                        np.repeat(polygon[0], len(polygon) - 2),
                        polygon[1:-1],
                        polygon[2:],
                    ]
                )
            )
        faces = np.concatenate(triangle_parts, axis=0).astype(np.int32)
    if not len(vertices) or not len(faces):
        raise ValueError(f"Empty triangle mesh: {path}")
    return vertices, faces


def replica_raw_to_inference_transform(
    scene_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive raw Replica -> HoloScene world translation from paired cameras."""
    raw_rows = np.loadtxt(scene_dir / "traj_ct.txt", skiprows=2)
    raw_centers = np.asarray(raw_rows[:, :3], dtype=np.float64)
    ns_payload = json.loads(
        (scene_dir / "ns" / "transforms.json").read_text(encoding="utf-8")
    )
    inference_centers = np.stack(
        [
            np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
            for frame in ns_payload["frames"]
        ]
    )
    if len(raw_centers) != len(inference_centers):
        raise ValueError(
            "Replica paired camera count mismatch: "
            f"{len(raw_centers)} raw vs {len(inference_centers)} HoloScene"
        )
    per_frame_translation = inference_centers - raw_centers
    translation = np.median(per_frame_translation, axis=0)
    residual = per_frame_translation - translation[None]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    residual_norm = np.linalg.norm(residual, axis=1)
    return transform, {
        "source": (
            "Median translation between the 811 paired raw traj_ct.txt camera "
            "centers and ns/transforms.json HoloScene camera centers."
        ),
        "paired_frames": int(len(raw_centers)),
        "rotation": np.eye(3, dtype=np.float64).tolist(),
        "translation_m": translation.tolist(),
        "per_axis_residual_std_m": residual.std(axis=0).tolist(),
        "residual_norm_m": distance_stats(residual_norm),
    }


def concatenate_meshes(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for part_vertices, part_faces, color in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        colors.append(np.repeat(color[None], len(part_vertices), axis=0))
        offset += len(part_vertices)
    return (
        np.concatenate(vertices, axis=0),
        np.concatenate(faces, axis=0),
        np.concatenate(colors, axis=0),
    )


def write_triangle_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(faces, dtype=np.int32)
    )
    if colors is not None:
        color_values = np.asarray(colors, dtype=np.float64)
        if color_values.max(initial=0.0) > 1.0:
            color_values = color_values / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(color_values, 0.0, 1.0)
        )
    if not o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=False,
        write_vertex_colors=colors is not None,
        write_triangle_uvs=False,
    ):
        raise OSError(f"Failed to write triangle mesh: {path}")


def write_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    color_values = np.asarray(colors, dtype=np.float64)
    if color_values.max(initial=0.0) > 1.0:
        color_values = color_values / 255.0
    cloud.colors = o3d.utility.Vector3dVector(
        np.clip(color_values, 0.0, 1.0)
    )
    if not o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    ):
        raise OSError(f"Failed to write point cloud: {path}")


def make_distance_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> o3d.t.geometry.RaycastingScene:
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def point_to_mesh_distances(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    chunks: list[np.ndarray] = []
    for start in range(0, len(points), 250_000):
        chunks.append(
            scene.compute_distance(
                o3d.core.Tensor(points[start : start + 250_000])
            ).numpy()
        )
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def distance_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [0.5, 0.75, 0.9, 0.95, 0.99])
    result: dict[str, float | int] = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(values.max()),
    }
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.05):
        result[f"fraction_le_{threshold:g}m"] = float(
            np.mean(values <= threshold)
        )
    return result


def bounds_stats(points: np.ndarray) -> dict[str, list[float] | float]:
    points = np.asarray(points, dtype=np.float64)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return {
        "min": lower.tolist(),
        "max": upper.tolist(),
        "extent": (upper - lower).tolist(),
        "diagonal": float(np.linalg.norm(upper - lower)),
    }


def distance_colors(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    colors = np.empty((len(distances), 3), dtype=np.uint8)
    colors[distances <= 0.005] = (20, 170, 90)
    colors[(distances > 0.005) & (distances <= 0.01)] = (35, 190, 215)
    colors[(distances > 0.01) & (distances <= 0.02)] = (245, 195, 40)
    colors[(distances > 0.02) & (distances <= 0.05)] = (245, 120, 35)
    colors[distances > 0.05] = (220, 35, 45)
    return colors


def surface_id(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except ValueError:
        return abs(hash(sample_id)) % 1_000_000


def load_cached_object(object_dir: Path) -> dict[str, Any]:
    metadata = json.loads(
        (object_dir / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(object_dir / "observations.npz") as observations:
        points_model = np.asarray(
            observations["points_model"],
            dtype=np.float64,
        )
    points_world = transform_points(
        points_model,
        np.asarray(metadata["T_world_model"], dtype=np.float64),
    )
    return {
        "sample_id": object_dir.name,
        "surface_id": int(metadata.get("surface_id", surface_id(object_dir.name))),
        "points_world": points_world,
        "metadata_path": str((object_dir / "metadata.json").resolve()),
        "observations_path": str((object_dir / "observations.npz").resolve()),
    }


def load_scene_points(
    spec: SceneSpec,
    inference_root: Path,
    background_cache: str,
) -> list[dict[str, Any]]:
    foreground_dir = (
        inference_root / spec.foreground_cache / "objects" / spec.key
    )
    if not foreground_dir.is_dir():
        raise FileNotFoundError(f"Missing foreground cache: {foreground_dir}")
    objects: list[dict[str, Any]] = []
    for object_dir in sorted(foreground_dir.iterdir()):
        if not object_dir.is_dir() or object_dir.name == "surface_0000":
            continue
        objects.append(load_cached_object(object_dir))

    background_dir = (
        inference_root
        / background_cache
        / "objects"
        / spec.key
        / "surface_0000"
    )
    if background_dir.is_dir():
        objects.insert(0, load_cached_object(background_dir))
    else:
        fallback = foreground_dir / "surface_0000"
        if fallback.is_dir():
            objects.insert(0, load_cached_object(fallback))
    if not objects:
        raise ValueError(f"No cached points found for {spec.key}")
    return objects


def load_gt(
    spec: SceneSpec,
    archive_root: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    source = archive_root / spec.gt_path
    if spec.gt_kind == "single_mesh":
        vertices, faces = load_mesh_arrays(source)
        transform = np.eye(4, dtype=np.float64)
        transform_info: dict[str, Any] = {
            "source": "Identity: extracted GT is already in the inference world frame.",
            "matrix": transform.tolist(),
        }
        if spec.key == "replica_room_0":
            transform, derived = replica_raw_to_inference_transform(source.parent)
            vertices = transform_points(vertices, transform)
            transform_info = {
                **derived,
                "matrix": transform.tolist(),
            }
        colors = np.repeat(
            np.asarray([[178, 183, 194]], dtype=np.uint8),
            len(vertices),
            axis=0,
        )
        return vertices, faces, colors, [], transform_info

    paths = sorted(source.glob("*.ply"))
    if not paths:
        raise FileNotFoundError(f"No Gibson instance meshes under {source}")
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    instances: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        vertices, faces = load_mesh_arrays(path)
        name = path.stem
        identifier = int(name) if name.isdigit() else 1_000_000 + index
        color = stable_color(identifier)
        parts.append((vertices, faces, color))
        instances.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "vertices": vertices,
                "faces": faces,
                "bounds_min": vertices.min(axis=0),
                "bounds_max": vertices.max(axis=0),
                "center": (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5,
                "diagonal": float(
                    np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
                ),
            }
        )
    vertices, faces, colors = concatenate_meshes(parts)
    return vertices, faces, colors, instances, {
        "source": (
            "Identity: extracted Gibson instance meshes are already in the "
            "inference world frame."
        ),
        "matrix": np.eye(4, dtype=np.float64).tolist(),
    }


def normalize_for_plot(
    vertices: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.minimum(vertices.min(axis=0), points.min(axis=0))
    upper = np.maximum(vertices.max(axis=0), points.max(axis=0))
    center = (lower + upper) * 0.5
    scale = max(float(np.max(upper - lower)), 1e-8)
    return (vertices - center[None]) / scale, (points - center[None]) / scale


def configure_axis(axis: Any, elevation: float, azimuth: float) -> None:
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlim(-0.58, 0.58)
    axis.set_ylim(-0.58, 0.58)
    axis.set_zlim(-0.58, 0.58)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_axis_off()


def simplify_for_plot(
    vertices: np.ndarray,
    faces: np.ndarray,
    maximum_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= maximum_faces:
        return vertices, faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(faces, dtype=np.int32)
    )
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(maximum_faces),
        boundary_weight=1.0,
    )
    simple_vertices = np.asarray(simplified.vertices, dtype=np.float64)
    simple_faces = np.asarray(simplified.triangles, dtype=np.int32)
    if len(simple_vertices) and len(simple_faces):
        return simple_vertices, simple_faces
    indices = deterministic_indices(len(faces), maximum_faces)
    return vertices, faces[indices]


def render_scene_overview(
    path: Path,
    scene_key: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    distances: np.ndarray,
    *,
    max_faces: int,
    max_points: int,
    stats: dict[str, Any],
) -> None:
    point_indices = deterministic_indices(len(points), max_points)
    mesh_vertices, mesh_faces = simplify_for_plot(
        vertices,
        faces,
        max_faces,
    )
    plot_vertices, plot_points = normalize_for_plot(
        mesh_vertices,
        points[point_indices],
    )
    point_colors = distance_colors(distances[point_indices]).astype(np.float64) / 255.0
    views = ((24, 35), (24, 125), (62, 40), (12, 220))
    figure = plt.figure(figsize=(15.0, 13.0), facecolor="#f3f5f8")
    figure.suptitle(
        (
            f"{scene_key}: exact conditioning points vs extracted GT mesh\n"
            f"P→mesh mean {stats['mean']:.5f} m | median {stats['median']:.5f} m | "
            f"p95 {stats['p95']:.5f} m | ≤1 cm {100.0 * stats['fraction_le_0.01m']:.1f}%"
        ),
        fontsize=15,
        fontweight="bold",
    )
    for index, (elevation, azimuth) in enumerate(views):
        axis = figure.add_subplot(2, 2, index + 1, projection="3d")
        collection = Poly3DCollection(
            plot_vertices[mesh_faces],
            facecolors=(0.40, 0.43, 0.49, 0.28),
            edgecolors=(0.18, 0.20, 0.23, 0.025),
            linewidths=0.04,
        )
        axis.add_collection3d(collection)
        axis.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            plot_points[:, 2],
            s=1.4,
            c=point_colors,
            alpha=0.9,
            depthshade=False,
        )
        configure_axis(axis, elevation, azimuth)
        axis.set_title(f"elev {elevation}°, az {azimuth}°", fontsize=10)
    figure.text(
        0.5,
        0.018,
        "Point color: green ≤5 mm, cyan ≤1 cm, yellow ≤2 cm, orange ≤5 cm, red >5 cm",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def aabb_point_distance(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    outside = np.maximum(np.maximum(lower[None] - points, points - upper[None]), 0.0)
    return float(np.median(np.linalg.norm(outside, axis=1)))


def match_gibson_instances(
    objects: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    *,
    match_points: int,
    candidate_count: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    numeric_instances = [
        instance for instance in instances if instance["name"].isdigit()
    ]
    scenes: dict[str, o3d.t.geometry.RaycastingScene] = {}
    for obj in objects:
        if int(obj["surface_id"]) == 0:
            continue
        points = np.asarray(obj["points_world"], dtype=np.float64)
        indices = deterministic_indices(len(points), match_points)
        query = points[indices]
        point_center = np.median(query, axis=0)
        coarse: list[tuple[float, dict[str, Any]]] = []
        for instance in numeric_instances:
            aabb_distance = aabb_point_distance(
                query,
                instance["bounds_min"],
                instance["bounds_max"],
            )
            center_distance = float(
                np.linalg.norm(point_center - instance["center"])
            )
            coarse.append(
                (
                    aabb_distance + 0.02 * center_distance,
                    instance,
                )
            )
        coarse.sort(key=lambda item: item[0])
        candidates: list[dict[str, Any]] = []
        for coarse_score, instance in coarse[:candidate_count]:
            name = str(instance["name"])
            if name not in scenes:
                scenes[name] = make_distance_scene(
                    instance["vertices"],
                    instance["faces"],
                )
            distances = point_to_mesh_distances(scenes[name], query)
            stats = distance_stats(distances)
            candidates.append(
                {
                    "instance": instance,
                    "coarse_score": float(coarse_score),
                    "distances": distances,
                    "stats": stats,
                    "score": float(stats["median"]) + 0.25 * float(stats["mean"]),
                }
            )
        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        matches.append(
            {
                "sample_id": obj["sample_id"],
                "surface_id": int(obj["surface_id"]),
                "point_count": int(len(points)),
                "query_point_count": int(len(query)),
                "matched_instance": str(best["instance"]["name"]),
                "matched_instance_path": str(best["instance"]["path"]),
                "distance": best["stats"],
                "candidate_diagnostics": [
                    {
                        "instance": str(candidate["instance"]["name"]),
                        "coarse_score": float(candidate["coarse_score"]),
                        "score": float(candidate["score"]),
                        "median": float(candidate["stats"]["median"]),
                        "mean": float(candidate["stats"]["mean"]),
                    }
                    for candidate in candidates[:5]
                ],
                "_points": query,
                "_instance": best["instance"],
                "_distances": best["distances"],
            }
        )
    return matches


def render_gibson_object_matches(
    path: Path,
    matches: list[dict[str, Any]],
    *,
    maximum: int,
) -> None:
    selected = sorted(
        matches,
        key=lambda item: (
            float(item["distance"]["median"]),
            -int(item["point_count"]),
        ),
    )[:maximum]
    columns = 4
    rows = max(1, math.ceil(len(selected) / columns))
    figure = plt.figure(
        figsize=(4.2 * columns, 4.1 * rows),
        facecolor="#f3f5f8",
    )
    figure.suptitle(
        "Gibson Beechwood 0: cached object points vs nearest extracted GT instance",
        fontsize=15,
        fontweight="bold",
    )
    for index, match in enumerate(selected):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        instance = match["_instance"]
        points = match["_points"]
        distances = match["_distances"]
        vertices, plot_points = normalize_for_plot(
            instance["vertices"],
            points,
        )
        face_indices = deterministic_indices(len(instance["faces"]), 8_000)
        point_indices = deterministic_indices(len(points), 4_000)
        collection = Poly3DCollection(
            vertices[instance["faces"][face_indices]],
            facecolors=(0.67, 0.69, 0.73, 0.16),
            edgecolors=(0.28, 0.30, 0.34, 0.06),
            linewidths=0.08,
        )
        axis.add_collection3d(collection)
        axis.scatter(
            plot_points[point_indices, 0],
            plot_points[point_indices, 1],
            plot_points[point_indices, 2],
            s=2.0,
            c=distance_colors(distances[point_indices]).astype(np.float64) / 255.0,
            alpha=0.9,
            depthshade=False,
        )
        configure_axis(axis, 24, 42 + 23 * (index % columns))
        axis.set_title(
            (
                f"{match['sample_id']} → GT {match['matched_instance']}\n"
                f"median {match['distance']['median']:.4f} m, "
                f"p95 {match['distance']['p95']:.4f} m"
            ),
            fontsize=9,
        )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def clean_match_payload(match: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in match.items()
        if not key.startswith("_")
    }


def audit_scene(
    spec: SceneSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = args.output_root / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, vertex_colors, instances, gt_transform = load_gt(
        spec,
        args.archive_root,
    )
    objects = load_scene_points(
        spec,
        args.inference_root,
        args.background_cache,
    )
    distance_scene = make_distance_scene(vertices, faces)
    all_points: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    all_object_colors: list[np.ndarray] = []
    per_object: list[dict[str, Any]] = []
    for obj in objects:
        points = obj["points_world"]
        distances = point_to_mesh_distances(distance_scene, points)
        all_points.append(points)
        all_distances.append(distances)
        all_object_colors.append(
            np.repeat(
                stable_color(int(obj["surface_id"]))[None],
                len(points),
                axis=0,
            )
        )
        per_object.append(
            {
                "sample_id": obj["sample_id"],
                "surface_id": int(obj["surface_id"]),
                "point_count": int(len(points)),
                "distance": distance_stats(distances),
                "point_bounds": bounds_stats(points),
                "metadata_path": obj["metadata_path"],
                "observations_path": obj["observations_path"],
            }
        )

    points = np.concatenate(all_points, axis=0)
    distances = np.concatenate(all_distances, axis=0)
    object_colors = np.concatenate(all_object_colors, axis=0)
    stats = distance_stats(distances)
    export_indices = deterministic_indices(
        len(points),
        args.max_export_points,
    )
    write_triangle_mesh(
        output_dir / "extracted_gt_scene_mesh.ply",
        vertices,
        faces,
        vertex_colors,
    )
    write_point_cloud(
        output_dir / "conditioning_points_by_distance.ply",
        points[export_indices],
        distance_colors(distances[export_indices]),
    )
    write_point_cloud(
        output_dir / "conditioning_points_by_object.ply",
        points[export_indices],
        object_colors[export_indices],
    )
    render_scene_overview(
        output_dir / "scene_points_vs_extracted_gt.png",
        spec.key,
        vertices,
        faces,
        points,
        distances,
        max_faces=args.max_plot_faces,
        max_points=args.max_plot_points,
        stats=stats,
    )

    gibson_matches: list[dict[str, Any]] = []
    if instances:
        gibson_matches = match_gibson_instances(
            objects,
            instances,
            match_points=args.gibson_match_points,
            candidate_count=args.gibson_candidates,
        )
        render_gibson_object_matches(
            output_dir / "gibson_object_points_vs_gt_instances.png",
            gibson_matches,
            maximum=args.gibson_visualize_objects,
        )

    report = {
        "schema": SCHEMA,
        "scene": spec.key,
        "gt_kind": spec.gt_kind,
        "gt_source": str((args.archive_root / spec.gt_path).resolve()),
        "foreground_cache": str(
            (
                args.inference_root
                / spec.foreground_cache
                / "objects"
                / spec.key
            ).resolve()
        ),
        "background_cache": str(
            (
                args.inference_root
                / args.background_cache
                / "objects"
                / spec.key
            ).resolve()
        ),
        "coordinate_frame": (
            "Dataset scene/world coordinates. Cache points_model are transformed "
            "with each metadata.json T_world_model. The only non-identity GT "
            "conversion is Replica's translation derived from paired cameras; "
            "no mesh/point fitting or ICP is applied."
        ),
        "gt_coordinate_conversion": gt_transform,
        "distance_definition": (
            "Unsigned Euclidean distance from each exact cached conditioning point "
            "to the closest triangle on the independently extracted GT scene mesh."
        ),
        "gt_mesh": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds": bounds_stats(vertices),
            "instance_meshes": int(len(instances)),
        },
        "conditioning_points": {
            "objects": int(len(objects)),
            "points": int(len(points)),
            "bounds": bounds_stats(points),
            "distance_to_gt_scene_mesh": stats,
            "per_object": per_object,
        },
        "gibson_object_diagnostic": {
            "definition": (
                "Each foreground cached point set is independently matched to the "
                "nearest numeric Gibson GT instance among a bounding-box shortlist. "
                "This is an oracle alignment diagnostic, not semantic correspondence."
            ),
            "matches": [clean_match_payload(match) for match in gibson_matches],
        },
        "artifacts": {
            "gt_scene_mesh": "extracted_gt_scene_mesh.ply",
            "points_by_distance": "conditioning_points_by_distance.ply",
            "points_by_object": "conditioning_points_by_object.ply",
            "scene_overview": "scene_points_vs_extracted_gt.png",
            "gibson_object_overview": (
                "gibson_object_points_vs_gt_instances.png"
                if gibson_matches
                else None
            ),
        },
    }
    atomic_json(output_dir / "alignment_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    selected = set(args.scene or [scene.key for scene in SCENES])
    reports = [
        audit_scene(scene, args)
        for scene in SCENES
        if scene.key in selected
    ]
    summary = {
        "schema": SCHEMA,
        "scenes": [
            {
                "scene": report["scene"],
                "gt_mesh": report["gt_mesh"],
                "conditioning_objects": report["conditioning_points"]["objects"],
                "conditioning_points": report["conditioning_points"]["points"],
                "distance_to_gt_scene_mesh": report["conditioning_points"][
                    "distance_to_gt_scene_mesh"
                ],
            }
            for report in reports
        ],
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
