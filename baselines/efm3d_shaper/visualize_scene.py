#!/usr/bin/env python3
"""Render a world-space EFM3D-to-ShapeR scene from saved review cameras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.efm3d.visualization import (  # noqa: E402
    add_label,
    assemble_sheet,
    effective_pinhole,
    load_cloud,
    load_predictions,
    overlay_efm_obbs,
    render_point_cloud,
)


PALETTE = np.asarray(
    [
        [0.00, 0.83, 1.00],
        [1.00, 0.36, 0.54],
        [0.55, 0.86, 0.29],
        [1.00, 0.71, 0.20],
        [0.72, 0.46, 1.00],
        [0.23, 0.88, 0.72],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--shaper-dir", type=Path, required=True)
    parser.add_argument("--view-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prob-threshold", type=float, default=0.2)
    parser.add_argument("--max-input-points", type=int, default=250000)
    parser.add_argument("--max-mesh-vertices", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--obb-line-width", type=int, default=3)
    return parser.parse_args()


def color_for_object(index: int) -> np.ndarray:
    return PALETTE[index % len(PALETTE)]


def select_vertices(vertices: np.ndarray, max_vertices: int, seed: int) -> np.ndarray:
    if max_vertices <= 0 or len(vertices) <= max_vertices:
        return vertices
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(vertices), size=max_vertices, replace=False))
    return vertices[indices]


def reconstruction_coverage(
    predictions: list[dict[str, object]], object_records: list[dict[str, object]]
) -> tuple[set[int], set[int], list[int]]:
    prediction_ids = {int(row["instance"]) for row in predictions}
    reconstruction_ids = {
        int(row["source_instance_id"])
        for row in object_records
        if row["source_instance_id"] is not None
    }
    unexpected_ids = sorted(reconstruction_ids - prediction_ids)
    if unexpected_ids:
        raise ValueError(
            "ShapeR reconstructions have no corresponding EFM3D detection IDs: "
            f"{unexpected_ids}"
        )
    return prediction_ids, reconstruction_ids, sorted(prediction_ids - reconstruction_ids)


def load_shaper_scene(
    shaper_dir: Path, max_mesh_vertices: int
) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh, list[dict[str, object]]]:
    manifest_path = shaper_dir / "object_obbs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    point_parts = []
    color_parts = []
    colored_meshes = []
    records = []
    for index, obj in enumerate(manifest["objects"]):
        name = str(obj["name"])
        mesh_path = shaper_dir / f"{name}.ply"
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices):
            raise ValueError(f"Expected a nonempty mesh at {mesh_path}")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        selected = select_vertices(vertices, max_mesh_vertices, seed=20260829 + index)
        source_instance_id = obj.get("source_instance_id")
        color_index = index if source_instance_id is None else int(source_instance_id)
        color = color_for_object(color_index)
        point_parts.append(selected)
        color_parts.append(np.broadcast_to(color, (len(selected), 3)).copy())

        colored = mesh.copy()
        rgba = np.concatenate([color, np.asarray([1.0], dtype=np.float32)])
        colored.visual.vertex_colors = np.broadcast_to(
            np.rint(rgba * 255.0).astype(np.uint8), (len(colored.vertices), 4)
        ).copy()
        colored_meshes.append(colored)

        obb = obj["obb_world"]
        transform = np.asarray(obb["T_world_object"], dtype=np.float64)
        local_vertices = trimesh.transform_points(vertices, np.linalg.inv(transform))
        local_half_extent = 0.5 * np.ptp(local_vertices, axis=0)
        obb_half_extent = 0.5 * np.asarray(obb["scale"], dtype=np.float64)
        center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        records.append(
            {
                "name": name,
                "mesh": str(mesh_path),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "source_instance_id": source_instance_id,
                "source_name": obj.get("source_object_name"),
                "source_prob": obj.get("source_prob"),
                "mesh_aabb_center_world": center.astype(float).tolist(),
                "obb_center_world": obb["translation"],
                "aabb_center_to_obb_center": float(
                    np.linalg.norm(center - np.asarray(obb["translation"]))
                ),
                "mesh_local_half_extent": local_half_extent.astype(float).tolist(),
                "obb_half_extent": obb_half_extent.astype(float).tolist(),
                "local_extent_over_obb": np.divide(
                    local_half_extent,
                    np.maximum(obb_half_extent, 1e-8),
                ).astype(float).tolist(),
            }
        )

    combined = trimesh.util.concatenate(colored_meshes)
    return (
        np.concatenate(point_parts, axis=0),
        np.concatenate(color_parts, axis=0),
        combined,
        records,
    )


def main() -> None:
    args = parse_args()
    if args.max_input_points <= 0 or args.max_mesh_vertices <= 0:
        raise ValueError("Point and mesh vertex caps must be positive")
    if args.point_size <= 0 or args.obb_line_width <= 0:
        raise ValueError("Point size and OBB line width must be positive")

    input_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.prediction_csv.resolve(), args.prob_threshold)
    instance_points, instance_colors = load_cloud(
        input_dir / "semi_points_instances.ply", args.max_input_points
    )
    context_points, context_colors = load_cloud(
        input_dir / "semi_points.ply", args.max_input_points
    )
    if context_points.shape != instance_points.shape or not np.allclose(
        context_points, instance_points, rtol=0.0, atol=1e-6
    ):
        raise ValueError("RGB and instance-colored input point coordinates differ")
    mesh_points, mesh_colors, combined_mesh, object_records = load_shaper_scene(
        args.shaper_dir.resolve(), args.max_mesh_vertices
    )
    prediction_points = np.concatenate([context_points, mesh_points], axis=0)
    prediction_colors = np.concatenate([context_colors * 0.18, mesh_colors], axis=0)
    prediction_ids, reconstruction_ids, skipped_detection_ids = reconstruction_coverage(
        predictions, object_records
    )

    combined_path = output_dir / "efm3d_shaper_world_mesh.ply"
    combined_mesh.export(combined_path)
    task = json.loads(args.view_task.resolve().read_text(encoding="utf-8"))
    camera = json.loads(Path(task["camera_path"]).read_text(encoding="utf-8"))
    width = int(task["render_width"])
    height = int(task["render_height"])
    pinhole = effective_pinhole(
        camera["K"],
        int(task["source_width"]),
        int(task["source_height"]),
        width,
        height,
    )

    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    render_records = []
    for rank, view in enumerate(task["views"]):
        candidate_id = str(view["candidate_id"])
        gt_image, gt_visible = render_point_cloud(
            instance_points,
            instance_colors,
            view["frame"],
            pinhole,
            width,
            height,
            args.point_size,
        )
        pred_image, pred_visible = render_point_cloud(
            prediction_points,
            prediction_colors,
            view["frame"],
            pinhole,
            width,
            height,
            args.point_size,
        )
        pred_image, visible_boxes, visible_edges = overlay_efm_obbs(
            pred_image,
            predictions,
            view["frame"],
            pinhole,
            args.obb_line_width,
        )
        gt_image = add_label(
            gt_image,
            f"GT instances | {candidate_id}",
            f"{task['scene_id']} | all points",
            prediction=False,
        )
        pred_image = add_label(
            pred_image,
            f"EFM3D -> ShapeR | {candidate_id}",
            f"{task['scene_id']} | context + {len(object_records)} meshes",
            prediction=True,
        )
        gt_path = views_dir / f"gt_{candidate_id}.png"
        pred_path = views_dir / f"efm3d_shaper_{candidate_id}.png"
        gt_image.save(gt_path)
        pred_image.save(pred_path)
        panels.append({"gt": gt_image, "pred": pred_image})
        render_records.append(
            {
                "candidate_id": candidate_id,
                "view_rank": rank,
                "gt_projected_points": gt_visible,
                "mesh_projected_vertices": pred_visible,
                "projected_boxes": visible_boxes,
                "projected_box_edges": visible_edges,
            }
        )

    sheet_path = output_dir / f"gt_vs_efm3d_shaper_{len(panels)}views.png"
    assemble_sheet(panels, width, height, sheet_path)
    summary = {
        "schema": "ff_efm3d_shaper_world_visualization_v1",
        "scene_id": task["scene_id"],
        "coordinate_frame": "raw_dataset_world",
        "input": str(input_dir),
        "prediction_csv": str(args.prediction_csv.resolve()),
        "shaper_dir": str(args.shaper_dir.resolve()),
        "view_task": str(args.view_task.resolve()),
        "num_detections": len(predictions),
        "num_reconstructions": len(object_records),
        "detection_instance_ids": sorted(prediction_ids),
        "reconstruction_instance_ids": sorted(reconstruction_ids),
        "skipped_detection_instance_ids": skipped_detection_ids,
        "num_context_points": int(len(context_points)),
        "num_mesh_display_vertices": int(len(mesh_points)),
        "combined_mesh": str(combined_path),
        "sheet": str(sheet_path),
        "objects": object_records,
        "renders": render_records,
    }
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sheet": str(sheet_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
