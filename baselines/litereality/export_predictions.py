#!/usr/bin/env python3
"""Export LiteReality assets in equalized and released OBB-fit geometry spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def as_mesh(value) -> trimesh.Trimesh:
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
    return as_mesh(trimesh.load(path, force="scene", process=False))


def find_geometry(selected_root: Path) -> Path:
    preferred = (
        "normalized_model.obj",
        "normalized_model.glb",
        "raw_model.obj",
        "raw_model.glb",
    )
    for name in preferred:
        matches = sorted(selected_root.rglob(name))
        if matches:
            return matches[0]
    for pattern in ("*.obj", "*.glb", "*.ply"):
        matches = sorted(selected_root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No supported geometry under {selected_root}")


def y_up_to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    )
    return result


def centered(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    if result.vertices.shape[0] == 0 or result.faces.shape[0] == 0:
        raise ValueError("Selected asset is empty")
    result.apply_translation(-result.bounds.mean(axis=0))
    return result


def unit_cube_equalized(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = centered(mesh)
    extent = float(result.extents.max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"Invalid selected-asset extent {extent}")
    result.apply_scale(1.0 / extent)
    return result


def gt_canonical_mesh(path: Path) -> trimesh.Trimesh:
    # Imaginarium/iTHOR source GLBs use the same raw-glTF -> FF-canonical
    # +90-degree-X conversion as the frozen evaluator.
    return centered(y_up_to_z_up(load_mesh(path)))


def obb_fit(mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = centered(mesh)
    source_extents = result.extents
    target_extents = gt_mesh.extents
    if np.any(source_extents <= 0) or np.any(target_extents <= 0):
        raise ValueError(
            f"Invalid OBB extents source={source_extents} target={target_extents}"
        )
    result.apply_scale(target_extents / source_extents)
    return result


def write_variant(
    output_root: Path,
    scene_id: str,
    variant: str,
    rows: list[dict],
) -> Path:
    scene_root = output_root / variant / scene_id
    objects_root = scene_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    predictions = {}
    variant_rows = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "meshes"}
        mesh = row.get("meshes", {}).get(variant)
        if mesh is not None:
            target = objects_root / f"object_{row['object_id']:04d}.ply"
            mesh.export(target)
            key = f"{scene_id}/object_{row['object_id']:04d}"
            predictions[key] = {
                "path": str(target),
                "prediction_space": "ff_canonical",
            }
            item.update(
                {
                    "status": "exported",
                    "prediction_path": str(target),
                    "prediction_space": "ff_canonical",
                    "canonical_extents": mesh.extents.tolist(),
                }
            )
        variant_rows.append(item)
    manifest = {
        "schema": "ff_litereality_prediction_manifest_v2",
        "scene_id": scene_id,
        "variant": variant,
        "num_expected": len(rows),
        "num_exported": len(predictions),
        "predictions": predictions,
        "objects": variant_rows,
    }
    path = scene_root / "prediction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    input_root = baseline_root / "input" / "object_stage" / args.scene_id
    retrieval_root = baseline_root / "output" / "object_stage" / args.scene_id
    adapter = json.loads((input_root / "adapter_manifest.json").read_text())
    timing_path = retrieval_root / "benchmark_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing_by_name = {row["object_name"]: row for row in timing.get("objects", [])}

    rows = []
    for record in adapter["objects"]:
        object_id = int(record["object_id"])
        object_name = record["object_name"]
        selected_root = retrieval_root / object_name / "selected_obj"
        row = {
            "object_id": object_id,
            "object_name": object_name,
            "semantic": record["semantic"],
            "status": "missing",
            "selected_root": str(selected_root),
            "source_geometry": None,
            "gt_mesh_path": record["gt_mesh_path"],
            "retrieval_timing": timing_by_name.get(object_name),
            "meshes": {},
        }
        if selected_root.exists():
            try:
                source = find_geometry(selected_root)
                # LiteReality's normalized OBJ/GLB assets are Y-up. Convert to
                # the benchmark's Z-up canonical object frame before scaling.
                selected = centered(y_up_to_z_up(load_mesh(source)))
                gt_mesh = gt_canonical_mesh(Path(record["gt_mesh_path"]))
                row.update(
                    {
                        "status": "ready",
                        "source_geometry": str(source),
                        "source_y_up_extents": load_mesh(source).extents.tolist(),
                        "gt_canonical_extents": gt_mesh.extents.tolist(),
                        "meshes": {
                            "equalized": unit_cube_equalized(selected),
                            "obbfit": obb_fit(selected, gt_mesh),
                        },
                    }
                )
            except Exception as exc:
                row["status"] = "export_failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    output_root = args.output_root.resolve()
    paths = {
        variant: write_variant(output_root, args.scene_id, variant, rows)
        for variant in ("equalized", "obbfit")
    }
    print(json.dumps({
        "scene_id": args.scene_id,
        "num_expected": len(rows),
        "num_exported": sum(row["status"] == "ready" for row in rows),
        "manifests": {key: str(value) for key, value in paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
