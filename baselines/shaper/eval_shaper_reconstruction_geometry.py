from __future__ import annotations

"""Evaluate ShapeR reconstructions with the original FF geometry protocol.

Predictions are matched to dataset pickles by sample stem. Per-object CD x100,
F1, NC, precision, and recall are aggregated by first averaging objects inside
each scene and then averaging scene means equally.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.reconstruction.geometry_eval_utils import calculate_geometry_metrics


DEFAULT_DATASET_DIR = REPO_ROOT / "data/evaluation/shaper"
MESH_EXTENSIONS = (".glb", ".ply", ".obj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping sample stem to mesh path. It may be a direct "
            "mapping, a list of {name,pred_mesh_path} records, or an object with "
            "a 'predictions' mapping/list. Relative paths are resolved against "
            "the manifest parent."
        ),
    )
    parser.add_argument(
        "--prediction-space",
        choices=("original", "shaper_normalized", "unit_bbox"),
        default="original",
        help=(
            "Coordinate convention of predicted meshes. 'original' matches raw "
            "ShapeR mesh_vertices; 'shaper_normalized' uses ShapeR's 0.9/max(bounds) "
            "normalization; 'unit_bbox' assumes a [-0.5,0.5] canonical prediction "
            "and applies the GT full OBB side lengths (2*bounds)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--n-points", type=int, default=100000)
    parser.add_argument("--ff-fscore-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--identity-gt-prediction",
        action="store_true",
        help="Use each GT mesh as its prediction; intended only for metric smoke tests.",
    )
    args = parser.parse_args()
    if args.n_points <= 0:
        parser.error("--n-points must be positive")
    if args.ff_fscore_threshold <= 0:
        parser.error("--ff-fscore-threshold must be positive")
    if not args.identity_gt_prediction and args.prediction_root is None and args.prediction_manifest is None:
        parser.error("pass --prediction-root, --prediction-manifest, or --identity-gt-prediction")
    return args


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


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def as_mesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    if isinstance(value, trimesh.Scene):
        meshes = [geometry for geometry in value.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    raise TypeError(f"Unsupported mesh value: {type(value)!r}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    return as_mesh(trimesh.load(path, process=False))


def load_shaper_sample(path: Path) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, Any]]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: archive[key] for key in archive.files}
    elif path.suffix == ".pkl":
        # ShapeR-Evaluation is an official trusted dataset. Never use this path
        # for untrusted pickle files. Prefer the compact Fire3D NPZ release.
        with path.open("rb") as handle:
            sample = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported ShapeR GT sample: {path}")
    missing = [key for key in ("mesh_vertices", "mesh_faces", "bounds") if key not in sample]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    vertices = to_numpy(sample["mesh_vertices"]).astype(np.float64, copy=False).reshape(-1, 3)
    faces = to_numpy(sample["mesh_faces"]).astype(np.int64, copy=False).reshape(-1, 3)
    bounds = to_numpy(sample["bounds"]).astype(np.float64, copy=False).reshape(3)
    if not np.all(np.isfinite(bounds)) or float(bounds.max()) <= 0:
        raise ValueError(f"Invalid bounds in {path}: {bounds}")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    metadata = {
        "category": str(np.asarray(sample["category"]).item()) if "category" in sample else None,
        "caption": str(np.asarray(sample["caption"]).item()) if "caption" in sample else None,
        "num_input_points": int(to_numpy(sample["points_model"]).shape[0]) if "points_model" in sample else None,
        "num_slam_views": len(sample.get("image_data", [])),
        "num_rgb_views": len(sample.get("rgb_image_data", [])),
    }
    return mesh, bounds, metadata


def scene_name(sample_name: str) -> str:
    return sample_name.split("__", 1)[0]


def sample_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def parse_prediction_manifest(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    mapping: dict[str, Any]
    if isinstance(data, dict):
        mapping = data
    elif isinstance(data, list):
        mapping = {}
        for record in data:
            if not isinstance(record, dict):
                raise ValueError("Prediction manifest list entries must be objects")
            name = record.get("name") or record.get("sample") or record.get("sample_name")
            mesh_path = record.get("pred_mesh_path") or record.get("mesh_path") or record.get("path")
            if not name or not mesh_path:
                raise ValueError(f"Prediction manifest record lacks name/path: {record}")
            mapping[str(name)] = mesh_path
    else:
        raise ValueError("Unsupported prediction manifest JSON structure")

    resolved = {}
    for name, value in mapping.items():
        mesh_value = value
        if isinstance(value, dict):
            mesh_value = value.get("pred_mesh_path") or value.get("mesh_path") or value.get("path")
        if not mesh_value:
            continue
        mesh_path = Path(mesh_value).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        resolved[Path(str(name)).stem] = mesh_path.resolve()
    return resolved


def build_prediction_index(root: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for extension in MESH_EXTENSIONS:
        for path in root.rglob(f"*{extension}"):
            if path.name.startswith(("GT__", "PAIR__")):
                continue
            candidates[path.stem].append(path.resolve())
            if path.stem in {"pred_local", "pred", "prediction"}:
                candidates[path.parent.name].append(path.resolve())

    index = {}
    ambiguous = {}
    for name, paths in candidates.items():
        unique = sorted({str(path): path for path in paths}.values(), key=str)
        if len(unique) == 1:
            index[name] = unique[0]
        elif len(unique) > 1:
            ambiguous[name] = [str(path) for path in unique]
    return index, ambiguous


def prediction_in_original_space(
    mesh: trimesh.Trimesh,
    bounds: np.ndarray,
    space: str,
) -> trimesh.Trimesh:
    """Convert a prediction to the GT mesh's original canonical coordinate frame."""
    original = mesh.copy()
    scale = 0.9 / float(bounds.max())
    if space == "original":
        pass
    elif space == "shaper_normalized":
        original.apply_scale(1.0 / scale)
    elif space == "unit_bbox":
        original.vertices = np.asarray(original.vertices) * (2.0 * bounds[None])
    else:
        raise ValueError(space)
    return original


def metric_summary(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float | int | None]:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


METRIC_KEYS = {
    "ff_geometry": ("CD", "CD_x100", "F1", "NC", "precision", "recall"),
}


def metric_tree(
    records: list[dict[str, Any]],
    *,
    empty_as_zero: bool = False,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    evaluated = [record for record in records if record.get("status") == "evaluated"]
    metrics = {
        family: {
            key: metric_summary(evaluated, ("metrics", family, key))
            for key in keys
        }
        for family, keys in METRIC_KEYS.items()
    }
    if empty_as_zero and not evaluated:
        for family in metrics.values():
            for key in family:
                family[key] = {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return metrics


def summary_of_scene_means(
    by_scene: dict[str, dict[str, Any]],
    family: str,
    key: str,
) -> dict[str, float | int | None]:
    values = np.asarray(
        [scene["metrics"][family][key]["mean"] for scene in by_scene.values()],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    object_micro_metrics = metric_tree(records)
    by_scene = {}
    for name in sorted({record["scene"] for record in records}):
        scene_records = [record for record in records if record["scene"] == name]
        by_scene[name] = {
            "num_requested": len(scene_records),
            "num_evaluated": sum(record.get("status") == "evaluated" for record in scene_records),
            # The historical FF evaluator writes zero scene means when no object matches.
            "metrics": metric_tree(scene_records, empty_as_zero=True),
        }
    scene_macro_metrics = {
        family: {
            key: summary_of_scene_means(by_scene, family, key)
            for key in keys
        }
        for family, keys in METRIC_KEYS.items()
    }
    return {
        # Primary report: historical FF two-level mean, with every scene equally weighted.
        "metrics": scene_macro_metrics,
        "aggregation": {
            "primary": "scene_macro",
            "formula": "mean_over_scenes(mean_over_evaluated_objects_in_scene(metric))",
            "empty_scene_mean": 0.0,
            "missing_objects": "excluded_from_scene_mean",
            "num_scenes": len(by_scene),
        },
        # Secondary diagnostic retained so prior direct-object reports remain reproducible.
        "object_micro_metrics": object_micro_metrics,
        "by_scene": by_scene,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name", "scene", "status", "pkl_path", "pred_mesh_path", "error",
        "ff_cd", "ff_cd_x100", "ff_f1", "ff_nc", "ff_precision", "ff_recall",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            ff = (record.get("metrics") or {}).get("ff_geometry") or {}
            writer.writerow({
                "name": record["name"],
                "scene": record["scene"],
                "status": record["status"],
                "pkl_path": record["pkl_path"],
                "pred_mesh_path": record.get("pred_mesh_path"),
                "error": record.get("error"),
                "ff_cd": ff.get("CD"),
                "ff_cd_x100": ff.get("CD_x100"),
                "ff_f1": ff.get("F1"),
                "ff_nc": ff.get("NC"),
                "ff_precision": ff.get("precision"),
                "ff_recall": ff.get("recall"),
            })


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    sample_paths = sorted(dataset_dir.glob("*.npz")) or sorted(dataset_dir.glob("*.pkl"))
    if args.scene:
        wanted = set(args.scene)
        sample_paths = [path for path in sample_paths if scene_name(path.stem) in wanted]
    if args.max_samples is not None:
        sample_paths = sample_paths[: args.max_samples]
    if not sample_paths:
        raise FileNotFoundError(f"No selected .npz or .pkl files under {dataset_dir}")

    prediction_index: dict[str, Path] = {}
    ambiguous_predictions: dict[str, list[str]] = {}
    if args.prediction_root is not None:
        prediction_index, ambiguous_predictions = build_prediction_index(args.prediction_root.resolve())
    if args.prediction_manifest is not None:
        prediction_index.update(parse_prediction_manifest(args.prediction_manifest.resolve()))

    records = []
    for sample_path in sample_paths:
        name = sample_path.stem
        record: dict[str, Any] = {
            "name": name,
            "scene": scene_name(name),
            "pkl_path": str(sample_path),
            "pred_mesh_path": None,
            "prediction_space": args.prediction_space,
            "status": "pending",
            "metrics": None,
            "error": None,
        }
        try:
            gt_mesh, bounds, metadata = load_shaper_sample(sample_path)
            record["bounds"] = bounds.tolist()
            record["metadata"] = metadata
            pred_path = prediction_index.get(name)
            if args.identity_gt_prediction:
                pred_mesh = gt_mesh.copy()
                pred_original = pred_mesh
                record["prediction_source"] = "identity_gt_smoke"
            else:
                if name in ambiguous_predictions and pred_path is None:
                    record["status"] = "ambiguous_prediction"
                    record["prediction_candidates"] = ambiguous_predictions[name]
                    records.append(record)
                    continue
                if pred_path is None or not pred_path.is_file():
                    record["status"] = "missing_prediction"
                    records.append(record)
                    continue
                record["pred_mesh_path"] = str(pred_path)
                pred_mesh = load_mesh(pred_path)
                pred_original = prediction_in_original_space(pred_mesh, bounds, args.prediction_space)

            seed = sample_seed(args.seed, name)
            np.random.seed(seed)
            ff_metrics = calculate_geometry_metrics(
                pred_original,
                gt_mesh,
                n_points=args.n_points,
                f1_threshold=args.ff_fscore_threshold,
            )
            ff_metrics["CD_x100"] = float(ff_metrics["CD"]) * 100.0
            record["metrics"] = {"ff_geometry": ff_metrics}
            record["status"] = "evaluated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    aggregate_result = aggregate(records)
    num_missing = sum(record["status"] == "missing_prediction" for record in records)
    num_ambiguous = sum(record["status"] == "ambiguous_prediction" for record in records)
    num_failed = sum(record["status"] == "failed" for record in records)
    output = {
        "protocol": {
            "name": "shaper_reconstruction_ff_geometry_v2",
            "aggregation": "historical_ff_scene_macro",
            "dataset_dir": str(dataset_dir),
            "prediction_root": str(args.prediction_root.resolve()) if args.prediction_root else None,
            "prediction_manifest": str(args.prediction_manifest.resolve()) if args.prediction_manifest else None,
            "prediction_space": args.prediction_space,
            "n_points": args.n_points,
            "seed": args.seed,
            "ff_geometry": {
                "source": "eval/reconstruction/geometry_eval_utils.py",
                "chamfer": "symmetric Chamfer-L1 in original canonical coordinates",
                "reported_chamfer": "CD x100",
                "f_score_threshold": args.ff_fscore_threshold,
            },
        },
        "coverage": {
            "num_requested": len(records),
            "num_evaluated": sum(record["status"] == "evaluated" for record in records),
            "num_missing_prediction": num_missing,
            "num_ambiguous_prediction": num_ambiguous,
            "num_failed": num_failed,
        },
        **aggregate_result,
        "ambiguous_prediction_index": ambiguous_predictions,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2) + "\n")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(csv_output, records)
    print(json.dumps(jsonable({"coverage": output["coverage"], "metrics": output["metrics"]}), indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_output}")
    if (num_missing or num_ambiguous or num_failed) and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
