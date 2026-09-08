#!/usr/bin/env python3
"""Evaluate and aggregate a selected LiteReality/LC64 multi-scene comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.geometry_protocol import (  # noqa: E402
    aggregate_records,
    jsonable,
)


EVALUATOR = REPO_ROOT / "benchmarks/scene_reconstruction/evaluate_geometry.py"
METRICS = ("CD", "CD_x100", "F1", "NC", "precision", "recall")
BOUNDED = ("F1", "NC", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "baselines/_upstream/litereality",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results/baselines/litereality",
    )
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--n-points", type=int, default=100000)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a matching existing metric JSON instead of recomputing it.",
    )
    return parser.parse_args()


def run_evaluator(
    manifest: Path,
    scenes: list[str],
    prediction_root: Path,
    output: Path,
    fire3d_test_root: Path,
    n_points: int,
    allow_incomplete: bool,
    reuse_existing: bool,
) -> dict[str, Any]:
    if output.is_file() and reuse_existing:
        existing = json.loads(output.read_text())
        protocol = existing.get("protocol", {})
        selection = existing.get("selection", {})
        if (
            int(protocol.get("n_points", -1)) == n_points
            and len(selection.get("scene_ids", [])) == len(scenes)
            and set(selection.get("scene_ids", [])) == set(scenes)
            and Path(protocol.get("prediction_root", "")).resolve()
            == prediction_root.resolve()
        ):
            print(f"Reusing {output}", flush=True)
            return existing
        raise ValueError(
            f"Existing metrics do not match requested protocol: {output}"
        )
    command = [
        sys.executable,
        str(EVALUATOR),
        "--manifest",
        str(manifest),
        "--fire3d-test-root",
        str(fire3d_test_root),
        "--prediction-root",
        str(prediction_root),
        "--output",
        str(output),
        "--n-points",
        str(n_points),
    ]
    for scene_id in scenes:
        command.extend(["--scene", scene_id])
    if allow_incomplete:
        command.append("--allow-incomplete")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(output.read_text())


def compact_aggregation(value: dict[str, Any]) -> dict[str, Any]:
    partial = value["partial_scene_macro_metrics"]
    missing = value["zero_for_missing_scene_macro_metrics"]
    return {
        "complete": value["complete"],
        "coverage": value["coverage"],
        "conditional_scene_macro": {
            key: partial[key]["mean"] for key in METRICS
        },
        "zero_for_missing_scene_macro": {
            key: missing[key]["mean"] for key in BOUNDED
        },
        "by_scene": {
            scene_id: {
                "coverage": scene["coverage"],
                "num_expected_objects": scene["num_expected_objects"],
                "num_evaluated_objects": scene["num_evaluated_objects"],
                "conditional": {
                    key: scene["conditional_metrics"][key]["mean"]
                    for key in METRICS
                },
                "zero_for_missing": scene["zero_for_missing_metrics"],
            }
            for scene_id, scene in value["by_scene"].items()
        },
    }


def scene_runtime(
    baseline_root: Path,
    lc64_root: Path,
    scene_id: str,
    lc64_runtime_override: float | None = None,
) -> dict[str, Any]:
    adapter_path = (
        baseline_root
        / "input"
        / "object_stage"
        / scene_id
        / "adapter_manifest.json"
    )
    timing_path = (
        baseline_root
        / "output"
        / "object_stage"
        / scene_id
        / "benchmark_timing.json"
    )
    lc64_path = lc64_root / scene_id / "inference_summary.json"
    adapter = json.loads(adapter_path.read_text())
    timing = json.loads(timing_path.read_text())
    lc64 = json.loads(lc64_path.read_text())
    object_seconds = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in timing["objects"]
    )
    litereality_seconds = (
        float(adapter["elapsed_seconds"])
        + float(timing["model_load_seconds"])
        + object_seconds
    )
    lc64_seconds = (
        float(lc64_runtime_override)
        if lc64_runtime_override is not None
        else float(lc64["elapsed_seconds"])
    )
    return {
        "scene_id": scene_id,
        "num_expected_objects": len(timing["objects"]),
        "num_litereality_completed": sum(
            record["status"] == "completed" for record in timing["objects"]
        ),
        "litereality": {
            "adapter_seconds": float(adapter["elapsed_seconds"]),
            "model_load_seconds": float(timing["model_load_seconds"]),
            "object_stage_sum_seconds": object_seconds,
            "logical_total_seconds": litereality_seconds,
        },
        "lc64_seconds": lc64_seconds,
        "lc64_runtime_source": (
            "selection_manifest_reused_artifact"
            if lc64_runtime_override is not None
            else "inference_summary"
        ),
        "ratio_litereality_over_lc64": (
            litereality_seconds / lc64_seconds
        ),
    }


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.resolve().read_text())
    reused_artifacts = selection.get("reused_artifacts", {})
    args.output_root = args.output_root.resolve()
    args.baseline_root = args.baseline_root.resolve()
    metrics_root = args.output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)

    evaluations: dict[str, dict[str, dict[str, Any]]] = {
        "lc64": {},
        "litereality_equalized": {},
        "litereality_obbfit": {},
    }
    scene_ids: list[str] = []
    scene_dataset: dict[str, str] = {}
    for dataset in selection["datasets"]:
        name = dataset["name"]
        scenes = list(dataset["scenes"])
        scene_ids.extend(scenes)
        scene_dataset.update({scene_id: name for scene_id in scenes})
        manifest = (REPO_ROOT / dataset["manifest"]).resolve()
        evaluations["lc64"][name] = run_evaluator(
            manifest,
            scenes,
            args.output_root / "lc64",
            metrics_root / f"{name}_lc64.json",
            args.fire3d_test_root,
            args.n_points,
            allow_incomplete=False,
            reuse_existing=args.reuse_existing,
        )
        for variant in ("equalized", "obbfit"):
            key = f"litereality_{variant}"
            evaluations[key][name] = run_evaluator(
                manifest,
                scenes,
                args.output_root / "litereality_predictions" / variant,
                metrics_root / f"{name}_{key}.json",
                args.fire3d_test_root,
                args.n_points,
                allow_incomplete=True,
                reuse_existing=args.reuse_existing,
            )

    combined = {}
    for method, by_dataset in evaluations.items():
        records = [
            record
            for dataset_result in by_dataset.values()
            for record in dataset_result["objects"]
        ]
        combined[method] = aggregate_records(records, scene_ids)

    runtimes = [
        {
            **scene_runtime(
                args.baseline_root,
                args.output_root / "lc64",
                scene_id,
                reused_artifacts.get(scene_id, {}).get(
                    "lc64_runtime_seconds"
                ),
            ),
            "dataset": scene_dataset[scene_id],
        }
        for scene_id in scene_ids
    ]
    total_litereality = sum(
        row["litereality"]["logical_total_seconds"] for row in runtimes
    )
    total_lc64 = sum(row["lc64_seconds"] for row in runtimes)

    report = {
        "schema": "ff_litereality_lc64_ten_scene_comparison_v1",
        "selection": selection,
        "protocol": {
            "n_points": args.n_points,
            "fscore_threshold": 0.01,
            "aggregation": "scene macro: mean scenes(mean available objects)",
            "missing_policy": (
                "conditional CD/F1/NC plus bounded F1/NC/precision/recall "
                "with missing predictions assigned zero"
            ),
            "litereality_variants": {
                "equalized": "uniform canonical scaling; preserves aspect ratio",
                "obbfit": "privileged independent x/y/z fit to GT OBB extents",
            },
        },
        "per_dataset": {
            method: {
                dataset: compact_aggregation(value)
                for dataset, value in by_dataset.items()
            }
            for method, by_dataset in evaluations.items()
        },
        "combined_10_scene": {
            method: compact_aggregation(value)
            for method, value in combined.items()
        },
        "runtime": {
            "scope": "sum of per-scene one-GPU logical runtimes",
            "litereality_seconds": total_litereality,
            "lc64_seconds": total_lc64,
            "ratio_litereality_over_lc64": total_litereality / total_lc64,
            "by_scene": runtimes,
        },
    }
    output = args.output_root / "comparison_summary.json"
    output.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(json.dumps(jsonable(report["combined_10_scene"]), indent=2))
    print(json.dumps(jsonable(report["runtime"]), indent=2))
    print(output)


if __name__ == "__main__":
    main()
