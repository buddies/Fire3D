#!/usr/bin/env python3
"""Run LiteReality object retrieval reproducibly with resumable timing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    os.chdir(baseline_root)
    retrieval_root = baseline_root / "litereality" / "LR_retrieval"
    sys.path.insert(0, str(retrieval_root))
    os.environ.setdefault(
        "QWEN_MODEL_PATH",
        str(baseline_root / "third_party" / "pre-trained" / "qwen3-vl-8b-instruct"),
    )
    os.environ.setdefault(
        "HF_HOME", str(baseline_root / "third_party" / "hf_cache")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from main_qwen import (  # pylint: disable=import-outside-toplevel
        process_images,
        process_object,
        setup_environment,
    )
    from qwan import unload_model as qwen_unload_model  # pylint: disable=import-outside-toplevel

    input_root = Path("input") / "object_stage" / args.scene_id
    output_root = Path("output") / "object_stage" / args.scene_id
    adapter_manifest = json.loads((input_root / "adapter_manifest.json").read_text())
    objects = adapter_manifest["objects"]
    if args.max_objects is not None:
        objects = objects[: args.max_objects]
    output_root.mkdir(parents=True, exist_ok=True)
    timing_path = output_root / "benchmark_timing.json"
    existing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    existing_by_name = {
        record["object_name"]: record for record in existing.get("objects", [])
    }

    process_started = time.time()
    load_started = time.time()
    config = setup_environment()
    model_load_seconds = time.time() - load_started
    records = []
    num_resumed_objects = 0
    try:
        for index, object_record in enumerate(objects):
            object_name = object_record["object_name"]
            selected_root = output_root / object_name / "selected_obj"
            if (
                not args.force
                and selected_root.exists()
                and any(selected_root.iterdir())
                and object_name in existing_by_name
                and existing_by_name[object_name].get("status") == "completed"
            ):
                records.append(existing_by_name[object_name])
                num_resumed_objects += 1
                print(f"[{index + 1}/{len(objects)}] resume {object_name}")
                continue
            object_input = input_root / object_name
            started = time.time()
            status = "failed"
            error = None
            try:
                stitched_image, semantic = process_images(str(object_input))
                if stitched_image is None:
                    raise RuntimeError("process_images returned no stitched image")
                result = process_object(
                    str(object_input),
                    config,
                    stitched_image,
                    semantic,
                    use_multi_image_selection=True,
                )
                if result is None:
                    raise RuntimeError("LiteReality process_object returned None")
                if not selected_root.exists() or not any(selected_root.iterdir()):
                    raise RuntimeError(f"No selected asset at {selected_root}")
                status = "completed"
            except Exception as exc:  # retain failures instead of aborting the scene
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            record = {
                "object_id": object_record["object_id"],
                "object_name": object_name,
                "semantic": object_record["semantic"],
                "status": status,
                "elapsed_seconds": time.time() - started,
                "error": error,
                "selected_asset_root": str(selected_root.resolve()),
            }
            records.append(record)
            partial = {
                "schema": "ff_litereality_retrieval_timing_v1",
                "scene_id": args.scene_id,
                "model_load_seconds": model_load_seconds,
                "objects": records,
                "updated_unix": time.time(),
                "complete": False,
            }
            timing_path.write_text(json.dumps(partial, indent=2))
            print(
                f"[{index + 1}/{len(objects)}] {object_name}: {status} "
                f"{record['elapsed_seconds']:.2f}s"
            )
    finally:
        qwen_unload_model()

    object_stage_sum_seconds = sum(
        float(row.get("elapsed_seconds", 0.0)) for row in records
    )
    report = {
        "schema": "ff_litereality_retrieval_timing_v1",
        "scene_id": args.scene_id,
        "model_load_seconds": model_load_seconds,
        "process_wall_seconds": time.time() - process_started,
        "object_stage_sum_seconds": object_stage_sum_seconds,
        # Unlike process_wall_seconds, this remains comparable after a resumed
        # scene run because cached objects retain their original elapsed time.
        "logical_scene_runtime_seconds": (
            model_load_seconds + object_stage_sum_seconds
        ),
        "num_resumed_objects": num_resumed_objects,
        "num_requested_objects": len(objects),
        "num_completed_objects": sum(row["status"] == "completed" for row in records),
        "num_failed_objects": sum(row["status"] != "completed" for row in records),
        "objects": records,
        "updated_unix": time.time(),
        "complete": len(records) == len(objects),
    }
    timing_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "scene_id": args.scene_id,
        "model_load_seconds": model_load_seconds,
        "process_wall_seconds": report["process_wall_seconds"],
        "logical_scene_runtime_seconds": report["logical_scene_runtime_seconds"],
        "num_resumed_objects": num_resumed_objects,
        "num_completed_objects": report["num_completed_objects"],
        "num_failed_objects": report["num_failed_objects"],
        "timing_path": str(timing_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
