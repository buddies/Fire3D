from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.common.comparison import (
    atomic_write_json,
    exact_render_dir,
    load_assignment_scenes,
    load_manifest_scene,
    partition_scenes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3D Objects on the frozen exact-rerender iTHOR/Imaginarium scene set."
    )
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "baselines/_upstream/sam3d",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = partition_scenes(
        load_assignment_scenes(args.assignments.resolve()),
        args.worker_index,
        args.num_workers,
    )
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()

    conda_prefix = Path(sys.executable).resolve().parents[1]
    os.environ.setdefault("CONDA_PREFIX", str(conda_prefix))
    os.environ.setdefault("CUDA_HOME", str(conda_prefix))
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    sam3d_mv = None if args.dry_run else importlib.import_module("sam3d_mv")

    records = []
    for scene_number, scene in enumerate(scenes, start=1):
        dataset = scene["dataset"]
        scene_id = scene["scene_id"]
        render_dir = exact_render_dir(args.source_run.resolve(), dataset, scene_id)
        scene_output = output_root / dataset / scene_id
        objects_dir = scene_output / "objects"
        status_path = scene_output / "status.json"
        manifest_scene = load_manifest_scene(REPO_ROOT, dataset, scene_id)
        expected_ids = sorted(int(value) for value in manifest_scene["visible_object_ids"])
        if args.max_objects is not None:
            if args.max_objects <= 0:
                raise ValueError("--max-objects must be positive")
            expected_ids = expected_ids[: args.max_objects]

        if status_path.exists() and not args.force:
            previous = __import__("json").loads(status_path.read_text())
            if previous.get("complete"):
                print(f"[{scene_number}/{len(scenes)}] complete, skipping {dataset}/{scene_id}", flush=True)
                records.append(previous)
                continue

        print(
            f"[{scene_number}/{len(scenes)}] SAM3D {dataset}/{scene_id}: "
            f"{len(expected_ids)} visible objects",
            flush=True,
        )
        started = time.time()
        error = None
        if not args.dry_run:
            try:
                sam3d_mv.main(
                    dataset,
                    str(render_dir),
                    str(objects_dir),
                    seed=args.seed,
                    compile_model=args.compile,
                    render_id=0,
                    skip_existing=not args.force,
                    pose_source="sam3d",
                    max_objects=args.max_objects,
                )
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)

        produced_ids = sorted(
            int(path.stem.split("_")[-1]) for path in objects_dir.glob("object_*.glb")
        )
        missing_ids = sorted(set(expected_ids) - set(produced_ids))
        record = {
            "baseline": "sam3d-objects",
            "protocol": "exact_rgb_gt_instance_mask_native_sam3d_pose",
            "dataset": dataset,
            "scene_id": scene_id,
            "render_dir": str(render_dir),
            "output_dir": str(scene_output),
            "expected_visible_object_ids": expected_ids,
            "produced_object_ids": produced_ids,
            "missing_object_ids": missing_ids,
            "elapsed_seconds": time.time() - started,
            "error": error,
            "complete": error is None and not missing_ids and not args.dry_run,
            "dry_run": args.dry_run,
        }
        atomic_write_json(status_path, record)
        records.append(record)

    summary = {
        "baseline": "sam3d-objects",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "num_assigned_scenes": len(scenes),
        "num_complete_scenes": sum(bool(record.get("complete")) for record in records),
        "scenes": records,
    }
    atomic_write_json(output_root / "workers" / f"worker_{args.worker_index:02d}.json", summary)
    return 0 if all(record.get("complete") or args.dry_run for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
