#!/usr/bin/env python3
"""Batch-export Imaginarium source scenes into canonical Fire3D assets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(
    os.environ.get(
        "FIRE3D_IMAGINARIUM_ROOT",
        REPO_ROOT / "data/training_scenes/Imaginarium",
    )
)


def discover_scenes(source_root: Path) -> list[tuple[str, Path, Path]]:
    records = []
    for scene_type in sorted(source_root.iterdir()):
        if not scene_type.is_dir():
            continue
        for scene_dir in sorted(path for path in scene_type.iterdir() if path.is_dir()):
            blend = scene_dir / f"{scene_dir.name}.blend"
            metadata = scene_dir / f"{scene_dir.name}_meta.json"
            if blend.is_file() and metadata.is_file():
                records.append((scene_dir.name, blend, metadata))
    return records


def export_one(
    record: tuple[str, Path, Path],
    script: Path,
    dataset_root: Path,
    blender_path: Path,
    log_dir: Path,
) -> tuple[str, int]:
    scene_name, blend, metadata = record
    log_path = log_dir / f"{scene_name}.log"
    command = [
        sys.executable,
        str(script),
        "--file_path",
        str(blend),
        "--meta_path",
        str(metadata),
        "--dataset-root",
        str(dataset_root),
        "--blender-path",
        str(blender_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return scene_name, result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--blender-path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root or (
        args.dataset_root / "asset_data/imaginarium_3d_scene_layout_dataset"
    )
    args.log_dir = args.log_dir or args.dataset_root / "logs_export"
    args.blender_path = args.blender_path or (
        REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"
    )
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> int:
    args = parse_args()
    script = Path(__file__).with_name("decompose_scene_glb.py")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records = discover_scenes(args.source_root)
    if not args.overwrite:
        records = [
            record
            for record in records
            if not (args.dataset_root / "transforms" / f"{record[0]}.pkl").is_file()
        ]
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Exporting {len(records)} Imaginarium scenes", flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                export_one,
                record,
                script,
                args.dataset_root,
                args.blender_path,
                args.log_dir,
            )
            for record in records
        ]
        for future in as_completed(futures):
            scene_name, returncode = future.result()
            print(f"{scene_name}: returncode={returncode}", flush=True)
            if returncode:
                failures.append(scene_name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
