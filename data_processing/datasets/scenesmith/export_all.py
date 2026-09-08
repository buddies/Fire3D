#!/usr/bin/env python3
"""Extract and export all downloaded SceneSmith scenes.

The Hugging Face example-scenes snapshot stores each scene as a category-level
``scene_*.tar`` archive.  This script builds ``scene_paths.json`` from those
archives, extracts missing scenes in place, and calls the patched SceneSmith
export helper for each scene directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "FIRE3D_SCENESMITH_ROOT",
        os.environ.get(
            "SCENESMITH_DATASET_ROOT",
            REPO_ROOT / "data/training_scenes/Scenesmith",
        ),
    )
)
DEFAULT_KIT_ROOT = Path(
    os.environ.get(
        "FIRE3D_SCENESMITH_KIT_ROOT",
        os.environ.get(
            "SCENESMITH_KIT_ROOT",
            REPO_ROOT / "data_processing/_upstream/scenesmith",
        ),
    )
)
EXCLUDED_ROOT_DIRS = {
    ".cache",
    "export_logs",
    "latents",
    "pbr_latents",
    "realistic",
    "renders",
    "scenes",
    "ss_latents",
    "transforms",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def has_scene_payload(scene_dir: Path) -> bool:
    combined = scene_dir / "combined_house"
    return (combined / "house_state.json").exists() or (combined / "house.dmd.yaml").exists()


def safe_log_stem(scene_path: Path, dataset_root: Path) -> str:
    try:
        rel = scene_path.relative_to(dataset_root)
    except ValueError:
        rel = scene_path
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel)).strip("_")


def discover_archives(dataset_root: Path, categories: set[str] | None = None) -> list[Path]:
    archives: list[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name in EXCLUDED_ROOT_DIRS or child.name.startswith("."):
            continue
        if categories is not None and child.name not in categories:
            continue
        archives.extend(sorted(child.glob("scene_*.tar")))
    return archives


def _safe_extractall(tar: tarfile.TarFile, out_dir: Path) -> None:
    root = out_dir.resolve()
    members = tar.getmembers()
    for member in members:
        target = (out_dir / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe tar member outside target dir: {member.name}")
    tar.extractall(out_dir, members=members)


def extract_if_needed(tar_path: Path, scene_dir: Path) -> None:
    if has_scene_payload(scene_dir):
        return
    scene_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        _safe_extractall(tf, scene_dir)
    if not has_scene_payload(scene_dir):
        raise RuntimeError(f"extracted archive has no combined_house metadata: {tar_path}")


def build_scene_records(
    dataset_root: Path,
    scene_paths_json: Path,
    categories: set[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    archives = discover_archives(dataset_root, categories)
    records = [
        {
            "tar_path": str(tar_path),
            "scene_path": str(tar_path.with_suffix("")),
            "category": tar_path.parent.name,
            "scene_name": tar_path.stem,
        }
        for tar_path in archives
    ]
    if limit is not None:
        records = records[:limit]

    scene_paths_json.parent.mkdir(parents=True, exist_ok=True)
    scene_paths_json.write_text(
        json.dumps([record["scene_path"] for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path = scene_paths_json.with_name(scene_paths_json.stem + "_records.json")
    metadata_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def expected_transform_paths(scene_path: Path, dataset_root: Path) -> list[Path]:
    # Most SceneSmith example scenes are single-room; this handles already-exported
    # scenes without importing heavy SceneSmith/Drake dependencies in the scheduler.
    pattern = f"{scene_path.parent.name}_{scene_path.name}_room_*.pkl"
    return sorted((dataset_root / "transforms").glob(pattern))


def export_scene(record: dict[str, str], args: argparse.Namespace) -> dict[str, str | int | None]:
    dataset_root = Path(args.dataset_root)
    kit_root = Path(args.kit_root)
    scene_path = Path(record["scene_path"])
    tar_path = Path(record["tar_path"])
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{safe_log_stem(scene_path, dataset_root)}.log"

    if args.skip_existing and expected_transform_paths(scene_path, dataset_root):
        return {"scene_path": str(scene_path), "returncode": 0, "log": str(log_path), "status": "skipped"}

    env = os.environ.copy()
    env["SCENESMITH_DATASET_ROOT"] = str(dataset_root)
    env["SCENESMITH_KIT_ROOT"] = str(kit_root)
    env["PYTHONPATH"] = str(kit_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        extract_if_needed(tar_path, scene_path)
        cmd = [args.python, str(args.export_script), str(scene_path)]
        with log_path.open("w", encoding="utf-8") as log_file:
            print(f"[{timestamp()}] exporting {scene_path}", file=log_file)
            print("cmd:", " ".join(cmd), file=log_file)
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
        status = "ok" if proc.returncode == 0 else "failed"
        return {
            "scene_path": str(scene_path),
            "returncode": proc.returncode,
            "log": str(log_path),
            "status": status,
        }
    except Exception as exc:  # noqa: BLE001 - written to per-scene log for batch diagnosis.
        with log_path.open("a", encoding="utf-8") as log_file:
            print(f"[{timestamp()}] exception while exporting {scene_path}: {exc}", file=log_file)
            traceback.print_exc(file=log_file)
        return {"scene_path": str(scene_path), "returncode": 1, "log": str(log_path), "status": "exception"}


def parse_categories(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    categories: set[str] = set()
    for value in values:
        categories.update(part for part in value.split(",") if part)
    return categories or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT_ROOT)
    parser.add_argument("--scene-paths-json", type=Path, default=None)
    parser.add_argument("--export-script", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", help="Restrict to one or more root category dirs; comma-separated is allowed.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes with an existing exported transform pkl.")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.kit_root = args.kit_root.expanduser().resolve()
    args.scene_paths_json = (
        args.scene_paths_json.expanduser().resolve()
        if args.scene_paths_json
        else args.dataset_root / "scene_paths.json"
    )
    args.export_script = (
        args.export_script.expanduser().resolve()
        if args.export_script
        else args.kit_root / "scripts" / "export_scene_and_transforms.py"
    )
    args.log_dir = args.log_dir.expanduser().resolve() if args.log_dir else args.dataset_root / "export_logs"
    args.status_json = args.status_json.expanduser().resolve() if args.status_json else args.dataset_root / "export_status.json"
    args.categories = parse_categories(args.category)
    return args


def main() -> int:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {args.dataset_root}")
    if not args.export_script.is_file():
        raise SystemExit(f"export script does not exist: {args.export_script}")

    records = build_scene_records(args.dataset_root, args.scene_paths_json, args.categories, args.limit)
    print(f"Found {len(records)} SceneSmith archives")
    print(f"Wrote {args.scene_paths_json}")
    if not records:
        return 0

    results: list[dict[str, str | int | None]] = []
    max_workers = max(1, min(args.max_workers, len(records)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(export_scene, record, args) for record in records]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{idx}/{len(records)}] {result['status']} rc={result['returncode']} "
                f"scene={result['scene_path']} log={result['log']}",
                flush=True,
            )
            args.status_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    failed = [result for result in results if result.get("returncode") != 0]
    print(f"Finished {len(results)} scenes; failed={len(failed)}; status={args.status_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
