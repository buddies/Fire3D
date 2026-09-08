"""ProcTHOR-specific helpers for voxelize_v2 scripts."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from utils import DEFAULT_TRAINING_ROOT, REPO_ROOT, load_raw_kit_utils

RAW_PROCTHOR_KIT = REPO_ROOT / "data_processing/datasets/ai2thor/procthor"
DEFAULT_PROCTHOR_ROOT = Path(
    os.environ.get("FIRE3D_PROCTHOR_ROOT", DEFAULT_TRAINING_ROOT / "ProcTHOR")
)
DEFAULT_AI2THOR_HAB_ROOT = DEFAULT_PROCTHOR_ROOT / "ai2thor-hab"


def status_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return path.read_text(errors="ignore").lstrip().startswith("completed")
    except OSError:
        return False


def has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def dependency_state(args: argparse.Namespace) -> dict[str, bool]:
    ai2thor_hab_root = args.ai2thor_hab_root
    config_scene_dir = ai2thor_hab_root / "configs" / "scenes" / "ProcTHOR"
    stage_dir = ai2thor_hab_root / "assets" / "stages" / "ProcTHOR"
    return {
        "ai2thor_status_completed_or_data_present": status_completed(args.ai2thor_status)
        or (has_entries(config_scene_dir) and has_entries(stage_dir)),
        "procthor_config_scenes_present": has_entries(config_scene_dir),
        "procthor_stage_assets_present": has_entries(stage_dir),
        "ai2thor_assets_present": has_entries(ai2thor_hab_root / "assets"),
        "blender_present": args.blender_path.exists() and os.access(args.blender_path, os.X_OK),
    }


def wait_for_dependencies(args: argparse.Namespace) -> None:
    while True:
        state = dependency_state(args)
        missing = [name for name, ok in state.items() if not ok]
        if not missing:
            print("All ProcTHOR voxelization dependencies are ready.", flush=True)
            return
        if not args.wait_for_deps:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")
        print(f"Waiting for dependencies: {', '.join(missing)}", flush=True)
        time.sleep(args.poll_seconds)


def configure_procthor_import(args: argparse.Namespace):
    module = load_raw_kit_utils("procthor", RAW_PROCTHOR_KIT)
    ai2thor_hab_root = args.ai2thor_hab_root
    assets_dir = ai2thor_hab_root / "assets"
    config_dir = ai2thor_hab_root / "configs"

    module.DATASET_DIR = str(ai2thor_hab_root)
    module.ASSETS_DIR = str(assets_dir)
    module.CONFIG_DIR = str(config_dir)
    module.CONFIG_SCENE_DIR = str(config_dir / "scenes")
    module.ASSETS_OBJECT_DIR = str(assets_dir / "objects")
    module.ASSETS_STAGE_DIR = str(assets_dir / "stages")
    module.PROCTHOR_CONFIG_SCENE_DIR = str(config_dir / "scenes" / "ProcTHOR")
    module.EXPORT_DIR = str(args.procthor_root / "exports")
    return module
