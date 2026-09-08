#!/usr/bin/env python3
"""Resumable end-to-end SimRecon runner for one FF scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("ithor", "imaginarium"),
        default="imaginarium",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--simrecon-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--scene-id", default="diningroom_05")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cropformer-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--iterations-2dgs", type=int, default=7_000)
    parser.add_argument("--iterations-semantic", type=int, default=2_500)
    parser.add_argument(
        "--stages",
        default="prepare,cropformer,2dgs,semantic,evaluate",
        help="Comma-separated ordered subset of prepare,cropformer,2dgs,semantic,evaluate",
    )
    parser.add_argument("--force-stage", action="append", default=[])
    return parser.parse_args()


def run_stage(
    name: str,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    marker_dir: Path,
    env: dict[str, str],
    force: bool,
) -> None:
    marker = marker_dir / f"{name}.json"
    if marker.exists() and not force:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if int(previous.get("returncode", 1)) == 0:
            print(f"[skip] {name}: {marker}")
            return
        print(f"[retry] {name}: previous marker failed: {marker}")
    log_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = time.time()
    print(f"[run] {name}: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record = {
        "stage": name,
        "command": command,
        "cwd": str(cwd),
        "started_unix": started,
        "finished_unix": time.time(),
        "duration_seconds": time.time() - started,
        "returncode": process.returncode,
        "log": str(log_path),
    }
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"Stage {name} failed; see {log_path}")


def main() -> None:
    args = parse_args()
    stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    valid = {"prepare", "cropformer", "2dgs", "semantic", "evaluate"}
    if not set(stages) <= valid:
        raise ValueError(f"Unknown stages: {set(stages) - valid}")
    scene_dir = args.work_root / "data" / args.scene_id
    logs = args.work_root / "logs" / args.scene_id
    markers = args.work_root / "stages" / args.scene_id
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    if "prepare" in stages:
        run_stage(
            "prepare",
            [
                args.python,
                str(Path(__file__).with_name("prepare_imaginarium_scene.py")),
                "--dataset",
                args.dataset,
                "--data-root",
                str(args.data_root),
                "--manifest",
                str(args.manifest),
                "--output-root",
                str(args.work_root / "data"),
                "--scene-id",
                args.scene_id,
            ],
            args.repo_root,
            logs,
            markers,
            env,
            "prepare" in args.force_stage,
        )

    if "cropformer" in stages:
        config = (
            args.simrecon_root
            / "semantic_modules/CropFormer/configs/entityv2/entity_segmentation/"
            "cropformer_hornet_3x.yaml"
        )
        run_stage(
            "cropformer",
            [
                args.python,
                "run_cropformer.py",
                "--config-file",
                str(config),
                "--scene_dir",
                str(scene_dir),
                "--image_path_pattern",
                "images/*",
                "--dataset",
                "scannet",
                "--opts",
                "MODEL.WEIGHTS",
                str(args.cropformer_checkpoint),
            ],
            args.simrecon_root / "semantic_modules/CropFormer",
            logs,
            markers,
            env,
            "cropformer" in args.force_stage,
        )

    output_prefix = (
        args.simrecon_root
        / "output"
        / scene_dir.parent.name
        / scene_dir.name
    )
    model_2dgs = f"ff_{args.dataset}_2dgs"
    if "2dgs" in stages:
        run_stage(
            "2dgs",
            [
                args.python,
                "train_2dgs.py",
                "-s",
                str(scene_dir),
                "-m",
                model_2dgs,
                "--iterations",
                str(args.iterations_2dgs),
                "--save_iterations",
                str(args.iterations_2dgs),
                "--port",
                str(6010 + args.gpu),
                "--quiet",
            ],
            args.simrecon_root,
            logs,
            markers,
            env,
            "2dgs" in args.force_stage,
        )
        source_ply = (
            output_prefix
            / model_2dgs
            / "point_cloud"
            / f"iteration_{args.iterations_2dgs}"
            / "point_cloud.ply"
        )
        if not source_ply.exists():
            raise FileNotFoundError(source_ply)
        shutil.copy2(source_ply, scene_dir / "point_cloud.ply")

    model_semantic = f"ff_{args.dataset}_semantic"
    if "semantic" in stages:
        run_stage(
            "semantic",
            [
                args.python,
                "train_semantic.py",
                "-s",
                str(scene_dir),
                "-m",
                model_semantic,
                "--use_seg_feature",
                "--iterations",
                str(args.iterations_semantic),
                "--load_filter_segmap",
                "--consider_negative_labels",
            ],
            args.simrecon_root,
            logs,
            markers,
            env,
            "semantic" in args.force_stage,
        )

    semantic_label_dir = (
        output_prefix
        / model_semantic
        / "point_cloud"
        / f"iteration_{args.iterations_semantic}"
    )
    if "evaluate" in stages:
        run_stage(
            "evaluate",
            [
                args.python,
                str(Path(__file__).with_name("evaluate_imaginarium_perception.py")),
                "--dataset",
                args.dataset,
                "--repo-root",
                str(args.repo_root),
                "--data-root",
                str(args.data_root),
                "--scene-id",
                args.scene_id,
                "--prepared-scene-dir",
                str(scene_dir),
                "--semantic-label-dir",
                str(semantic_label_dir),
                "--output-dir",
                str(args.work_root / "metrics" / args.scene_id),
            ],
            args.repo_root,
            logs,
            markers,
            env,
            "evaluate" in args.force_stage,
        )

    print(args.work_root / "metrics" / args.scene_id / "metrics.json")


if __name__ == "__main__":
    main()
