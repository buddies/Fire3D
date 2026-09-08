#!/usr/bin/env python3
"""Run EFM3D detection and ShapeR reconstruction for one prepared scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EFM3D_ROOT = REPO_ROOT / "baselines/_upstream/efm3d"
SHAPER_ROOT = REPO_ROOT / "baselines/_upstream/shaper"
VALID_STAGES = ("efm", "prepare", "reconstruct", "visualize")
PROTOCOL_DIR = Path(__file__).resolve().parent / "protocols"
DEFAULT_PROTOCOL = "shaper_scripts_v1"
PROTOCOL_SCHEMA = "ff_efm3d_shaper_inference_protocol_v1"

PROTOCOL_ARGUMENTS = {
    "frame_rate": ("efm3d", "frame_rate"),
    "frames_per_snippet": ("efm3d", "frames_per_snippet"),
    "stride_frames": ("efm3d", "stride_frames"),
    "num_snips": ("efm3d", "num_snips"),
    "max_points_per_frame": ("efm3d", "max_points_per_frame"),
    "point_source": ("efm3d", "point_source"),
    "det_threshold": ("efm3d", "det_threshold"),
    "tracker_inst_threshold": ("efm3d", "tracker_inst_threshold"),
    "tracker_assoc_threshold": ("efm3d", "tracker_assoc_threshold"),
    "voxel_res": ("efm3d", "voxel_res"),
    "obb_only": ("efm3d", "obb_only"),
    "prob_threshold": ("shaper", "prob_threshold"),
    "padding_scale": ("shaper", "padding_scale"),
    "max_object_points": ("shaper", "max_object_points"),
    "max_frames": ("shaper", "max_frames"),
    "shaper_config": ("shaper", "config"),
}


def parse_stages(value: str) -> list[str]:
    stages = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(stages) - set(VALID_STAGES))
    if invalid or not stages:
        raise ValueError(f"Invalid stages {invalid}; expected a subset of {VALID_STAGES}")
    return stages


def load_protocol(value: str) -> tuple[dict, Path]:
    candidate = Path(value).expanduser()
    path = candidate if candidate.is_file() else PROTOCOL_DIR / f"{value}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in PROTOCOL_DIR.glob("*.json")))
        raise FileNotFoundError(
            f"Unknown EFM3D + ShapeR protocol {value!r}; available: {available}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Invalid protocol schema in {path}: {payload.get('schema')!r}")
    for argument, (section, key) in PROTOCOL_ARGUMENTS.items():
        if key not in (payload.get(section) or {}):
            raise KeyError(f"Protocol {path} is missing {section}.{key} for {argument}")
    return payload, path.resolve()


def apply_protocol_defaults(args: argparse.Namespace, protocol: dict) -> argparse.Namespace:
    for argument, (section, key) in PROTOCOL_ARGUMENTS.items():
        if getattr(args, argument) is None:
            setattr(args, argument, protocol[section][key])
    return args


def sequence_name(scene_dir: Path) -> str:
    return f"{scene_dir.parent.name}_{scene_dir.name}" if scene_dir.name.isdigit() else scene_dir.name


def prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(path.resolve()), current) if value
    )


def run_logged(
    command: list[str], cwd: Path, log_path: Path, env: dict[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ithor", "imaginarium"), required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--view-task", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(VALID_STAGES))
    parser.add_argument("--gpu", default="6")
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
        help="Named protocol under protocols/ or a protocol JSON path.",
    )
    parser.add_argument(
        "--efm-python",
        default=os.environ.get("FIRE3D_EFM3D_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--shaper-python",
        default=os.environ.get("FIRE3D_SHAPER_PYTHON", "python"),
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=EFM3D_ROOT / "ckpt/model_release.pth",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=EFM3D_ROOT / "efm3d/config/evl_inf.yaml",
    )
    parser.add_argument("--reuse-efm-result-dir", type=Path)
    parser.add_argument("--prob-threshold", type=float)
    parser.add_argument("--frame-rate", type=int)
    parser.add_argument("--frames-per-snippet", type=int)
    parser.add_argument("--stride-frames", type=int)
    parser.add_argument("--num-snips", type=int)
    parser.add_argument("--max-points-per-frame", type=int)
    parser.add_argument(
        "--point-source",
        choices=("per-frame-visible", "per-frame-depth", "global-repeat", "none"),
        default=None,
    )
    parser.add_argument("--det-threshold", type=float)
    parser.add_argument("--tracker-inst-threshold", type=float)
    parser.add_argument("--tracker-assoc-threshold", type=float)
    parser.add_argument("--voxel-res", type=float)
    parser.add_argument(
        "--obb-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip EFM3D occupancy decoding and fusion.",
    )
    parser.add_argument("--padding-scale", type=float)
    parser.add_argument("--max-object-points", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--shaper-config",
        choices=("speed", "balance", "quality"),
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol, protocol_path = load_protocol(args.protocol)
    apply_protocol_defaults(args, protocol)
    args.protocol_name = protocol["name"]
    args.protocol_path = protocol_path
    return args


def main() -> None:
    args = parse_args()
    stages = parse_stages(args.stages)
    scene_dir = args.scene_dir.resolve()
    output_root = args.output_root.resolve()
    scene_id = sequence_name(scene_dir)
    model_checkpoint = args.model_checkpoint.resolve()
    model_config = args.model_config.resolve()
    logs_dir = output_root / "logs"
    efm_base = output_root / "efm" / args.dataset
    default_efm_result = efm_base / model_checkpoint.stem / scene_id
    efm_result_dir = (
        args.reuse_efm_result_dir.resolve()
        if args.reuse_efm_result_dir
        else default_efm_result
    )
    prepared_dir = output_root / "prepared" / args.dataset / scene_id
    reconstruction_dir = output_root / "reconstruction" / args.dataset / scene_id
    visualization_dir = output_root / "visualization" / args.dataset / scene_id
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    prepend_pythonpath(env, REPO_ROOT)

    if "efm" in stages:
        tracked_csv = efm_result_dir / "tracked_scene_obbs.csv"
        if tracked_csv.is_file() and not args.force:
            print(f"SKIP existing EFM3D result: {tracked_csv}")
        elif args.reuse_efm_result_dir:
            raise FileNotFoundError(
                f"Reused EFM3D directory has no tracked_scene_obbs.csv: {efm_result_dir}"
            )
        else:
            command = [
                args.efm_python,
                "infer_fisheye_scene.py",
                "--input",
                os.fspath(scene_dir),
                "--model_ckpt",
                os.fspath(model_checkpoint),
                "--model_cfg",
                os.fspath(model_config),
                "--output_dir",
                os.fspath(efm_base),
                "--frames_per_snippet",
                str(args.frames_per_snippet),
                "--frame_rate",
                str(args.frame_rate),
                "--stride_frames",
                str(args.stride_frames),
                "--num_snips",
                str(args.num_snips),
                "--max_points_per_frame",
                str(args.max_points_per_frame),
                "--point_source",
                args.point_source,
                "--det_threshold",
                str(args.det_threshold),
                "--tracker_inst_threshold",
                str(args.tracker_inst_threshold),
                "--tracker_assoc_threshold",
                str(args.tracker_assoc_threshold),
                "--voxel_res",
                str(args.voxel_res),
            ]
            if args.obb_only:
                command.append("--obb_only")
            run_logged(
                command,
                EFM3D_ROOT,
                logs_dir / f"{args.dataset}_{scene_id}_efm.log",
                env,
            )

    tracked_csv = efm_result_dir / "tracked_scene_obbs.csv"
    if not tracked_csv.is_file():
        raise FileNotFoundError(f"Missing EFM3D tracked boxes: {tracked_csv}")

    if "prepare" in stages:
        prepared_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                args.shaper_python,
                "preprocess_fisheye_scene.py",
                "--data_dir",
                os.fspath(scene_dir),
                "--obb_result_dir",
                os.fspath(efm_result_dir),
                "--save_dir",
                os.fspath(prepared_dir),
                "--prob_threshold",
                str(args.prob_threshold),
                "--padding_scale",
                str(args.padding_scale),
                "--max_points",
                str(args.max_object_points),
                "--max_frames",
                str(args.max_frames),
            ],
            SHAPER_ROOT,
            logs_dir / f"{args.dataset}_{scene_id}_prepare.log",
            env,
        )

    prepared_pickles = sorted(prepared_dir.glob("*.pkl"))
    if not prepared_pickles:
        raise FileNotFoundError(f"No ShapeR object pickles in {prepared_dir}")

    if "reconstruct" in stages:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.shaper_python,
            "infer_shapes_for_scene.py",
            "--input_pkl",
            os.fspath(prepared_dir),
            "--config",
            args.shaper_config,
            "--output_dir",
            os.fspath(reconstruction_dir),
            "--do_transform_to_world",
        ]
        if not args.force:
            command.append("--skip_existing")
        run_logged(
            command,
            SHAPER_ROOT,
            logs_dir / f"{args.dataset}_{scene_id}_shaper_{args.shaper_config}.log",
            env,
        )

    meshes = sorted(reconstruction_dir.glob("*.ply"))
    if len(meshes) != len(prepared_pickles):
        raise ValueError(
            f"ShapeR coverage mismatch: {len(meshes)} meshes for {len(prepared_pickles)} pickles"
        )

    if "visualize" in stages:
        if args.view_task is None:
            raise ValueError("--view-task is required for the visualize stage")
        run_logged(
            [
                args.efm_python,
                os.fspath(REPO_ROOT / "baselines/efm3d_shaper/visualize_scene.py"),
                "--input",
                os.fspath(scene_dir),
                "--prediction-csv",
                os.fspath(tracked_csv),
                "--shaper-dir",
                os.fspath(reconstruction_dir),
                "--view-task",
                os.fspath(args.view_task.resolve()),
                "--output-dir",
                os.fspath(visualization_dir),
                "--prob-threshold",
                str(args.prob_threshold),
            ],
            REPO_ROOT,
            logs_dir / f"{args.dataset}_{scene_id}_visualize.log",
            env,
        )

    summary = {
        "schema": "ff_efm3d_shaper_scene_pipeline_v1",
        "dataset": args.dataset,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "stages": stages,
        "gpu": str(args.gpu),
        "protocol": args.protocol_name,
        "protocol_path": str(args.protocol_path),
        "model_checkpoint": str(model_checkpoint),
        "model_config": str(model_config),
        "frame_rate": args.frame_rate,
        "frames_per_snippet": args.frames_per_snippet,
        "stride_frames": args.stride_frames,
        "num_snips": args.num_snips,
        "max_points_per_frame": args.max_points_per_frame,
        "point_source": args.point_source,
        "det_threshold": args.det_threshold,
        "tracker_inst_threshold": args.tracker_inst_threshold,
        "tracker_assoc_threshold": args.tracker_assoc_threshold,
        "voxel_res": args.voxel_res,
        "obb_only": args.obb_only,
        "prob_threshold": args.prob_threshold,
        "shaper_config": args.shaper_config,
        "efm_result_dir": str(efm_result_dir),
        "num_tracked_detections": count_csv_rows(tracked_csv),
        "prepared_dir": str(prepared_dir),
        "num_prepared_objects": len(prepared_pickles),
        "reconstruction_dir": str(reconstruction_dir),
        "num_reconstructed_objects": len(meshes),
        "visualization_dir": str(visualization_dir),
    }
    summary_path = output_root / f"pipeline_{args.dataset}_{scene_id}.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
