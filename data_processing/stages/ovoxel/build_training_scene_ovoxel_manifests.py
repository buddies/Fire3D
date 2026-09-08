#!/usr/bin/env python3
"""Build native per-dataset train-scene manifests for scene o-voxelization."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_JSON = Path(
    os.environ.get(
        "FIRE3D_SCENE_SPLIT",
        REPO_ROOT / "data/training_manifests/scene_train.json",
    )
)
DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)

DATASET_ROOT_NAMES = {
    "InternScenes": "InternScenes",
    "MansionWorld": "MansionWorld",
    "ProcTHOR": "ProcTHOR",
    "SAGE-10k": "SAGE-10k",
    "SceneSmith": "Scenesmith",
}

DATASET_ORDER = ["SceneSmith", "InternScenes", "MansionWorld", "SAGE-10k", "ProcTHOR"]


def intern_native_scene(split_scene_name: str) -> str:
    prefix = "InternScenes_"
    if not split_scene_name.startswith(prefix):
        raise ValueError(f"Unexpected InternScenes split name: {split_scene_name}")
    raw = split_scene_name[len(prefix) :]
    if raw.startswith("3rscan_"):
        return "3rscan/" + raw[len("3rscan_") :]
    if raw.startswith("scannet_"):
        return "scannet/" + raw[len("scannet_") :]
    if raw.startswith("arkitscenes_Training_"):
        return "arkitscenes/Training/" + raw[len("arkitscenes_Training_") :]
    if raw.startswith("arkitscenes_Validation_"):
        return "arkitscenes/Validation/" + raw[len("arkitscenes_Validation_") :]
    if raw.startswith("matterport3d_"):
        parts = raw.split("_")
        if len(parts) >= 3 and parts[-1].startswith("region"):
            return f"matterport3d/{parts[1]}/{parts[-1]}"
    return raw.replace("_", "/")


def parse_mansionworld(scene_id: str) -> tuple[str, str]:
    match = re.match(r"(.+)_floor_(\d+)_room_(.+)$", scene_id)
    if not match:
        raise ValueError(f"Unexpected MansionWorld scene_id: {scene_id}")
    scene_root, floor_id, room_id = match.groups()
    return f"{scene_root}/floor_{floor_id}.json", room_id


def parse_procthor(scene_id: str) -> tuple[str, str]:
    match = re.match(r"(.+)_room_([^_]+)$", scene_id)
    if not match:
        raise ValueError(f"Unexpected ProcTHOR scene_id: {scene_id}")
    scene_base, room_id = match.groups()
    parts = scene_base.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected ProcTHOR scene base: {scene_base}")
    return f"{parts[0]}/{parts[1]}/{parts[2]}", room_id


def native_record(record: dict[str, Any]) -> dict[str, Any]:
    dataset = str(record["dataset_name"])
    scene_id = str(record["scene_id"])
    scene_name = str(record["scene_name"])
    out: dict[str, Any] = {
        "dataset_name": dataset,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "video_ids": record.get("video_ids", []),
        "data_names": record.get("data_names", []),
    }
    if dataset == "InternScenes":
        out["native_scene"] = intern_native_scene(scene_name)
        out["native_key"] = out["native_scene"]
    elif dataset == "SceneSmith":
        out["native_room"] = scene_id
        out["native_key"] = scene_id
    elif dataset == "SAGE-10k":
        out["native_scene"] = scene_id
        out["native_key"] = scene_id
    elif dataset == "MansionWorld":
        scene_path, room_id = parse_mansionworld(scene_id)
        out["native_scene_path"] = scene_path
        out["native_room"] = room_id
        out["native_key"] = f"{scene_path}\t{room_id}"
    elif dataset == "ProcTHOR":
        scene, room_id = parse_procthor(scene_id)
        out["native_scene"] = scene
        out["native_room"] = room_id
        out["native_key"] = f"{scene}\t{room_id}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return out


def ordered_unique(values: list[str]) -> list[str]:
    return list(OrderedDict((value, None) for value in values).keys())


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def manifest_dir(training_root: Path, dataset: str, split_name: str) -> Path:
    return training_root / DATASET_ROOT_NAMES[dataset] / "ovoxels" / "manifests" / split_name


def write_dataset_manifests(
    training_root: Path,
    dataset: str,
    records: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    out_dir = manifest_dir(training_root, dataset, split_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "records.jsonl", records)

    scene_list: list[str] = []
    room_list: list[str] = []
    scene_room_list: list[str] = []
    for record in records:
        if dataset in {"InternScenes", "SAGE-10k"}:
            scene_list.append(str(record["native_scene"]))
        elif dataset == "SceneSmith":
            room_list.append(str(record["native_room"]))
        elif dataset == "MansionWorld":
            scene_list.append(str(record["native_scene_path"]))
            scene_room_list.append(f"{record['native_scene_path']}\t{record['native_room']}")
        elif dataset == "ProcTHOR":
            scene_list.append(str(record["native_scene"]))
            scene_room_list.append(f"{record['native_scene']}\t{record['native_room']}")

    scene_list = ordered_unique(scene_list)
    room_list = ordered_unique(room_list)
    scene_room_list = ordered_unique(scene_room_list)
    write_lines(out_dir / "scene_list.txt", scene_list)
    write_lines(out_dir / "room_list.txt", room_list)
    write_lines(out_dir / "scene_room_list.tsv", scene_room_list)

    smoke_records = records[:1]
    write_jsonl(out_dir / "smoke_records.jsonl", smoke_records)
    if smoke_records:
        first = smoke_records[0]
        if dataset in {"InternScenes", "SAGE-10k"}:
            write_lines(out_dir / "smoke_scene_list.txt", [str(first["native_scene"])])
            write_lines(out_dir / "smoke_room_list.txt", [])
            write_lines(out_dir / "smoke_scene_room_list.tsv", [])
        elif dataset == "SceneSmith":
            write_lines(out_dir / "smoke_scene_list.txt", [])
            write_lines(out_dir / "smoke_room_list.txt", [str(first["native_room"])])
            write_lines(out_dir / "smoke_scene_room_list.tsv", [])
        elif dataset == "MansionWorld":
            write_lines(out_dir / "smoke_scene_list.txt", [str(first["native_scene_path"])])
            write_lines(out_dir / "smoke_room_list.txt", [])
            write_lines(out_dir / "smoke_scene_room_list.tsv", [f"{first['native_scene_path']}\t{first['native_room']}"])
        elif dataset == "ProcTHOR":
            write_lines(out_dir / "smoke_scene_list.txt", [str(first["native_scene"])])
            write_lines(out_dir / "smoke_room_list.txt", [])
            write_lines(out_dir / "smoke_scene_room_list.tsv", [f"{first['native_scene']}\t{first['native_room']}"])
    else:
        write_lines(out_dir / "smoke_scene_list.txt", [])
        write_lines(out_dir / "smoke_room_list.txt", [])
        write_lines(out_dir / "smoke_scene_room_list.tsv", [])

    summary = {
        "dataset_name": dataset,
        "split_name": split_name,
        "record_count": len(records),
        "scene_count": len(scene_list),
        "room_count": len(room_list),
        "scene_room_count": len(scene_room_list),
        "records": str(out_dir / "records.jsonl"),
        "scene_list": str(out_dir / "scene_list.txt"),
        "room_list": str(out_dir / "room_list.txt"),
        "scene_room_list": str(out_dir / "scene_room_list.tsv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--manifest-split-name", default="train")
    parser.add_argument("--dataset", action="append", choices=DATASET_ORDER, default=[])
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.split_json.read_text())
    raw_records = data.get("scenes")
    if not isinstance(raw_records, list):
        raise ValueError(f"Split JSON does not contain a scenes list: {args.split_json}")
    wanted = args.dataset or DATASET_ORDER
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        dataset = raw.get("dataset_name")
        if dataset not in wanted:
            continue
        grouped[str(dataset)].append(native_record(raw))

    summaries = []
    for dataset in wanted:
        summaries.append(
            write_dataset_manifests(
                args.training_root,
                dataset,
                grouped.get(dataset, []),
                args.manifest_split_name,
            )
        )

    total = {
        "split_json": str(args.split_json),
        "manifest_split_name": args.manifest_split_name,
        "training_root": str(args.training_root),
        "datasets": summaries,
        "total_records": sum(item["record_count"] for item in summaries),
    }
    summary_path = (
        args.summary_json
        or args.training_root / "_ovoxel_manifests" / f"{args.manifest_split_name}_scene_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(total, indent=2, sort_keys=True) + "\n")
    print(json.dumps(total, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
