import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = os.environ.get(
    "FIRE3D_MANIFEST_ROOT", str(REPO_ROOT / "data/training_manifests")
)
DEFAULT_SPLIT_PREFIX = "obj_gen_occu_with_objects"


def stable_uint32(*parts):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def split_path(split_dir, prefix, kind, split):
    return os.path.join(split_dir, f"{prefix}_{kind}_{split}.json")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_scene_split(split, split_dir=DEFAULT_SPLIT_DIR, prefix=DEFAULT_SPLIT_PREFIX):
    return load_json(split_path(split_dir, prefix, "scene", split))


def load_object_split(split, split_dir=DEFAULT_SPLIT_DIR, prefix=DEFAULT_SPLIT_PREFIX):
    return load_json(split_path(split_dir, prefix, "object", split))


def load_interleaved_split(split, split_dir=DEFAULT_SPLIT_DIR, prefix=DEFAULT_SPLIT_PREFIX):
    return load_json(split_path(split_dir, prefix, "interleaved", split))


def scene_sequence_names(scene_split):
    names = []
    for scene in scene_split.get("scenes", []):
        names.extend(scene.get("data_names", []))
    return names


def build_interleaved_split(
    scene_split,
    object_split,
    scene_data_weight=1,
    object_data_weight=3,
    num_object_data_per_sample=64,
):
    scene_names = scene_sequence_names(scene_split)
    object_names = list(object_split.get("object_names", []))

    scene_entries = [
        {"sample_type": "scene", "data_name": data_name}
        for _ in range(max(int(scene_data_weight), 0))
        for data_name in scene_names
    ]

    object_pseudo_count = (
        max(len(scene_names), 1) * max(int(object_data_weight), 0)
        if object_names
        else 0
    )
    object_entries = []
    for object_idx in range(object_pseudo_count):
        object_start = (object_idx * int(num_object_data_per_sample)) % len(object_names)
        data_name = f"object_pseudo_{object_idx:08d}"
        object_entries.append(
            {
                "sample_type": "object",
                "data_name": data_name,
                "object_start": object_start,
                "num_objects": int(num_object_data_per_sample),
                "seed": stable_uint32(scene_split.get("split", ""), data_name, object_start),
            }
        )

    entries = []
    object_cursor = 0
    object_data_weight = max(int(object_data_weight), 0)
    base_scene_count = len(scene_names)
    for scene_idx, scene_entry in enumerate(scene_entries):
        entries.append(scene_entry)
        if scene_idx < base_scene_count:
            for _ in range(object_data_weight):
                if object_cursor < len(object_entries):
                    entries.append(object_entries[object_cursor])
                    object_cursor += 1
    entries.extend(object_entries[object_cursor:])

    return {
        "exp_name": scene_split.get("exp_name", object_split.get("exp_name")),
        "split": scene_split.get("split", object_split.get("split")),
        "count": len(entries),
        "scene_sequence_count": len(scene_entries),
        "object_pseudo_count": len(object_entries),
        "object_count": len(object_names),
        "scene_data_weight": int(scene_data_weight),
        "object_data_weight": int(object_data_weight),
        "num_object_data_per_sample": int(num_object_data_per_sample),
        "entries": entries,
    }


def write_interleaved_split_files(
    split_dir=DEFAULT_SPLIT_DIR,
    prefix=DEFAULT_SPLIT_PREFIX,
    scene_data_weight=1,
    object_data_weight=3,
    num_object_data_per_sample=64,
):
    os.makedirs(split_dir, exist_ok=True)
    output_paths = []
    for split in ("train", "trainval", "val"):
        scene_split = load_scene_split(split, split_dir=split_dir, prefix=prefix)
        object_split = load_object_split(split, split_dir=split_dir, prefix=prefix)
        interleaved = build_interleaved_split(
            scene_split,
            object_split,
            scene_data_weight=scene_data_weight,
            object_data_weight=object_data_weight,
            num_object_data_per_sample=num_object_data_per_sample,
        )
        path = split_path(split_dir, prefix, "interleaved", split)
        with open(path, "w") as f:
            json.dump(interleaved, f, indent=2)
        output_paths.append(path)
    return output_paths


def materialize_interleaved_entries(
    interleaved_split,
    scene_data_by_name,
    strict=True,
):
    data_list = []
    missing_scene_names = []
    for entry in interleaved_split.get("entries", []):
        sample_type = entry.get("sample_type")
        if sample_type == "scene":
            data_name = entry["data_name"]
            scene_item = scene_data_by_name.get(data_name)
            if scene_item is None:
                missing_scene_names.append(data_name)
                continue
            data_list.append(scene_item)
        elif sample_type == "object":
            data_list.append(dict(entry))
        else:
            raise ValueError(f"Unknown interleaved sample_type: {sample_type}")

    if missing_scene_names and strict:
        preview = ", ".join(missing_scene_names[:5])
        raise ValueError(
            f"Interleaved split references {len(missing_scene_names)} missing scene data names. "
            f"First missing: {preview}"
        )
    return data_list, missing_scene_names


def reorder_object_items_from_split(object_items, object_split, strict=True):
    item_by_name = {item["data_name"]: item for item in object_items}
    ordered_items = []
    missing_names = []
    for object_name in object_split.get("object_names", []):
        item = item_by_name.get(object_name)
        if item is None:
            missing_names.append(object_name)
            continue
        ordered_items.append(item)

    if missing_names and strict:
        preview = ", ".join(missing_names[:5])
        raise ValueError(
            f"Object split references {len(missing_names)} missing object names. "
            f"First missing: {preview}"
        )
    return ordered_items, missing_names
