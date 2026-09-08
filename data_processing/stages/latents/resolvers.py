from __future__ import annotations

import json
import os
import pickle
import re
import sys
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
DEFAULT_OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
DEFAULT_POSTPROCESS_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_POSTPROCESS_ROOT", DEFAULT_OBJECT_ROOT / "postprocess")
)
DEFAULT_SPLIT_JSON = Path(
    os.environ.get(
        "FIRE3D_SCENE_SPLIT",
        REPO_ROOT / "data/training_manifests/scene_train.json",
    )
)
DEFAULT_INTERNSCENES_ROTATION_JSON = Path(
    os.environ.get(
        "FIRE3D_INTERNSCENES_ROTATIONS",
        REPO_ROOT / "data/training_manifests/internscenes_object_rotations.json",
    )
)
DEFAULT_SHAPE_OBJECT_LATENT_ROOT = Path(
    os.environ.get(
        "FIRE3D_SHAPE_LATENT_ROOT", DEFAULT_OBJECT_ROOT / "shape_latents_rotated"
    )
)
DEFAULT_PBR_OBJECT_LATENT_ROOT = Path(
    os.environ.get(
        "FIRE3D_PBR_LATENT_ROOT", DEFAULT_OBJECT_ROOT / "pbr_latents_rotated"
    )
)
SHAPE_OBJECT_LATENT_KEY = "trellis2_shape_encoding"
PBR_OBJECT_LATENT_KEY = "trellis2_pbr_encoding"

DATASET_FOLDERS = {
    "InternScenes": "InternScenes",
    "MansionWorld": "MansionWorld",
    "ProcTHOR": "ProcTHOR",
    "SAGE-10k": "SAGE-10k",
    "SceneSmith": "Scenesmith",
}

DATASET_ALIASES = {
    "internscenes": "InternScenes",
    "intern": "InternScenes",
    "mansionworld": "MansionWorld",
    "mansion": "MansionWorld",
    "procthor": "ProcTHOR",
    "sage10k": "SAGE-10k",
    "sage-10k": "SAGE-10k",
    "sage": "SAGE-10k",
    "scenesmith": "SceneSmith",
}


@dataclass(frozen=True)
class PathConfig:
    training_root: Path = DEFAULT_TRAINING_ROOT
    postprocess_root: Path = DEFAULT_POSTPROCESS_ROOT
    split_json: Path = DEFAULT_SPLIT_JSON
    shape_object_latent_root: Path = DEFAULT_SHAPE_OBJECT_LATENT_ROOT
    pbr_object_latent_root: Path = DEFAULT_PBR_OBJECT_LATENT_ROOT
    object_cache_rotation: str = "000"
    internscenes_rotation_json: Path | None = DEFAULT_INTERNSCENES_ROTATION_JSON
    scene_bg_only: bool = False
    strict: bool = False


@dataclass(frozen=True)
class LatentItem:
    dataset_name: str
    dataset_folder: str
    scene_id: str
    local_name: str
    kind: str
    shape_vxz: Path
    pbr_vxz: Path
    shape_output: Path
    pbr_output: Path
    source_dataset: str | None = None
    source_key: str | None = None
    shape_cache: Path | None = None
    pbr_cache: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_object(self) -> bool:
        return self.kind == "object"

    @property
    def is_scene_bg(self) -> bool:
        return self.kind == "scene_bg"

    def source_npz(self, latent_kind: str) -> Path:
        if latent_kind == "shape":
            return self.shape_cache if self.shape_cache is not None else self.shape_output
        if latent_kind == "pbr":
            return self.pbr_cache if self.pbr_cache is not None else self.pbr_output
        raise ValueError(f"unknown latent kind: {latent_kind}")

    def output_npz(self, latent_kind: str) -> Path:
        if latent_kind == "shape":
            return self.shape_output
        if latent_kind == "pbr":
            return self.pbr_output
        raise ValueError(f"unknown latent kind: {latent_kind}")

    def vxz_path(self, latent_kind: str) -> Path:
        if latent_kind == "shape":
            return self.shape_vxz
        if latent_kind == "pbr":
            return self.pbr_vxz
        raise ValueError(f"unknown latent kind: {latent_kind}")

    def to_record(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_folder": self.dataset_folder,
            "scene_id": self.scene_id,
            "local_name": self.local_name,
            "kind": self.kind,
            "source_dataset": self.source_dataset,
            "source_key": self.source_key,
            "shape_vxz": str(self.shape_vxz),
            "pbr_vxz": str(self.pbr_vxz),
            "shape_cache": str(self.shape_cache) if self.shape_cache else None,
            "pbr_cache": str(self.pbr_cache) if self.pbr_cache else None,
            "shape_output": str(self.shape_output),
            "pbr_output": str(self.pbr_output),
            "metadata": self.metadata,
        }


@dataclass
class ManifestResult:
    items: list[LatentItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def extend(self, other: "ManifestResult") -> None:
        self.items.extend(other.items)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def summary(self) -> dict[str, Any]:
        by_dataset: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        object_sources: dict[str, set[str]] = {}
        scene_count = len({(item.dataset_name, item.scene_id) for item in self.items})
        for item in self.items:
            by_dataset[item.dataset_name] = by_dataset.get(item.dataset_name, 0) + 1
            by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
            if item.source_dataset and item.source_key:
                object_sources.setdefault(item.source_dataset, set()).add(item.source_key)
        return {
            "items": len(self.items),
            "scenes": scene_count,
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_kind": dict(sorted(by_kind.items())),
            "object_cache_counts": {k: len(v) for k, v in sorted(object_sources.items())},
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


def normalize_dataset_name(name: str) -> str:
    if name in DATASET_FOLDERS:
        return name
    key = name.lower().replace("_", "").replace(" ", "")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(f"unknown dataset name: {name}")


def safe_name(value: str) -> str:
    cleaned = str(value).replace("/", "_").replace("|", "_").replace(" ", "_")
    return quote(cleaned, safe="")


def read_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


_INTERNSCENES_ROTATION_CACHE: dict[str, dict[str, Any]] = {}


def internscenes_rotation_overrides(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    key = str(path.resolve())
    if key not in _INTERNSCENES_ROTATION_CACHE:
        data = read_json(path)
        rotations = data.get("rotations", data) if isinstance(data, dict) else {}
        if isinstance(rotations, dict) and "InternScenes" in rotations:
            rotations = rotations["InternScenes"]
        _INTERNSCENES_ROTATION_CACHE[key] = rotations if isinstance(rotations, dict) else {}
    return _INTERNSCENES_ROTATION_CACHE[key]


def internscenes_cache_rotation(
    config: PathConfig,
    scene_stem: str,
    local_name: str,
    source_dataset: str | None = None,
) -> str:
    scene_overrides = internscenes_rotation_overrides(config.internscenes_rotation_json).get(scene_stem, {})
    override = scene_overrides.get(local_name) if isinstance(scene_overrides, dict) else None
    if isinstance(override, str):
        return f"{int(override) % 360:03d}"
    if isinstance(override, dict):
        value = override.get("rotation") or override.get("cache_rotation")
        if value is not None:
            return f"{int(value) % 360:03d}"

    # InternScenes scene-canonical objects include an extra local +90deg Z step.
    # For assets cached from the generic InternScenes source pool, that means the
    # source cache must use rot090.  HSSD and 3D-FUTURE are different: their
    # external object loaders already apply the same net convention as
    # InternScenes init rotation plus that canonical +90deg Z, so rot000 is the
    # correct default unless a measured per-instance override says otherwise.
    if source_dataset in {"HSSD", "3D-FUTURE"}:
        return "000"
    return "090"


def read_split_records(
    split_json: Path,
    datasets: Iterable[str] | None = None,
    scene_ids: set[str] | None = None,
    limit_scenes: int | None = None,
) -> list[dict[str, Any]]:
    dataset_filter = None
    if datasets is not None:
        dataset_filter = {normalize_dataset_name(name) for name in datasets}

    data = read_json(split_json)
    records = data.get("scenes") or data.get("entries")
    if not isinstance(records, list):
        raise ValueError(f"split does not contain a scenes/entries list: {split_json}")

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        dataset_name = normalize_dataset_name(record.get("dataset_name") or record.get("dataset"))
        scene_id = record["scene_id"]
        key = (dataset_name, scene_id)
        if key in seen:
            continue
        seen.add(key)
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        item = dict(record)
        item["dataset_name"] = dataset_name
        selected.append(item)
        if limit_scenes is not None and len(selected) >= limit_scenes:
            break
    return selected


def group_records_by_dataset(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(normalize_dataset_name(record["dataset_name"]), []).append(record)
    return grouped


def scene_output_paths(config: PathConfig, dataset_name: str, scene_id: str, local_name: str) -> tuple[Path, Path]:
    folder = DATASET_FOLDERS[dataset_name]
    root = config.training_root / folder
    stem = safe_name(local_name)
    shape_output = root / "trellis2_shape_latents" / scene_id / f"{stem}.npz"
    pbr_output = root / "trellis2_pbr_latents" / scene_id / f"{stem}.npz"
    return shape_output, pbr_output


def object_cache_npz(
    config: PathConfig,
    latent_kind: str,
    source_dataset: str,
    source_key: str,
    rotation: str | None = None,
) -> Path:
    source_key = safe_name(source_key)
    rotation = f"{int(rotation if rotation is not None else config.object_cache_rotation) % 360:03d}"
    if latent_kind == "shape":
        return (
            config.shape_object_latent_root
            / source_dataset
            / "latents"
            / SHAPE_OBJECT_LATENT_KEY
            / f"{source_key}__rot{rotation}.npz"
        )
    if latent_kind == "pbr":
        return (
            config.pbr_object_latent_root
            / source_dataset
            / "latents"
            / PBR_OBJECT_LATENT_KEY
            / f"{source_key}__rot{rotation}.npz"
        )
    raise ValueError(f"unknown latent kind: {latent_kind}")


def make_bg_item(
    config: PathConfig,
    dataset_name: str,
    scene_id: str,
    local_name: str,
    bg_stem: str,
    metadata: dict[str, Any] | None = None,
) -> LatentItem:
    folder = DATASET_FOLDERS[dataset_name]
    root = config.training_root / folder
    shape_output, pbr_output = scene_output_paths(config, dataset_name, scene_id, local_name)
    return LatentItem(
        dataset_name=dataset_name,
        dataset_folder=folder,
        scene_id=scene_id,
        local_name=local_name,
        kind="scene_bg",
        shape_vxz=root / "ovoxels/scene_bg/shape" / f"{bg_stem}.vxz",
        pbr_vxz=root / "ovoxels/scene_bg/pbr" / f"{bg_stem}.vxz",
        shape_output=shape_output,
        pbr_output=pbr_output,
        metadata=metadata or {},
    )


def make_object_item(
    config: PathConfig,
    dataset_name: str,
    scene_id: str,
    local_name: str,
    source_dataset: str,
    source_key: str,
    metadata: dict[str, Any] | None = None,
    cache_rotation: str | None = None,
) -> LatentItem:
    source_key = safe_name(source_key)
    shape_output, pbr_output = scene_output_paths(config, dataset_name, scene_id, local_name)
    return LatentItem(
        dataset_name=dataset_name,
        dataset_folder=DATASET_FOLDERS[dataset_name],
        scene_id=scene_id,
        local_name=local_name,
        kind="object",
        source_dataset=source_dataset,
        source_key=source_key,
        shape_vxz=config.postprocess_root / "shape_ovoxels" / source_dataset / f"{source_key}.vxz",
        pbr_vxz=config.postprocess_root / "pbr_ovoxels" / source_dataset / f"{source_key}.vxz",
        shape_cache=object_cache_npz(config, "shape", source_dataset, source_key, rotation=cache_rotation),
        pbr_cache=object_cache_npz(config, "pbr", source_dataset, source_key, rotation=cache_rotation),
        shape_output=shape_output,
        pbr_output=pbr_output,
        metadata=metadata or {},
    )


def is_layout_entry(local_name: str, entry: dict[str, Any]) -> bool:
    latent = str(entry.get("latent", ""))
    return local_name.startswith("layout_") or latent.startswith("layout_") or entry.get("asset_id") is None and "layout" in local_name


def mansion_bg_stem(scene_id: str) -> str:
    match = re.match(r"^(?P<prefix>.+)#(?P<frag>[^_]+)_floor_(?P<floor>[^_]+)_room_(?P<room>.+)$", scene_id)
    if match:
        raw = (
            f"{match.group('prefix')}#{match.group('frag')}"
            f"__floor_{match.group('floor')}__room_{match.group('room')}"
        )
        return quote(raw.replace(" ", "_").replace("|", "_"), safe="")
    raw = scene_id.replace("_floor_", "__floor_", 1).replace("_room_", "__room_", 1)
    return quote(raw.replace(" ", "_").replace("|", "_"), safe="")


def procthor_bg_stem(scene_id: str) -> str:
    match = re.match(r"^ProcTHOR_(?P<split>[^_]+)_(?P<house>.+)_room_(?P<room>.+)$", scene_id)
    if not match:
        return quote(scene_id.replace("/", "__").replace(" ", "_").replace("|", "_"), safe="")
    return f"ProcTHOR__{match.group('split')}__{match.group('house')}__room_{match.group('room')}"


def procthor_scene_name_and_room(scene_id: str) -> tuple[str, str]:
    match = re.match(r"^ProcTHOR_(?P<split>[^_]+)_(?P<house>.+)_room_(?P<room>.+)$", scene_id)
    if not match:
        raise ValueError(f"Unexpected ProcTHOR scene_id: {scene_id}")
    return f"ProcTHOR/{match.group('split')}/{match.group('house')}", match.group("room")


def _voxelize_v2_dir() -> Path:
    return REPO_ROOT / "data_processing/stages/ovoxel"


def _procthor_args(training_root: Path) -> Any:
    dependency_root = Path(
        os.environ.get("FIRE3D_DEPENDENCY_ROOT", REPO_ROOT / "data/dependencies")
    )
    return type(
        "ProcTHORResolverArgs",
        (),
        {
            "procthor_root": training_root / "ProcTHOR",
            "ai2thor_hab_root": training_root / "ProcTHOR" / "ai2thor-hab",
            "ai2thor_status": dependency_root / "hf_download_procthor.status",
            "wait_for_deps": False,
            "poll_seconds": 0,
            "blender_path": Path(
                os.environ.get(
                    "FIRE3D_BLENDER",
                    REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender",
                )
            ),
        },
    )()


@lru_cache(maxsize=2048)
def _procthor_room_source_names_by_room(training_root: str, scene_name: str) -> dict[str, tuple[str, ...]]:
    voxelize_dir = _voxelize_v2_dir()
    voxelize_dir_str = str(voxelize_dir)
    if voxelize_dir_str not in sys.path:
        sys.path.insert(0, voxelize_dir_str)
    from procthor_utils import configure_procthor_import  # noqa: PLC0415

    args = _procthor_args(Path(training_root))
    procthor_utils = configure_procthor_import(args)
    _room_regions, room_geoms_dict = procthor_utils.export_rooms_with_canonical_meshes(scene_name)
    return {
        str(room_key): tuple(sorted(str(name) for name in room_geoms if str(name) != "bg"))
        for room_key, room_geoms in room_geoms_dict.items()
    }


@lru_cache(maxsize=8192)
def procthor_room_source_names(training_root: str, scene_name: str, room_id: str) -> tuple[str, ...]:
    """Return ProcTHOR source object names for a room without transform sidecars.

    The transform PKL owns scene-local latent ids and transforms. This helper
    only resolves which global ProcTHOR object cache each `object_000N` should
    materialize from, by replaying the raw ProcTHOR room export that originally
    assigned objects to rooms.
    """

    source_names_by_room = _procthor_room_source_names_by_room(training_root, scene_name)
    room_key = str(int(room_id)) if room_id.isdigit() and str(int(room_id)) in source_names_by_room else str(room_id)
    if room_key not in source_names_by_room:
        raise KeyError(f"ProcTHOR room {room_id!r} not found in {scene_name}")
    return source_names_by_room[room_key]


def intern_native_scene(scene_id: str, scene_name: str | None = None) -> str:
    if scene_name and scene_name.startswith("InternScenes_"):
        raw = scene_name[len("InternScenes_") :]
    else:
        raw = scene_id
    if "/" in raw:
        return raw
    if raw.startswith("scannet_"):
        return raw.replace("scannet_", "scannet/", 1)
    if raw.startswith("3rscan_"):
        return raw.replace("3rscan_", "3rscan/", 1)
    if raw.startswith("arkitscenes_Training_"):
        return "arkitscenes/Training/" + raw[len("arkitscenes_Training_") :]
    if raw.startswith("arkitscenes_Validation_"):
        return "arkitscenes/Validation/" + raw[len("arkitscenes_Validation_") :]
    if raw.startswith("matterport3d_"):
        parts = raw.split("_")
        if len(parts) >= 3 and parts[-1].startswith("region"):
            return f"matterport3d/{parts[1]}/{parts[-1]}"
    if "_" in raw:
        head, tail = raw.split("_", 1)
        return f"{head}/{tail}"
    return raw


def intern_scene_stem(scene_id: str, scene_name: str | None = None) -> str:
    return intern_native_scene(scene_id, scene_name).replace("/", "_")


def intern_source(uid: str) -> tuple[str, str]:
    if uid.startswith("3D-FUTURE-model/"):
        return "3D-FUTURE", uid.rsplit("/", 1)[-1]
    if uid.startswith("hssd-models/"):
        return "HSSD", uid.rsplit("/", 1)[-1]
    return "internscenes", uid.replace("/", "_")


def procthor_source_candidates(record: dict[str, Any]) -> list[str]:
    source = record.get("source_name") or record.get("asset_id") or record.get("latent") or record.get("name")
    source = str(source).rsplit("/", 1)[-1]
    source = re.sub(r"^object_\d+_objects_", "", source)
    candidates = [source]
    if source.endswith("_open"):
        candidates.append(source[:-5])
    return list(dict.fromkeys(candidates))


def choose_existing_object_source(
    config: PathConfig,
    source_dataset_candidates: Iterable[str],
    source_key_candidates: Iterable[str],
) -> tuple[str, str]:
    first_dataset = None
    first_key = None
    for source_dataset in source_dataset_candidates:
        for source_key in source_key_candidates:
            key = safe_name(source_key)
            if first_dataset is None:
                first_dataset, first_key = source_dataset, source_key
            shape = config.postprocess_root / "shape_ovoxels" / source_dataset / f"{key}.vxz"
            pbr = config.postprocess_root / "pbr_ovoxels" / source_dataset / f"{key}.vxz"
            if shape.exists() and pbr.exists():
                return source_dataset, source_key
    if first_dataset is None or first_key is None:
        raise ValueError("empty object source candidates")
    return first_dataset, first_key


def build_sage10k_items(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    root = config.training_root / "SAGE-10k"
    for record in records:
        scene_id = record["scene_id"]
        pkl_path = root / "transforms" / f"{scene_id}.pkl"
        if not pkl_path.exists():
            result.errors.append(f"SAGE-10k missing transform pkl: {pkl_path}")
            continue
        transforms = read_pickle(pkl_path)
        for local_name, entry in transforms.items():
            if is_layout_entry(local_name, entry):
                latent = str(entry.get("latent", local_name))
                result.items.append(make_bg_item(config, "SAGE-10k", scene_id, local_name, scene_id, {"latent": latent}))
            elif not config.scene_bg_only:
                source_key = f"{scene_id}__{entry['latent']}"
                result.items.append(
                    make_object_item(
                        config,
                        "SAGE-10k",
                        scene_id,
                        local_name,
                        "sage10k",
                        source_key,
                        {"latent": entry.get("latent")},
                    )
                )
    return result


def build_scenesmith_items(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    root = config.training_root / "Scenesmith"
    for record in records:
        scene_id = record["scene_id"]
        pkl_path = root / "transforms" / f"{scene_id}.pkl"
        if not pkl_path.exists():
            result.errors.append(f"SceneSmith missing transform pkl: {pkl_path}")
            continue
        transforms = read_pickle(pkl_path)
        for local_name, entry in transforms.items():
            if is_layout_entry(local_name, entry):
                latent = str(entry.get("latent", local_name))
                result.items.append(make_bg_item(config, "SceneSmith", scene_id, local_name, latent, {"latent": latent}))
            elif not config.scene_bg_only:
                source_key = f"{scene_id}__{local_name}"
                result.items.append(
                    make_object_item(
                        config,
                        "SceneSmith",
                        scene_id,
                        local_name,
                        "scenesmith",
                        source_key,
                        {"latent": entry.get("latent"), "mesh_name": entry.get("mesh_name")},
                    )
                )
    return result


def build_mansionworld_items(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    root = config.training_root / "MansionWorld"
    for record in records:
        scene_id = record["scene_id"]
        pkl_path = root / "transforms" / f"{scene_id}.pkl"
        if not pkl_path.exists():
            result.errors.append(f"MansionWorld missing transform pkl: {pkl_path}")
            continue
        transforms = read_pickle(pkl_path)
        for local_name, entry in transforms.items():
            if is_layout_entry(local_name, entry):
                latent = str(entry.get("latent", local_name))
                result.items.append(
                    make_bg_item(config, "MansionWorld", scene_id, local_name, mansion_bg_stem(scene_id), {"latent": latent})
                )
                continue
            if config.scene_bg_only:
                continue
            asset_id = entry.get("asset_id")
            if not asset_id:
                result.errors.append(f"MansionWorld missing asset_id for {scene_id}/{local_name}")
                continue
            source_dataset, source_key = choose_existing_object_source(config, ["objathor", "procthor"], [str(asset_id)])
            result.items.append(
                make_object_item(
                    config,
                    "MansionWorld",
                    scene_id,
                    local_name,
                    source_dataset,
                    source_key,
                    {"asset_id": asset_id, "latent": entry.get("latent")},
                )
            )
    return result


def build_procthor_items(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    root = config.training_root / "ProcTHOR"
    for record in records:
        scene_id = record["scene_id"]
        pkl_path = root / "transforms" / f"{scene_id}.pkl"
        if not pkl_path.exists():
            result.errors.append(f"ProcTHOR missing transform pkl: {pkl_path}")
            continue
        transforms = read_pickle(pkl_path)
        object_keys = [key for key, value in transforms.items() if not is_layout_entry(key, value)]
        for local_name, entry in transforms.items():
            if is_layout_entry(local_name, entry):
                latent = str(entry.get("latent", local_name))
                result.items.append(
                    make_bg_item(config, "ProcTHOR", scene_id, local_name, procthor_bg_stem(scene_id), {"latent": latent})
                )
        if config.scene_bg_only:
            continue
        try:
            scene_name, room_id = procthor_scene_name_and_room(scene_id)
            source_names = procthor_room_source_names(str(config.training_root), scene_name, room_id)
        except Exception as exc:  # noqa: BLE001 - manifest should report all unresolved scenes.
            result.errors.append(f"ProcTHOR source mapping failed for {scene_id}: {exc!r}")
            continue
        if len(object_keys) != len(source_names):
            result.errors.append(
                f"ProcTHOR object count mismatch for {scene_id}: pkl={len(object_keys)} raw_export={len(source_names)}"
            )
            continue
        for local_name, source_name in zip(object_keys, source_names, strict=True):
            candidates = procthor_source_candidates({"source_name": source_name})
            source_dataset, source_key = choose_existing_object_source(config, ["procthor"], candidates)
            result.items.append(
                make_object_item(
                    config,
                    "ProcTHOR",
                    scene_id,
                    local_name,
                    source_dataset,
                    source_key,
                    {"source_name": source_name, "source_mapping": "raw_procthor_export"},
                )
            )
    return result


def build_internscenes_items(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    root = config.training_root / "InternScenes"
    for record in records:
        scene_id = record["scene_id"]
        scene_name = record.get("scene_name")
        scene_stem = intern_scene_stem(scene_id, scene_name)
        native_scene = intern_native_scene(scene_id, scene_name)
        pkl_path = root / "transforms" / f"{scene_stem}.pkl"
        layout_path = root / "downloaded/Layout_info" / native_scene / "layout.json"
        if not pkl_path.exists():
            result.errors.append(f"InternScenes missing transform pkl: {pkl_path}")
            continue
        if not layout_path.exists():
            result.errors.append(f"InternScenes missing layout json: {layout_path}")
            continue
        transforms = read_pickle(pkl_path)
        layout = read_json(layout_path)
        result.items.append(
            make_bg_item(config, "InternScenes", scene_stem, f"layout_{scene_stem}_bg", scene_stem, {"native_scene": native_scene})
        )
        for local_name, entry in transforms.items():
            if is_layout_entry(local_name, entry):
                continue
            try:
                layout_index = int(local_name)
            except ValueError:
                result.errors.append(f"InternScenes non-numeric object key for {scene_stem}: {local_name}")
                continue
            if layout_index < 0 or layout_index >= len(layout):
                result.errors.append(f"InternScenes layout index out of range for {scene_stem}: {local_name}")
                continue
            uid = layout[layout_index].get("model_uid")
            if not uid:
                result.errors.append(f"InternScenes missing model_uid for {scene_stem}/{local_name}")
                continue
            source_dataset, source_key = intern_source(uid)
            cache_rotation = internscenes_cache_rotation(config, scene_stem, local_name, source_dataset)
            result.items.append(
                make_object_item(
                    config,
                    "InternScenes",
                    scene_stem,
                    local_name,
                    source_dataset,
                    source_key,
                    cache_rotation=cache_rotation,
                    metadata={
                        "model_uid": uid,
                        "native_scene": native_scene,
                        "cache_rotation": cache_rotation,
                    },
                )
            )
    return result


BUILDERS = {
    "InternScenes": build_internscenes_items,
    "MansionWorld": build_mansionworld_items,
    "ProcTHOR": build_procthor_items,
    "SAGE-10k": build_sage10k_items,
    "SceneSmith": build_scenesmith_items,
}


def build_manifest(records: list[dict[str, Any]], config: PathConfig) -> ManifestResult:
    result = ManifestResult()
    for dataset_name, dataset_records in group_records_by_dataset(records).items():
        result.extend(BUILDERS[dataset_name](dataset_records, config))
    return result


def validate_source_paths(items: Iterable[LatentItem], limit: int | None = None) -> list[str]:
    errors: list[str] = []
    exists_cache: dict[Path, bool] = {}

    def cached_exists(path: Path) -> bool:
        if path not in exists_cache:
            exists_cache[path] = path.exists()
        return exists_cache[path]

    for item in items:
        if not cached_exists(item.shape_vxz):
            errors.append(f"missing shape vxz: {item.dataset_name}/{item.scene_id}/{item.local_name} -> {item.shape_vxz}")
        if not cached_exists(item.pbr_vxz):
            errors.append(f"missing pbr vxz: {item.dataset_name}/{item.scene_id}/{item.local_name} -> {item.pbr_vxz}")
        if limit is not None and len(errors) >= limit:
            break
    return errors


T = TypeVar("T")


def shard_by_index(values: list[T], rank: int, world_size: int) -> list[T]:
    if world_size <= 1:
        return values
    return [value for index, value in enumerate(values) if index % world_size == rank]


def unique_items_by_target(items: Iterable[LatentItem], latent_kind: str, use_cache: bool) -> list[LatentItem]:
    seen: set[Path] = set()
    result: list[LatentItem] = []
    for item in items:
        target = item.source_npz(latent_kind) if use_cache and item.is_object else item.output_npz(latent_kind)
        if target in seen:
            continue
        seen.add(target)
        result.append(item)
    return result


def write_manifest_jsonl(items: Iterable[LatentItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_record(), sort_keys=True) + "\n")
