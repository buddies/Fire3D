"""Registry-backed object-dataset adapters used by Fire3D preprocessing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data_processing/registry.json"
ALIASES = {
    "3d-future": "3d_future",
    "3d_future": "3d_future",
    "abo": "abo",
    "hssd": "hssd",
    "objaversexl_github": "objaverse_github",
    "objaverse_github": "objaverse_github",
    "objaversexl_sketchfab": "objaverse_sketchfab",
    "objaverse_sketchfab": "objaverse_sketchfab",
}
DISPLAY_NAMES = {
    "3d_future": "3D-FUTURE",
    "abo": "ABO",
    "hssd": "HSSD",
    "objaverse_github": "ObjaverseXL_github",
    "objaverse_sketchfab": "ObjaverseXL_sketchfab",
}


def normalize_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    try:
        return ALIASES[key]
    except KeyError as error:
        choices = ", ".join(sorted(DISPLAY_NAMES))
        raise ValueError(f"Unsupported object dataset {name!r}; choose from {choices}") from error


def dataset_choices() -> tuple[str, ...]:
    return tuple(sorted(set(ALIASES) | set(DISPLAY_NAMES.values())))


def load_adapter(name: str) -> tuple[str, ModuleType]:
    key = normalize_name(name)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source = ROOT / registry["object_datasets"][key]
    spec = importlib.util.spec_from_file_location(f"fire3d_object_adapter_{key}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load object adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return DISPLAY_NAMES[key], module


__all__ = ["dataset_choices", "load_adapter", "normalize_name"]
