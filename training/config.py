"""Portable configuration loading for Fire3D training entry points."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any, *, source: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item, source=source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, source=source) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Unresolved environment variable(s) in {source}: {names}")
    return os.path.expanduser(os.path.expandvars(value))


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load YAML/JSON and recursively resolve ``${ENV_VAR}`` values."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream) if source.suffix == ".json" else yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must contain a mapping: {source}")
    return _expand_environment(payload, source=source)
