"""Shared discovery, metadata, and canonicalization for object datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data_processing.objects.metadata_utils import (
    build_metadata_mapping as build_metadata_records,
)
from data_processing.objects.metadata_utils import save_metadata_mapping


ORIENTATION_MATRIX = (
    (-1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1),
)


def discover_model_paths(
    data_dir: str | Path,
    cache_file: str | Path,
    patterns: Iterable[str],
    *,
    refresh: bool = False,
) -> list[str]:
    data_dir = Path(data_dir).expanduser().resolve()
    cache_file = Path(cache_file).expanduser().resolve()
    if cache_file.is_file() and not refresh:
        records = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(item, str) for item in records):
            raise ValueError(f"Invalid object index: {cache_file}")
        return records
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Object dataset root does not exist: {data_dir}. Set the adapter's "
            "documented FIRE3D_*_ROOT environment variable."
        )
    patterns = tuple(patterns)
    paths = sorted(
        {path.resolve().as_posix() for pattern in patterns for path in data_dir.rglob(pattern)}
    )
    if not paths:
        raise FileNotFoundError(f"No object assets matching {patterns} under {data_dir}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    return paths


def build_metadata(
    *,
    metadata_file: str | Path,
    data_dir: str | Path,
    model_paths: Iterable[str],
    model_filename: str | None = None,
) -> dict:
    metadata_file = Path(metadata_file).expanduser().resolve()
    return build_metadata_records(
        metadata_file=metadata_file,
        dataset_root=metadata_file.parent,
        data_dir=Path(data_dir).expanduser().resolve(),
        model_paths=model_paths,
        model_filename=model_filename,
    )


def load_normalized_model(model_path: str | Path):
    import numpy as np
    import trimesh

    scene = trimesh.load(model_path, force="scene")
    scene.apply_transform(np.asarray(ORIENTATION_MATRIX, dtype=np.float64))
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extent = bounds[1] - bounds[0]
    longest = float(extent.max())
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError(f"Degenerate object bounds for {model_path}: {bounds.tolist()}")
    center = bounds.mean(axis=0)
    center_transform = np.eye(4, dtype=np.float64)
    center_transform[:3, 3] = -center
    scale_transform = np.eye(4, dtype=np.float64)
    scale_transform[np.diag_indices(3)] = 0.99999 / longest
    scene.apply_transform(scale_transform @ center_transform)
    return scene


@dataclass(frozen=True)
class ObjectAdapter:
    name: str
    data_dir: Path
    metadata_file: Path
    cache_file: Path
    metadata_cache_file: Path
    patterns: tuple[str, ...]
    model_filename: str | None = None

    def list_all_model_paths(
        self,
        data_dir: str | Path | None = None,
        cache_file: str | Path | None = None,
        refresh: bool = False,
    ) -> list[str]:
        return discover_model_paths(
            data_dir or self.data_dir,
            cache_file or self.cache_file,
            self.patterns,
            refresh=refresh,
        )

    def build_metadata_mapping(
        self,
        data_dir: str | Path | None = None,
        metadata_file: str | Path | None = None,
    ) -> dict:
        active_data_dir = data_dir or self.data_dir
        return build_metadata(
            metadata_file=metadata_file or self.metadata_file,
            data_dir=active_data_dir,
            model_paths=self.list_all_model_paths(active_data_dir),
            model_filename=self.model_filename,
        )

    def save_metadata_mapping(self, output_file: str | Path | None = None) -> str:
        return save_metadata_mapping(
            self.build_metadata_mapping(), output_file or self.metadata_cache_file
        )

    @staticmethod
    def load_model(model_path: str | Path):
        return load_normalized_model(model_path)

    def run_cli(self) -> None:
        parser = argparse.ArgumentParser(
            description=f"Index and normalize the {self.name} object data."
        )
        parser.add_argument("--data-root", type=Path, default=self.data_dir)
        parser.add_argument("--metadata-file", type=Path, default=self.metadata_file)
        parser.add_argument("--cache-file", type=Path, default=self.cache_file)
        parser.add_argument(
            "--metadata-output", type=Path, default=self.metadata_cache_file
        )
        parser.add_argument("--refresh", action="store_true")
        parser.add_argument("--skip-metadata", action="store_true")
        args = parser.parse_args()
        paths = discover_model_paths(
            args.data_root, args.cache_file, self.patterns, refresh=args.refresh
        )
        print(f"{self.name}: indexed {len(paths)} objects")
        if not args.skip_metadata:
            mapping = build_metadata(
                metadata_file=args.metadata_file,
                data_dir=args.data_root,
                model_paths=paths,
                model_filename=self.model_filename,
            )
            save_metadata_mapping(mapping, args.metadata_output)
            print(f"Metadata: {args.metadata_output}")
