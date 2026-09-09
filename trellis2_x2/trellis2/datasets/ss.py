"""Dense occupancy supports used to train the sparse-structure VAE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


QUARTER_YAWS = (0, 90, 180, 270)


def _normalize_roots(
    roots: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    values = str(roots).split(",") if isinstance(roots, (str, Path)) else roots
    normalized = tuple(
        Path(value).expanduser() for value in values if str(value).strip()
    )
    if not normalized:
        raise ValueError("at least one shape-latent root is required")
    return normalized


def rotate_quarter_yaw(
    coords: np.ndarray,
    degrees: int,
    resolution: int,
) -> np.ndarray:
    """Rotate integer voxel coordinates around +Z without interpolation."""

    degrees = int(degrees) % 360
    if degrees not in QUARTER_YAWS:
        raise ValueError(f"rotation must be one of {QUARTER_YAWS}, got {degrees}")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    if degrees == 0:
        return coords.copy()
    if degrees == 90:
        return np.stack((resolution - 1 - y, x, z), axis=1)
    if degrees == 180:
        return np.stack((resolution - 1 - x, resolution - 1 - y, z), axis=1)
    return np.stack((y, resolution - 1 - x, z), axis=1)


class SparseStructure(Dataset):
    """Load binary occupancy grids from shape HC-VAE latent coordinates.

    Every root is a directory of object NPZ files emitted by
    ``data_processing/stages/latents/shape_enc_seq.py``. Each NPZ must contain
    an integer ``coords`` array with shape ``[N, 3]`` in the configured grid.
    Quarter-yaw rotations expand the deterministic dataset index.
    """

    value_range = (0.0, 1.0)

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        *,
        resolution: int = 8,
        rotations: Iterable[int] = QUARTER_YAWS,
    ) -> None:
        self.roots = _normalize_roots(roots)
        self.resolution = int(resolution)
        if self.resolution < 1:
            raise ValueError("resolution must be positive")

        self.rotations = tuple(int(rotation) % 360 for rotation in rotations)
        if not self.rotations:
            raise ValueError("at least one rotation is required")
        invalid = sorted(set(self.rotations) - set(QUARTER_YAWS))
        if invalid:
            raise ValueError(f"unsupported rotations: {invalid}")
        if len(set(self.rotations)) != len(self.rotations):
            raise ValueError(f"rotations must be unique, got {self.rotations}")

        paths: list[Path] = []
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(f"shape-latent root does not exist: {root}")
            source_paths = sorted(root.glob("*.npz"))
            if not source_paths:
                raise ValueError(f"no NPZ files found in shape-latent root: {root}")
            paths.extend(source_paths)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate shape-latent paths were supplied across roots")
        self.paths = tuple(paths)

    @property
    def num_objects(self) -> int:
        return len(self.paths)

    def __len__(self) -> int:
        return self.num_objects * len(self.rotations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        object_index, rotation_index = divmod(index, len(self.rotations))
        path = self.paths[object_index]
        with np.load(path, allow_pickle=False) as archive:
            if "coords" not in archive:
                raise ValueError(f"missing coords array: {path}")
            coords = np.asarray(archive["coords"], dtype=np.int64)

        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
            raise ValueError(f"expected nonempty [N, 3] coords, got {coords.shape}: {path}")
        if coords.min() < 0 or coords.max() >= self.resolution:
            raise ValueError(
                f"coords outside [0, {self.resolution - 1}] "
                f"(min={coords.min()}, max={coords.max()}): {path}"
            )

        coords = rotate_quarter_yaw(
            coords,
            self.rotations[rotation_index],
            self.resolution,
        )
        occupancy = torch.zeros(
            (1, self.resolution, self.resolution, self.resolution),
            dtype=torch.float32,
        )
        occupancy[0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        return {"ss": occupancy}

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(objects={self.num_objects:,}, "
            f"rotations={self.rotations}, samples={len(self):,}, "
            f"resolution={self.resolution})"
        )
