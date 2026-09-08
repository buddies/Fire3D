"""Small dependency-free helpers for EFM3D-to-ShapeR comparisons."""

from __future__ import annotations

from collections.abc import Iterable


def is_room_slab_extents(extents: Iterable[float]) -> bool:
    ordered = sorted(float(value) for value in extents)
    if len(ordered) != 3:
        raise ValueError(f"Expected three mesh extents, found {len(ordered)}")
    return ordered[0] <= 0.5 and ordered[1] >= 4.0
