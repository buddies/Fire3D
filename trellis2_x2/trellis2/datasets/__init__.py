"""Datasets required by the released Fire3D HC-VAE recipes."""

from __future__ import annotations

import importlib


_ATTRIBUTES = {
    "SLat": "structured_latent",
    "SLatShape": "structured_latent_shape",
    "SLatPbrOnly": "structured_latent_pbr_only",
}

__all__ = list(_ATTRIBUTES)


def __getattr__(name: str):
    if name not in _ATTRIBUTES:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(f".{_ATTRIBUTES[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
