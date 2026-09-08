import json
import re
from pathlib import Path

import numpy as np

from data_processing.fetch import parse_args as parse_processing_fetch_args
from data_processing.objects import load_adapter
from data_processing.stages.latents.checkpoints import resolve_vae_checkpoints


ROOT = Path(__file__).resolve().parents[1]
OBJECT_DATASETS = (
    "3d_future",
    "abo",
    "hssd",
    "objaverse_github",
    "objaverse_sketchfab",
)


def test_object_adapters_expose_complete_contract():
    for name in OBJECT_DATASETS:
        display_name, adapter = load_adapter(name)
        assert display_name
        for symbol in (
            "list_all_model_paths",
            "build_metadata_mapping",
            "save_metadata_mapping",
            "load_model",
        ):
            assert callable(getattr(adapter, symbol))


def test_object_adapter_discovers_and_normalizes_mesh(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    mesh_path = source / "raw_model.obj"
    mesh_path.write_text(
        "v 0 0 0\nv 2 0 0\nv 0 1 0\nv 0 0 0.5\n"
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
        encoding="utf-8",
    )
    _, adapter = load_adapter("3d_future")
    cache = tmp_path / "index.json"
    assert adapter.list_all_model_paths(source, cache, refresh=True) == [
        mesh_path.resolve().as_posix()
    ]
    scene = adapter.load_model(mesh_path)
    bounds = np.asarray(scene.bounds)
    np.testing.assert_allclose(bounds.mean(axis=0), np.zeros(3), atol=1e-7)
    assert np.isclose((bounds[1] - bounds[0]).max(), 0.99999)


def test_vae_checkpoint_resolver_prefers_stable_public_aliases(tmp_path):
    root = tmp_path / "vae"
    ckpts = root / "ckpts"
    ckpts.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"trainer": {"args": {"ema_rate": 0.9999}}}),
        encoding="utf-8",
    )
    for name in ("encoder.pt", "decoder.pt"):
        (ckpts / name).write_bytes(name.encode())
    _, encoder, decoder, label = resolve_vae_checkpoints(root)
    assert encoder.name == "encoder.pt"
    assert decoder.name == "decoder.pt"
    assert label == "stable release"


def test_vae_checkpoint_resolver_retains_legacy_training_compatibility(tmp_path):
    root = tmp_path / "vae"
    ckpts = root / "ckpts"
    ckpts.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"trainer": {"args": {"ema_rate": 0.9999}}}),
        encoding="utf-8",
    )
    for value in (10, 20):
        for kind in ("encoder", "decoder"):
            (ckpts / f"{kind}_ema0.9999_step{value:07d}.pt").write_bytes(b"weights")
    _, encoder, decoder, label = resolve_vae_checkpoints(root)
    assert encoder.name.endswith("0000020.pt")
    assert decoder.name.endswith("0000020.pt")
    assert label == "ema training checkpoint 20"


def test_processing_upstream_sources_are_pinned_and_ignored():
    registry = json.loads((ROOT / "data_processing/registry.json").read_text())
    sources = registry["upstream_sources"]
    assert sources
    for record in sources.values():
        assert record["repository"].startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", record["revision"])
        assert record["checkout"].startswith("data_processing/_upstream/")
    assert parse_processing_fetch_args([]).sources == []
    assert "data_processing/_upstream/" in (ROOT / ".gitignore").read_text()
