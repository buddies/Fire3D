import hashlib
import json
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_model_release_builder_adds_query_file_and_stable_encoder_aliases(tmp_path):
    staging = tmp_path / "model"
    staging.mkdir()
    existing = staging / "perception/model.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"perception")
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fire3d_model_bundle_v1",
                "source_commit": "old",
                "files": [
                    {
                        "path": "perception/model.pt",
                        "bytes": existing.stat().st_size,
                        "sha256": digest(existing.read_bytes()),
                    }
                ],
            }
        )
    )
    shape_encoder = tmp_path / "shape.pt"
    pbr_encoder = tmp_path / "pbr.pt"
    shape_encoder.write_bytes(b"shape-encoder")
    pbr_encoder.write_bytes(b"pbr-encoder")

    revision = "a" * 40
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_model_release.py"),
            "--staging",
            str(staging),
            "--shape-encoder",
            str(shape_encoder),
            "--pbr-encoder",
            str(pbr_encoder),
            "--source-commit",
            revision,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((staging / "manifest.json").read_text())
    records = {record["path"]: record for record in manifest["files"]}
    assert manifest["source_commit"] == revision
    assert json.loads((staging / "config.json").read_text())["schema"] == (
        "fire3d_model_bundle_v1"
    )
    for relative, payload in {
        "reconstruction/vae/shape/ckpts/encoder.pt": b"shape-encoder",
        "reconstruction/vae/pbr/ckpts/encoder.pt": b"pbr-encoder",
    }.items():
        assert (staging / relative).read_bytes() == payload
        assert records[relative]["sha256"] == digest(payload)
    assert "library_name:" not in (staging / "README.md").read_text().split("---", 2)[1]


def test_shaper_release_builder_emits_pickle_free_bundle(tmp_path):
    source = tmp_path / "shaper-source"
    source.mkdir()
    (source / "LICENSE").write_text("ShapeR evaluation license\n")
    sample = {
        "mesh_vertices": np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        "mesh_faces": np.asarray([[0, 1, 2]], np.int64),
        "bounds": np.asarray([[0, 0, 0], [1, 1, 0]], np.float32),
        "points_model": np.asarray([[0.25, 0.25, 0]], np.float32),
        "T_zup_obj": np.eye(4, dtype=np.float32),
        "T_model_world": np.eye(4, dtype=np.float32),
        "category": "chair",
        "caption": "test chair",
    }
    with (source / "sample.pkl").open("wb") as stream:
        pickle.dump(sample, stream)
    staging = tmp_path / "dataset"
    staging.mkdir()
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fire3d_inference_data_v1",
                "datasets": {},
                "archives": {},
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_shaper_gt_release.py"),
            "--source",
            str(source),
            "--dataset-staging",
            str(staging),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    archive = staging / "evaluation/shaper_gt_v1.tar"
    with tarfile.open(archive) as handle:
        names = set(handle.getnames())
    assert "evaluation/shaper/gt/sample.npz" in names
    assert not any(name.endswith(".pkl") for name in names)
    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["evaluations"]["shaper"]["num_samples"] == 1
    assert manifest["archives"]["evaluation/shaper_gt_v1.tar"]["sha256"] == (
        digest(archive.read_bytes())
    )
