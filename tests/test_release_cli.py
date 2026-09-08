import hashlib
import json
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

from fire3d import cli
from fire3d import download as release_download


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_command_help_is_runnable():
    for arguments in (
        ["--help"],
        ["download", "--help"],
        ["infer", "--help"],
        ["sample-views", "--help"],
        ["render", "--help"],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "fire3d", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_internal_entrypoint_help_is_runnable():
    for relative in (
        "benchmarks/scene_reconstruction/run_lc64_geometry.py",
        "eval/perception/eval_perception.py",
    ):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / relative), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_infer_command_uses_public_roots_and_batches_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "validate_release_inputs", lambda *args: None)
    args = Namespace(
        dataset="ithor",
        scene_id=["scene_a", "scene_b"],
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
        protocol=None,
        gpu="2",
        skip_render=True,
        skip_existing=False,
    )
    command = cli.infer_command(args)
    assert command.count("--scene-id") == 2
    assert command[command.index("--fire3d-test-root") + 1] == str(
        (tmp_path / "data").resolve()
    )
    assert "--protocol-input-root" not in command
    assert "--skip-render" in command


def test_infer_command_uses_native_scannetpp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "validate_release_inputs", lambda *args: None)
    args = Namespace(
        dataset="scannetpp",
        scene_id=["09bced689e"],
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
        protocol=None,
        gpu="0",
        skip_render=False,
        skip_existing=False,
    )
    command = cli.infer_command(args)
    assert command[command.index("--protocol-input-root") + 1] == str(
        (tmp_path / "data/scannetpp").resolve()
    )


def test_download_rejects_non_whitelisted_scene(tmp_path, monkeypatch):
    archive_root = tmp_path / "data/.fire3d_archives"
    manifest = {
        "schema": "fire3d_inference_data_v1",
        "datasets": {
            "ithor": {
                "common_archives": [],
                "scenes": {"allowed": {"archive": "archives/allowed.tar"}},
            }
        },
        "archives": {},
    }

    def fake_snapshot(**kwargs):
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(release_download, "_snapshot_download", fake_snapshot)
    try:
        release_download.download_data(
            tmp_path / "data", ["ithor"], scene_ids=["not_allowed"]
        )
    except ValueError as error:
        assert "not in the ithor release whitelist" in str(error)
    else:
        raise AssertionError("non-whitelisted scene was accepted")


def test_model_download_verifies_manifest_files(tmp_path, monkeypatch):
    destination = tmp_path / "models"
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    snapshot_calls = []

    def fake_snapshot(**kwargs):
        snapshot_calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        model_path = target / "perception/model.pt"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(payload)
        (target / "config.json").write_text(
            json.dumps({"schema": "fire3d_model_bundle_v1"})
        )
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fire3d_model_bundle_v1",
                    "files": [
                        {
                            "path": "perception/model.pt",
                            "sha256": digest,
                        }
                    ],
                }
            )
        )

    monkeypatch.setattr(release_download, "_snapshot_download", fake_snapshot)
    assert release_download.download_models(destination) == destination.resolve()
    assert len(snapshot_calls) == 1
    assert snapshot_calls[0]["repo_type"] == "model"
    assert "allow_patterns" not in snapshot_calls[0]


def test_evaluation_download_verifies_and_extracts_archive(tmp_path, monkeypatch):
    source = tmp_path / "source"
    archive = source / "evaluation/shaper_gt_v1.tar"
    archive.parent.mkdir(parents=True)
    payload = tmp_path / "sample.npz"
    payload.write_bytes(b"compact-shaper-gt")
    with tarfile.open(archive, "w") as handle:
        handle.add(payload, arcname="evaluation/shaper/gt/sample.npz")
    digest = release_download.sha256_file(archive)
    manifest = {
        "schema": "fire3d_inference_data_v1",
        "datasets": {},
        "evaluations": {
            "shaper": {"archives": ["evaluation/shaper_gt_v1.tar"]}
        },
        "archives": {
            "evaluation/shaper_gt_v1.tar": {"sha256": digest}
        },
    }

    def fake_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps(manifest))
        if "evaluation/shaper_gt_v1.tar" in kwargs["allow_patterns"]:
            output = target / "evaluation/shaper_gt_v1.tar"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read_bytes())

    monkeypatch.setattr(release_download, "_snapshot_download", fake_snapshot)
    destination = tmp_path / "data"
    release_download.download_evaluation_data(destination, ["shaper"])
    assert (destination / "evaluation/shaper/gt/sample.npz").read_bytes() == payload.read_bytes()


def test_download_verifies_and_extracts_archive(tmp_path, monkeypatch):
    source = tmp_path / "source"
    archive = source / "archives/ithor/scenes/allowed.tar"
    archive.parent.mkdir(parents=True)
    payload = tmp_path / "payload.txt"
    payload.write_text("ready\n")
    with tarfile.open(archive, "w") as handle:
        handle.add(payload, arcname="ithor/example/ready.txt")
    digest = release_download.sha256_file(archive)
    manifest = {
        "schema": "fire3d_inference_data_v1",
        "datasets": {
            "ithor": {
                "common_archives": [],
                "scenes": {"allowed": {"archive": "archives/ithor/scenes/allowed.tar"}},
            }
        },
        "archives": {
            "archives/ithor/scenes/allowed.tar": {"sha256": digest}
        },
    }

    def fake_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps(manifest))
        if "archives/ithor/scenes/allowed.tar" in kwargs["allow_patterns"]:
            output = target / "archives/ithor/scenes/allowed.tar"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read_bytes())

    monkeypatch.setattr(release_download, "_snapshot_download", fake_snapshot)
    destination = tmp_path / "data"
    release_download.download_data(destination, ["ithor"], scene_ids=["allowed"])
    assert (destination / "ithor/example/ready.txt").read_text() == "ready\n"
    assert not (destination / ".fire3d_archives").exists()
