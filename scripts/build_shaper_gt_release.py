#!/usr/bin/env python3
"""Build the compact ShapeR GT archive used by Fire3D evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-staging", type=Path, required=True)
    parser.add_argument("--archive", default="evaluation/shaper_gt_v1.tar")
    return parser.parse_args()


def as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    staging = args.dataset_staging.expanduser().resolve()
    archive = staging / args.archive
    manifest_path = staging / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset release manifest: {manifest_path}")
    samples = sorted(source.glob("*.pkl"))
    if not samples:
        raise FileNotFoundError(f"No ShapeR PKL samples under {source}")

    with tempfile.TemporaryDirectory(prefix="fire3d_shaper_gt_") as temporary:
        package = Path(temporary) / "evaluation/shaper"
        gt_root = package / "gt"
        gt_root.mkdir(parents=True)
        records = []
        for index, path in enumerate(samples, start=1):
            # The official ShapeR evaluation files are trusted inputs. The
            # public derivative is NPZ and never requires pickle loading.
            with path.open("rb") as stream:
                sample = pickle.load(stream)
            required = ("mesh_vertices", "mesh_faces", "bounds")
            missing = [key for key in required if key not in sample]
            if missing:
                raise KeyError(f"{path.name} is missing {missing}")
            output = gt_root / f"{path.stem}.npz"
            np.savez_compressed(
                output,
                mesh_vertices=as_numpy(sample["mesh_vertices"], np.float32),
                mesh_faces=as_numpy(sample["mesh_faces"], np.int32),
                bounds=as_numpy(sample["bounds"], np.float32),
                points_model=as_numpy(sample.get("points_model", np.empty((0, 3))), np.float32),
                T_zup_obj=as_numpy(sample.get("T_zup_obj", np.eye(4)), np.float32),
                T_model_world=as_numpy(sample.get("T_model_world", np.eye(4)), np.float32),
                category=np.asarray(str(sample.get("category", ""))),
                caption=np.asarray(str(sample.get("caption", ""))),
            )
            records.append(
                {
                    "name": path.stem,
                    "path": f"gt/{output.name}",
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                }
            )
            print(f"[{index}/{len(samples)}] {path.stem}", flush=True)

        (package / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fire3d.shaper_gt.v1",
                    "source": "facebook/ShapeR-Evaluation",
                    "samples": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (package / "README.md").write_text(
            "# ShapeR Evaluation Ground Truth\n\n"
            "This compact derivative contains only meshes, bounds, transforms, "
            "and condition points needed by the Fire3D geometry evaluator. It "
            "was derived from `facebook/ShapeR-Evaluation`; the included source "
            "license and attribution continue to apply.\n"
        )
        source_license = source / "LICENSE"
        if not source_license.is_file():
            raise FileNotFoundError(f"Missing ShapeR source license: {source_license}")
        (package / "LICENSE").write_bytes(source_license.read_bytes())

        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w") as tar:
            tar.add(package, arcname="evaluation/shaper")

    manifest = json.loads(manifest_path.read_text())
    relative = archive.relative_to(staging).as_posix()
    manifest.setdefault("archives", {})[relative] = {
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    }
    manifest.setdefault("evaluations", {})["shaper"] = {
        "archives": [relative],
        "num_samples": len(samples),
        "format": "fire3d.shaper_gt.v1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_lines = [
        f"{record['sha256']}  {path}"
        for path, record in sorted(manifest["archives"].items())
    ]
    (staging / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(f"Wrote {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
