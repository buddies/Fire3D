#!/usr/bin/env python3
"""Install stable release metadata and HC-VAE encoders in a model staging tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = {
    "config.json": REPO_ROOT / "docs/huggingface_model_config.json",
    "README.md": REPO_ROOT / "docs/huggingface_model_card.md",
}
ENCODER_PATHS = {
    "shape": "reconstruction/vae/shape/ckpts/encoder.pt",
    "pbr": "reconstruction/vae/pbr/ckpts/encoder.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--shape-encoder", type=Path, required=True)
    parser.add_argument("--pbr-encoder", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        help="Code revision recorded in manifest.json; defaults to the current Git HEAD.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_changed(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == source.stat().st_size
        and sha256(destination) == sha256(source)
    ):
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def current_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    staging = args.staging.expanduser().resolve()
    manifest_path = staging / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")

    sources = {
        **PUBLIC_FILES,
        ENCODER_PATHS["shape"]: args.shape_encoder.expanduser().resolve(),
        ENCODER_PATHS["pbr"]: args.pbr_encoder.expanduser().resolve(),
    }
    for relative, source in sources.items():
        copy_if_changed(source, staging / relative)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fire3d_model_bundle_v1":
        raise ValueError("Unsupported model manifest schema")
    manifest["source_commit"] = args.source_commit or current_revision()
    records = {record["path"]: record for record in manifest.get("files", [])}
    for relative in ("config.json", *ENCODER_PATHS.values()):
        path = staging / relative
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest["files"] = [records[path] for path in sorted(records)]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = "\n".join(
        f"{record['sha256']}  {record['path']}" for record in manifest["files"]
    )
    (staging / "checksums.sha256").write_text(checksums + "\n", encoding="utf-8")
    print(f"Prepared {len(manifest['files'])} verified files under {staging}")


if __name__ == "__main__":
    main()
