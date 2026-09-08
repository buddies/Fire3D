from __future__ import annotations

"""Resumable authenticated downloader for facebook/ShapeR-Evaluation."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data/evaluation/shaper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="facebook/ShapeR-Evaluation")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Optional Hugging Face token file; defaults to HF_TOKEN.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-resources", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def api_json(url: str, token: str) -> Any:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def list_tree(repo_id: str, revision: str, token: str, subpath: str = "") -> list[dict[str, Any]]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    suffix = f"/{urllib.parse.quote(subpath, safe='/')}" if subpath else ""
    url = f"https://huggingface.co/api/datasets/{encoded_repo}/tree/{encoded_revision}{suffix}?limit=1000"
    return api_json(url, token)


def build_manifest(repo_id: str, revision: str, token: str, include_resources: bool) -> list[dict[str, Any]]:
    pending = [""]
    files = []
    while pending:
        subpath = pending.pop()
        for entry in list_tree(repo_id, revision, token, subpath):
            if entry.get("type") == "directory":
                if include_resources or entry["path"] != "resources":
                    pending.append(entry["path"])
            elif entry.get("type") == "file":
                files.append({
                    "path": entry["path"],
                    "size": int(entry.get("size") or 0),
                    "lfs_oid": (entry.get("lfs") or {}).get("oid"),
                })
    return sorted(files, key=lambda item: item["path"])


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(repo_id, safe='/')}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(path, safe='/')}?download=true"
    )


def download_one(
    entry: dict[str, Any],
    *,
    repo_id: str,
    revision: str,
    output_dir: Path,
    token: str,
) -> dict[str, Any]:
    target = output_dir / entry["path"]
    expected_size = int(entry.get("size") or 0)
    if target.is_file() and (expected_size <= 0 or target.stat().st_size == expected_size):
        return {**entry, "status": "already_complete"}

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    command = [
        "curl",
        "-fL",
        "--silent",
        "--show-error",
        "--retry",
        "12",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "-H",
        f"Authorization: Bearer {token}",
        "-o",
        str(partial),
        resolve_url(repo_id, revision, entry["path"]),
    ]
    subprocess.run(command, check=True)
    actual_size = partial.stat().st_size
    if expected_size > 0 and actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {entry['path']}: expected {expected_size}, got {actual_size}"
        )
    os.replace(partial, target)
    return {**entry, "status": "downloaded"}


def main() -> None:
    args = parse_args()
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("Set HF_TOKEN or pass --token-file for this gated dataset")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = build_manifest(args.repo_id, args.revision, token, args.include_resources)
    if args.max_files is not None:
        files = files[: args.max_files]
    total_bytes = sum(entry["size"] for entry in files)
    print(
        f"Manifest: {len(files)} files, {total_bytes / (1024 ** 3):.2f} GiB -> {output_dir}",
        flush=True,
    )
    if args.metadata_only:
        return

    completed = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_entry = {
            executor.submit(
                download_one,
                entry,
                repo_id=args.repo_id,
                revision=args.revision,
                output_dir=output_dir,
                token=token,
            ): entry
            for entry in files
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_entry), start=1):
            entry = future_to_entry[future]
            try:
                result = future.result()
                completed.append(result)
                print(
                    f"[{index}/{len(files)}] {result['status']}: {result['path']} "
                    f"({result['size'] / (1024 ** 2):.1f} MiB)",
                    flush=True,
                )
            except Exception as exc:
                failures.append({**entry, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(files)}] FAILED: {entry['path']}: {exc}", file=sys.stderr, flush=True)

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "num_files": len(files),
        "total_bytes": total_bytes,
        "num_complete": len(completed),
        "num_failures": len(failures),
        "files": completed,
        "failures": failures,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
