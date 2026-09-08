#!/usr/bin/env python3
"""Fetch pinned external toolkits required by Fire3D data processing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("registry.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        help="Source names from registry.json; omit to fetch every source.",
    )
    return parser.parse_args(argv)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def fetch_source(name: str, record: dict[str, str]) -> Path:
    destination = (ROOT / record["checkout"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"Refusing to replace non-Git directory: {destination}")
        run(["git", "clone", "--filter=blob:none", record["repository"], str(destination)])
    run(["git", "fetch", "--depth", "1", "origin", record["revision"]], cwd=destination)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if revision != record["revision"]:
        raise RuntimeError(f"{name} resolved to {revision}, expected {record['revision']}")
    print(f"{name}: {destination} @ {revision}")
    return destination


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = registry.get("upstream_sources", {})
    selected = args.sources or sorted(sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise SystemExit(f"Unknown data-processing sources: {', '.join(unknown)}")
    for name in selected:
        fetch_source(name, sources[name])


if __name__ == "__main__":
    main()
