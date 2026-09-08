#!/usr/bin/env python3
"""Clone baseline repositories at the revisions used by the Fire3D adapters."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("registry.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = json.loads(REGISTRY_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "methods",
        nargs="*",
        metavar="METHOD",
        help="Baseline names to fetch; omit to fetch every registered baseline.",
    )
    parser.add_argument("--checkout-root", type=Path, default=ROOT / registry["checkout_root"])
    args = parser.parse_args(argv)
    unknown = sorted(set(args.methods) - set(registry["methods"]))
    if unknown:
        parser.error(f"unknown baseline method(s): {', '.join(unknown)}")
    return args


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text())
    methods = args.methods or list(registry["methods"])
    args.checkout_root.mkdir(parents=True, exist_ok=True)
    for name in methods:
        record = registry["methods"][name]
        checkout = args.checkout_root / name
        if not checkout.exists():
            run(["git", "clone", "--filter=blob:none", record["repository"], str(checkout)])
        if not (checkout / ".git").is_dir():
            raise RuntimeError(f"Not a Git checkout: {checkout}")
        run(["git", "fetch", "origin", record["revision"]], cwd=checkout)
        run(["git", "checkout", "--detach", record["revision"]], cwd=checkout)
        print(f"{name}: {record['revision']}")


if __name__ == "__main__":
    main()
