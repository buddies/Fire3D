#!/usr/bin/env python3
"""Audit Fire3D baseline adapters and optional pinned upstream checkouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("registry.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-checkouts", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _head(checkout: Path) -> str | None:
    if not (checkout / ".git").is_dir():
        return None
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text())
    checkout_root = ROOT / registry["checkout_root"]
    report = {"schema": "fire3d.baseline_doctor.v1", "methods": {}}
    failed = False
    for name, record in registry["methods"].items():
        missing_adapters = [path for path in record["adapters"] if not (ROOT / path).is_file()]
        actual = _head(checkout_root / name)
        checkout_ok = actual == record["revision"]
        adapter_ok = not missing_adapters
        failed |= not adapter_ok or (args.require_checkouts and not checkout_ok)
        report["methods"][name] = {
            "adapters_ok": adapter_ok,
            "missing_adapters": missing_adapters,
            "checkout_present": actual is not None,
            "checkout_revision_ok": checkout_ok,
            "expected_revision": record["revision"],
            "actual_revision": actual,
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, record in report["methods"].items():
            checkout = "ready" if record["checkout_revision_ok"] else "not installed"
            adapters = "ready" if record["adapters_ok"] else "missing"
            print(f"{name:12s} adapters={adapters:7s} upstream={checkout}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
