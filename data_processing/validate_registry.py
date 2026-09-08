#!/usr/bin/env python3
"""Validate that every documented Fire3D preprocessing adapter is present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("registry.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(REGISTRY.read_text())
    records = []
    for name, value in payload.get("upstream_sources", {}).items():
        missing_fields = [
            field
            for field in ("repository", "revision", "checkout")
            if not value.get(field)
        ]
        records.append(
            {
                "group": "upstream_sources",
                "name": name,
                "missing": missing_fields,
            }
        )
    for group in ("scene_datasets", "object_datasets", "shared_stages"):
        for name, value in payload[group].items():
            paths = value.values() if isinstance(value, dict) else (value,)
            missing = [path for path in paths if not (ROOT / path).is_file()]
            records.append({"group": group, "name": name, "missing": missing})
    report = {
        "schema": "fire3d.data_processing_validation.v1",
        "ok": all(not record["missing"] for record in records),
        "records": records,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for record in records:
            state = "ready" if not record["missing"] else "missing"
            print(f"{record['group']:16s} {record['name']:28s} {state}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
