import csv
import json
from pathlib import Path


def parse_captions(value):
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def model_name_from_path(model_path):
    path = Path(model_path)
    if path.name in {"raw_model.obj", "model.glb"}:
        return path.parent.name
    return path.stem


def read_metadata_rows(metadata_file):
    metadata_file = Path(metadata_file)
    with open(metadata_file, "r", newline="") as f:
        return list(csv.DictReader(f))


def lookup_entry(row):
    return {
        "aesthetic_score": parse_float(row.get("aesthetic_score")),
        "captions": parse_captions(row.get("captions")),
    }


def build_metadata_mapping(metadata_file, dataset_root, data_dir, model_paths=None, model_filename=None):
    metadata_file = Path(metadata_file)
    dataset_root = Path(dataset_root)
    data_dir = Path(data_dir)
    rows = read_metadata_rows(metadata_file)
    model_paths = [str(Path(p)) for p in model_paths] if model_paths is not None else None
    model_path_set = set(model_paths) if model_paths is not None else None

    by_path = {}
    by_model_name = {}
    unmatched_rows = 0

    for row in rows:
        entry = lookup_entry(row)
        candidates = []

        local_path = row.get("local_path")
        if local_path:
            local_candidate = dataset_root / local_path
            candidates.append(local_candidate)
            if local_path.startswith("raw/"):
                candidates.append(dataset_root / local_path[4:])

        sha256 = row.get("sha256")
        if sha256 and model_filename:
            candidates.append(data_dir / sha256 / model_filename)

        if row.get("file_identifier") and not local_path:
            file_identifier = row["file_identifier"]
            candidates.append(data_dir / file_identifier)
            if model_filename:
                candidates.append(data_dir / file_identifier / model_filename)

        matched_paths = []
        for candidate in candidates:
            candidate_path = str(candidate)
            if candidate_path in matched_paths:
                continue
            if model_path_set is None or candidate_path in model_path_set:
                matched_paths.append(candidate_path)

        if not matched_paths:
            unmatched_rows += 1
            continue

        for model_path in matched_paths:
            by_path[model_path] = entry
            by_model_name[model_name_from_path(model_path)] = entry

    return {
        "metadata_csv": str(metadata_file),
        "dataset_root": str(dataset_root),
        "num_rows": len(rows),
        "num_matched_paths": len(by_path),
        "num_unmatched_rows": unmatched_rows,
        "by_path": by_path,
        "by_model_name": by_model_name,
    }


def save_metadata_mapping(mapping, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)
    return str(output_file)
