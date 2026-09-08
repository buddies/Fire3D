#!/usr/bin/env python3
"""Evaluate a released Fire3D HC-VAE encoder/decoder pair on sparse latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
for root in (REPO_ROOT, TRELLIS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("shape", "pbr"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a state-dict checkpoint: {path}")
    for key in ("model", "state_dict", "ema_model"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _build_model(config: dict[str, Any], name: str, checkpoint: Path, device: torch.device):
    from trellis2 import models

    model_config = config["models"][name]
    model = getattr(models, model_config["name"])(**model_config.get("args", {}))
    incompatible = model.load_state_dict(_state_dict(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch for {name}: {incompatible}")
    return model.to(device).eval()


def _sample_metrics(target, prediction) -> dict[str, float]:
    target_coords = target.coords.detach().cpu().numpy()
    pred_coords = prediction.coords.detach().cpu().numpy()
    target_index = {tuple(row): index for index, row in enumerate(target_coords)}
    pred_index = {tuple(row): index for index, row in enumerate(pred_coords)}
    shared = sorted(set(target_index) & set(pred_index))

    true_count = len(target_index)
    pred_count = len(pred_index)
    matched_count = len(shared)
    precision = matched_count / max(pred_count, 1)
    recall = matched_count / max(true_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result = {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }
    if shared:
        target_rows = torch.as_tensor(
            [target_index[key] for key in shared], device=target.feats.device
        )
        pred_rows = torch.as_tensor(
            [pred_index[key] for key in shared], device=prediction.feats.device
        )
        target_feats = target.feats[target_rows].float()
        pred_feats = prediction.feats[pred_rows].float()
        delta = pred_feats - target_feats
        result["feature_l1"] = float(delta.abs().mean().item())
        result["feature_mse"] = float(delta.square().mean().item())
    else:
        result["feature_l1"] = float("nan")
        result["feature_mse"] = float("nan")
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config or REPO_ROOT / f"configs/training/hcvae/{args.kind}.yaml"
    config = load_config(config_path)
    device = torch.device(args.device)

    from trellis2 import datasets

    dataset_config = config["dataset"]
    dataset = getattr(datasets, dataset_config["name"])(**dataset_config.get("args", {}))
    encoder = _build_model(config, "encoder", args.encoder, device)
    decoder = _build_model(config, "decoder", args.decoder, device)

    records = []
    limit = min(len(dataset), args.max_samples)
    with torch.inference_mode():
        for index in range(limit):
            target = dataset[index]["x_0"].to(device)
            latent = encoder(target, sample_posterior=False)
            prediction = decoder(latent)
            records.append(_sample_metrics(target, prediction))

    summary = {
        key: float(np.nanmean([record[key] for record in records]))
        for key in records[0]
    } if records else {}
    report = {
        "schema": "fire3d.hcvae_evaluation.v1",
        "kind": args.kind,
        "samples": len(records),
        "metrics": summary,
        "config": str(Path(config_path)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
