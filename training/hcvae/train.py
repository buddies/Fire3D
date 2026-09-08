#!/usr/bin/env python3
"""Train a Fire3D shape or PBR Hierarchical Compression VAE.

This is the local launcher for ``trellis2/configs/shape_vae_x2.yaml``.  It is
based on the reference TRELLIS2 launcher, with the behavior kept generic:

    python train_shape_vae_x2.py --config trellis2/configs/shape_vae_x2.yaml

    torchrun --standalone --nproc_per_node=4 train_shape_vae_x2.py \
        --config trellis2/configs/shape_vae_x2.yaml

Config values can be overridden with dotted keys:

    --dataset.args.roots /path/a,/path/b
    --trainer.args.output_dir results/shape_vae_x2/fire3d
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
for root in (REPO_ROOT, TRELLIS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training.config import load_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a Fire3D HC-VAE")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs/training/hcvae/shape.yaml"),
        help="YAML/JSON experiment config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory or parent run directory to resume from.",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Specific checkpoint step. If omitted, the latest complete step is used.",
    )
    parser.add_argument("--tryrun", action="store_true", help="Build everything and exit before training.")
    parser.add_argument("--profile", action="store_true", help="Run the trainer profiler instead of fit.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used before model construction on every rank.")
    return parser.parse_known_args()


def parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        return value


def update_config(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if not arg.startswith("--"):
            i += 1
            continue

        key = arg[2:]
        if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
            value = parse_override_value(overrides[i + 1])
            i += 2
        else:
            value = True
            i += 1

        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config


def find_latest_checkpoint_step(ckpt_dir: str) -> int | None:
    if not os.path.isdir(ckpt_dir):
        return None

    pattern = re.compile(r"(?:misc|encoder|decoder)_step(\d+)\.pt$")
    complete_steps: dict[int, set[str]] = {}
    for filename in os.listdir(ckpt_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        step = int(match.group(1))
        prefix = filename.split("_step", 1)[0]
        complete_steps.setdefault(step, set()).add(prefix)

    complete = [step for step, prefixes in complete_steps.items() if {"misc", "encoder", "decoder"} <= prefixes]
    return max(complete) if complete else None


def resolve_resume(resume_path: str, resume_step: int | None) -> tuple[str, int]:
    ckpt_dir = resume_path
    if os.path.basename(ckpt_dir) == "ckpts":
        load_dir = os.path.dirname(ckpt_dir)
    else:
        maybe_ckpts = os.path.join(ckpt_dir, "ckpts")
        if os.path.isdir(maybe_ckpts):
            load_dir = ckpt_dir
            ckpt_dir = maybe_ckpts
        else:
            load_dir = os.path.dirname(ckpt_dir)

    step = resume_step if resume_step is not None else find_latest_checkpoint_step(ckpt_dir)
    if step is None:
        raise ValueError(f"No complete checkpoints found in {ckpt_dir}")

    missing = [
        name
        for name in (
            f"misc_step{step:07d}.pt",
            f"encoder_step{step:07d}.pt",
            f"decoder_step{step:07d}.pt",
        )
        if not os.path.exists(os.path.join(ckpt_dir, name))
    ]
    if missing:
        raise ValueError(f"Incomplete checkpoint step {step} in {ckpt_dir}: missing {', '.join(missing)}")
    return load_dir, step


def setup_ddp() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
    if backend == "nccl":
        try:
            dist.init_process_group(backend, device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend)
    else:
        dist.init_process_group(backend)
    return True, rank, local_rank, world_size


def build_models(config: dict[str, Any], device: torch.device, is_master: bool) -> dict[str, torch.nn.Module]:
    from trellis2 import models

    built = {}
    for name, model_cfg in config["models"].items():
        model_cls = getattr(models, model_cfg["name"])
        with torch.cuda.device(device):
            model = model_cls(**model_cfg.get("args", {})).to(device)
        built[name] = model
        if is_master:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {model_cfg['name']} ({num_params:,} params)")
    return built


def save_run_config(config: dict[str, Any], argv: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(" ".join(argv) + "\n")


def main() -> None:
    args, overrides = parse_args()
    config = update_config(load_config(args.config), overrides)

    is_ddp, rank, local_rank, world_size = setup_ddp()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resume is not None:
        load_dir, step = resolve_resume(args.resume, args.resume_step)
        config.setdefault("trainer", {}).setdefault("args", {})["load_dir"] = load_dir
        config["trainer"]["args"]["step"] = step
    else:
        load_dir, step = None, None

    from trellis2 import datasets, trainers

    if is_master:
        print("=" * 80)
        print("Fire3D Hierarchical Compression VAE Training")
        print("=" * 80)
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        if load_dir is not None:
            print(f"Resume: {load_dir} @ step {step}")
        print("Building models...")

    model_dict = build_models(config, device, is_master)
    if is_ddp:
        torch.cuda.synchronize()
        dist.barrier()

    if is_master:
        print("Building dataset...")
    dataset_cfg = config["dataset"]
    dataset = getattr(datasets, dataset_cfg["name"])(**dataset_cfg.get("args", {}))
    if is_master:
        print(f"  {dataset_cfg['name']}: {len(dataset):,} samples")
        print("Building trainer...")

    trainer_cfg = config["trainer"]
    trainer = getattr(trainers, trainer_cfg["name"])(
        models=model_dict,
        dataset=dataset,
        **trainer_cfg.get("args", {}),
    )

    if is_master:
        save_run_config(config, sys.argv, trainer_cfg["args"]["output_dir"])

    if args.tryrun:
        if is_master:
            if getattr(trainer, 'wandb_run', None) is not None:
                trainer.wandb_run.finish()
            print("Tryrun complete; exiting before training.")
    elif args.profile:
        trainer.profile()
    else:
        trainer.run()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
