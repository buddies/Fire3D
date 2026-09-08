"""Train object-only LC64 Shape-X2 + RGB -> PBR-X2 sparse flow."""

import argparse
import os
import sys
from datetime import datetime

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.datasets.scene_objects_dataset_pbr_x2_offline_with_objects import get_dataloader
from models.object_gen_pbr_x2_offline_with_dino import ObjectGen
from training.trainers.object_gen_trainer_pbr_x2_offline_with_objects import Trainer
from training.config import load_config
from utils.grad_clip import AdaptiveGradClipper


def _latest_checkpoint_from_run_dir(run_dir):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    paths = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".pt")
    ]
    if not paths:
        raise FileNotFoundError(f"No checkpoints under {checkpoint_dir}")
    paths.sort(key=lambda path: int(path.rsplit("_", 1)[-1].split(".")[0]))
    return paths[-1]


def _resolve_checkpoint_path(path):
    if os.path.isdir(path):
        return _latest_checkpoint_from_run_dir(path)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs/training/flow_matching/pbr.yaml"),
    )
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--exp_name", default="default")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_keep_config", action="store_true")
    parser.add_argument("--init_from_checkpoint", default=None)
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--save_every_n_steps_override", type=int, default=None)
    parser.add_argument("--log_every_n_steps_override", type=int, default=None)
    parser.add_argument("--val_every_n_steps_override", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps_override", type=int, default=None)
    parser.add_argument("--seed_override", type=int, default=None)
    parser.add_argument("--verify_resume_only", action="store_true")
    args = parser.parse_args()

    if args.resume is not None and str(args.resume).lower() in {"", "none", "null"}:
        args.resume = None
    if args.resume is not None and args.init_from_checkpoint is not None:
        raise ValueError("Use either --resume or --init_from_checkpoint")
    if args.resume == "latest":
        experiment_dir = os.path.join(args.output_dir, args.exp_name)
        args.resume = None
        if os.path.isdir(experiment_dir):
            for candidate in sorted(os.listdir(experiment_dir), reverse=True):
                run_dir = os.path.join(experiment_dir, candidate)
                checkpoint_dir = os.path.join(run_dir, "checkpoints")
                if os.path.isdir(checkpoint_dir) and any(
                    name.endswith(".pt") for name in os.listdir(checkpoint_dir)
                ):
                    args.resume = run_dir
                    break
        if args.resume is None:
            print("No complete checkpoint found; starting a fresh run.")

    if args.resume is not None and not args.resume_keep_config:
        args.config = os.path.join(args.resume, "config.yaml")
    config = load_config(args.config)

    overrides = {
        "max_steps": args.max_steps_override,
        "save_every_n_steps": args.save_every_n_steps_override,
        "log_every_n_steps": args.log_every_n_steps_override,
        "val_every_n_steps": args.val_every_n_steps_override,
        "gradient_accumulation_steps": args.gradient_accumulation_steps_override,
    }
    config["training"].update({key: value for key, value in overrides.items() if value is not None})
    if args.seed_override is not None:
        config["training"]["seed"] = args.seed_override
        config["training"]["probe_validation_seed"] = args.seed_override
        config["data"]["loader_seed"] = args.seed_override
    seed = config["training"].get("seed")
    if seed is not None:
        set_seed(int(seed), device_specific=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.exp_name, timestamp)
    config["training"]["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")
    config["training"]["val_dir"] = os.path.join(output_dir, "val")
    config["training"]["trainval_dir"] = os.path.join(output_dir, "trainval")

    report_to = config["training"].get("report_to", "wandb")
    if report_to is not None and str(report_to).lower() in {"", "none", "null", "false"}:
        report_to = None
    accelerator = Accelerator(
        mixed_precision=config["training"]["mixed_precision"],
        log_with=report_to,
        project_dir=output_dir,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        split_batches=False,
    )
    if report_to is not None:
        accelerator.init_trackers(
            project_name=config.get("project_name", "Fire3D"),
            config=config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}_{timestamp}"}} if report_to == "wandb" else None,
        )

    model = ObjectGen(config["model"])
    train_loader = get_dataloader(config["data"], "train")
    enable_validation = bool(config["training"].get("enable_validation", False)) and int(
        config["training"].get("val_every_n_steps", 0)
    ) > 0
    trainval_loader = get_dataloader(config["data"], "trainval") if enable_validation else None
    val_loader = get_dataloader(config["data"], "val") if enable_validation else None

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accelerator.print(
        f"Trainable parameters: {sum(p.numel() for p in trainable):,}; "
        f"frozen: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        eps=float(config["training"]["eps"]),
    )
    grad_clipper = AdaptiveGradClipper(
        max_norm=float(config["training"]["gradient_clip"]),
        clip_percentile=float(config["training"]["gradient_clip_percentile"]),
    )

    resume_ema = None
    if args.init_from_checkpoint is not None:
        checkpoint = torch.load(
            _resolve_checkpoint_path(args.init_from_checkpoint), map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    if args.resume is not None:
        checkpoint_path = _latest_checkpoint_from_run_dir(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        grad_clipper.load_state_dict(checkpoint["gradient_clipper"])
        resume_ema = checkpoint.get("ema_model")
        global_step = int(checkpoint["global_step"])
        accelerator.print(f"Loaded {checkpoint_path} at step {global_step}")

    if enable_validation:
        model, optimizer, train_loader, trainval_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, trainval_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        grad_clipper=grad_clipper,
        train_loader=train_loader,
        trainval_loader=trainval_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        config=config,
    )
    if args.resume is not None:
        trainer.global_step = global_step
        trainer.load_ema_state_dict(resume_ema)
    if args.verify_resume_only:
        if args.resume is None:
            raise ValueError("--verify_resume_only requires --resume")
        accelerator.print(f"Verified complete resume at step {trainer.global_step}")
        return

    profile_mode = os.environ.get("FF_PROFILE_TRAINING", "").lower()
    if profile_mode in {"1", "timing", "lightweight"}:
        trainer.fit_with_timing_profile(
            steps=int(os.environ.get("FF_PROFILE_STEPS", "12")),
            warmup=int(os.environ.get("FF_PROFILE_WARMUP", "2")),
        )
    else:
        trainer.fit()
    accelerator.end_training()


if __name__ == "__main__":
    main()
