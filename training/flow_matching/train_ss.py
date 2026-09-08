"""Train dense LC64 SS-VAE X2 flow matching with scene/object conditioning."""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
from datetime import datetime

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.datasets.scene_objects_dataset_ss_x2_offline_with_objects import get_dataloader
from models.object_gen_ss_x2_offline_with_dino import ObjectGen
from training.trainers.object_gen_trainer_ss_x2_offline_with_objects import Trainer
from training.config import load_config
from utils.grad_clip import AdaptiveGradClipper


def _latest_checkpoint_from_run_dir(run_dir):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    files = [name for name in os.listdir(checkpoint_dir) if name.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under {checkpoint_dir}")
    files.sort(key=lambda name: int(name.rsplit("_", 1)[-1].split(".")[0]))
    return os.path.join(checkpoint_dir, files[-1])


def _resolve_checkpoint_path(path):
    if os.path.isdir(path):
        return _latest_checkpoint_from_run_dir(path)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def validate_static_contract(config):
    data = config["data"]
    model = config["model"]
    flow = model["flow_model_ss_x2"]
    expected = {
        "data.ss_input_resolution": (int(data.get("ss_input_resolution", -1)), 8),
        "data.ss_x2_resolution": (int(data.get("ss_x2_resolution", -1)), 2),
        "data.ss_x2_channels": (int(data.get("ss_x2_channels", -1)), 8),
        "flow.resolution": (int(flow.get("resolution", -1)), 2),
        "flow.in_channels": (int(flow.get("in_channels", -1)), 8),
        "flow.out_channels": (int(flow.get("out_channels", -1)), 8),
    }
    mismatches = [f"{name}={actual}, expected {wanted}" for name, (actual, wanted) in expected.items() if actual != wanted]
    if mismatches:
        raise ValueError("Invalid LC64 SS-flow tensor contract: " + "; ".join(mismatches))

    required_rotations = ["000", "090", "180", "270"]
    for key in ("object_shape_trellis2_rotation", "scene_shape_trellis2_rotation"):
        actual = [str(value).replace("rot", "").zfill(3) for value in data.get(key, [])]
        if actual != required_rotations:
            raise ValueError(f"data.{key} must be exactly {required_rotations}, got {actual}")

    architecture = model.get("architecture_contract")
    if architecture == "ss_flow_dit_1p3b_30block":
        required_architecture = {
            "model_channels": 1536,
            "num_blocks": 30,
            "num_heads": 12,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "scaled",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
        }
        bad = {
            key: (flow.get(key), value)
            for key, value in required_architecture.items()
            if flow.get(key) != value
        }
        if bad:
            raise ValueError(f"30-block SS DiT architecture mismatch: {bad}")

    data_artifact = data.get("expected_ss_artifact_id")
    model_artifact = model.get("ss_x2", {}).get("artifact_id")
    if not data_artifact or data_artifact != model_artifact:
        raise ValueError("SS artifact ID must be present and identical in data and model configs")
    data_sha = str(data.get("expected_ss_encoder_sha256", "")).lower()
    model_sha = str(model.get("ss_x2", {}).get("encoder_sha256", "")).lower()
    if not data_sha or data_sha != model_sha:
        raise ValueError("SS encoder SHA-256 must be present and identical in data and model configs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs/training/flow_matching/ss.yaml"),
    )
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--exp_name", default="obj_gen_ss_x2_offline")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_keep_config", action="store_true")
    parser.add_argument("--init_from_checkpoint", default=None)
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--save_every_n_steps_override", type=int, default=None)
    parser.add_argument("--log_every_n_steps_override", type=int, default=None)
    parser.add_argument("--val_every_n_steps_override", type=int, default=None)
    parser.add_argument("--val_num_override", type=int, default=None)
    parser.add_argument("--trainval_num_override", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps_override", type=int, default=None)
    parser.add_argument("--report_to_override", default=None)
    parser.add_argument(
        "--smoke_micro_steps",
        type=int,
        default=0,
        help="Run this many prepared micro-batches without checkpointing, then exit.",
    )
    parser.add_argument("--seed_override", type=int, default=None)
    parser.add_argument("--verify_resume_only", action="store_true")
    parser.add_argument(
        "--contract_check_only",
        action="store_true",
        help="Validate tensor/architecture/checkpoint identity fields without constructing data or models.",
    )
    return parser.parse_args()


def main():
    if os.environ.get("FF_REGISTER_STACK_DUMP", "0").lower() in {"1", "true", "yes", "on"}:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        print("Registered SIGUSR1 Python stack dumps", flush=True)
    args = parse_args()
    if args.resume is not None and str(args.resume).lower() in {"", "none", "null"}:
        args.resume = None
    if args.resume is not None and args.init_from_checkpoint is not None:
        raise ValueError("Use either --resume or --init_from_checkpoint, not both")
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
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(args.resume)
        if not args.resume_keep_config:
            args.config = os.path.join(args.resume, "config.yaml")

    config = load_config(args.config)
    validate_static_contract(config)
    if args.contract_check_only:
        print("LC64 SS-flow static contract verified")
        return

    override_map = {
        "max_steps": args.max_steps_override,
        "save_every_n_steps": args.save_every_n_steps_override,
        "log_every_n_steps": args.log_every_n_steps_override,
        "val_every_n_steps": args.val_every_n_steps_override,
        "val_num": args.val_num_override,
        "trainval_num": args.trainval_num_override,
        "gradient_accumulation_steps": args.gradient_accumulation_steps_override,
    }
    applied = {key: int(value) for key, value in override_map.items() if value is not None}
    config["training"].update(applied)
    if args.seed_override is not None:
        seed = int(args.seed_override)
        config["training"]["seed"] = seed
        config["training"]["probe_validation_seed"] = seed
        config["data"]["loader_seed"] = seed
        applied["seed"] = seed
    if args.report_to_override is not None:
        config["training"]["report_to"] = args.report_to_override
        applied["report_to"] = args.report_to_override
    if applied:
        print(f"Applied training overrides: {applied}")
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
        run_name = f"{args.exp_name}_{timestamp}"
        init_kwargs = None
        if report_to == "wandb":
            wandb_kwargs = {"name": run_name}
            wandb_run_id = os.environ.get("WANDB_RUN_ID")
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                wandb_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
            init_kwargs = {"wandb": wandb_kwargs}
        accelerator.init_trackers(
            project_name=config.get("project_name", "Fire3D"),
            config=config,
            init_kwargs=init_kwargs,
        )

    model = ObjectGen(config["model"])
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")
    train_loader = get_dataloader(config["data"], split="train")
    validation_enabled = bool(config["training"].get("enable_validation", True)) and int(
        config["training"].get("val_every_n_steps", 1000)
    ) > 0
    if validation_enabled:
        trainval_loader = get_dataloader(config["data"], split="trainval")
        val_loader = get_dataloader(config["data"], split="val")
    else:
        trainval_loader = val_loader = None

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        eps=float(config["training"]["eps"]),
    )
    grad_clipper = AdaptiveGradClipper(
        max_norm=float(config["training"]["gradient_clip"]),
        clip_percentile=float(config["training"]["gradient_clip_percentile"]),
    )
    resume_ema_state = None
    if args.init_from_checkpoint is not None:
        checkpoint = torch.load(
            _resolve_checkpoint_path(args.init_from_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
    if args.resume is not None:
        checkpoint_path = _latest_checkpoint_from_run_dir(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        grad_clipper.load_state_dict(checkpoint["gradient_clipper"])
        resume_ema_state = checkpoint.get("ema_model")
        global_step = int(checkpoint["global_step"])

    if validation_enabled:
        model, optimizer, train_loader, trainval_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, trainval_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

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
        trainer.load_ema_state_dict(resume_ema_state)
    if args.verify_resume_only:
        if args.resume is None:
            raise ValueError("--verify_resume_only requires --resume")
        accelerator.wait_for_everyone()
        print(f"Verified resume at global step {trainer.global_step}")
        return
    if args.smoke_micro_steps > 0:
        trainer.model.train()
        train_iter = iter(train_loader)
        synchronized_steps = 0
        for micro_step in range(1, args.smoke_micro_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            loss_dict, metrics_dict, did_sync = trainer.train_step(batch)
            synchronized_steps += int(did_sync)
            loss = loss_dict["loss"]
            print(
                f"[smoke] micro_step={micro_step}/{args.smoke_micro_steps} "
                f"did_sync={did_sync} loss={float(loss.detach().float().cpu()):.6f} "
                f"metrics={sorted(metrics_dict)}",
                flush=True,
            )
        if synchronized_steps < 1:
            raise RuntimeError(
                "Smoke run did not reach an optimizer step; increase --smoke_micro_steps"
            )
        accelerator.wait_for_everyone()
        print(
            f"SS-flow smoke completed: micro_steps={args.smoke_micro_steps} "
            f"optimizer_steps={synchronized_steps}",
            flush=True,
        )
        return
    trainer.fit()


if __name__ == "__main__":
    main()
