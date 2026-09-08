"""Train offline Shape VAE X2 flow matching with scene/object conditioning."""

import argparse
import contextlib
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

from training.datasets.scene_objects_dataset_shape_x2_offline_with_objects import get_dataloader
from models.object_gen_shape_x2_offline_with_dino import ObjectGen
from training.trainers.object_gen_trainer_shape_x2_offline_with_objects import Trainer
from training.config import load_config
from utils.grad_clip import AdaptiveGradClipper


@contextlib.contextmanager
def _maybe_disable_redundant_ddp_init_sync(config, resume_path, accelerator):
    """Skip DDP's initial parameter broadcast only for identical resume loads.

    Every rank has already loaded the same checkpoint before ``prepare``.  On
    this Blackwell node the redundant initial synchronization of this 1.6B
    parameter model can trigger a CUDA illegal-memory fault, even though normal
    NCCL collectives pass.  Training gradient synchronization is unaffected.
    """
    enabled = not bool(config["training"].get("ddp_init_sync", True))
    if not enabled or accelerator.num_processes <= 1:
        yield
        return
    if resume_path is None:
        raise ValueError("training.ddp_init_sync=false is allowed only with --resume")

    original_ddp = torch.nn.parallel.DistributedDataParallel

    class ResumeDistributedDataParallel(original_ddp):
        def __init__(self, *args, **kwargs):
            kwargs["init_sync"] = False
            super().__init__(*args, **kwargs)

    torch.nn.parallel.DistributedDataParallel = ResumeDistributedDataParallel
    accelerator.print(
        "DDP initial model synchronization disabled: every rank loaded the "
        "identical resume checkpoint; gradient synchronization remains enabled."
    )
    try:
        yield
    finally:
        torch.nn.parallel.DistributedDataParallel = original_ddp


def _latest_checkpoint_from_run_dir(run_dir: str) -> str:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")
    ckpt_paths = [os.path.join(ckpt_dir, f) for f in ckpt_files]
    ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return ckpt_paths[-1]


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        return _latest_checkpoint_from_run_dir(path)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "configs/training/flow_matching/shape.yaml"),
    )
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--resume_keep_config",
        action="store_true",
        help="Resume optimizer/model/global_step from --resume, but keep --config instead of loading resume/config.yaml.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help="Load model weights from a checkpoint file or run dir while starting a new run with --config.",
    )
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--save_every_n_steps_override", type=int, default=None)
    parser.add_argument("--log_every_n_steps_override", type=int, default=None)
    parser.add_argument("--val_every_n_steps_override", type=int, default=None)
    parser.add_argument("--val_num_override", type=int, default=None)
    parser.add_argument("--trainval_num_override", type=int, default=None)
    parser.add_argument(
        "--seed_override",
        type=int,
        default=None,
        help="Override training, DataLoader, and fixed-validation RNG seeds together.",
    )
    parser.add_argument(
        "--verify_resume_only",
        action="store_true",
        help="Fully load a --resume checkpoint (including optimizer/EMA) and exit before fitting.",
    )
    args = parser.parse_args()
    if args.resume is not None and str(args.resume).lower() in {"", "none", "null"}:
        args.resume = None
    if args.resume is not None and args.init_from_checkpoint is not None:
        raise ValueError("Use either --resume or --init_from_checkpoint, not both")

    if args.resume == "latest":
        exp_dir = os.path.join(args.output_dir, args.exp_name)
        args.resume = None
        if not os.path.isdir(exp_dir):
            print(f"Experiment directory {exp_dir} not found. Starting fresh training.")
        else:
            candidates = sorted(
                [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
                reverse=True,
            )
            for candidate in candidates:
                ckpt_dir_candidate = os.path.join(exp_dir, candidate, "checkpoints")
                if os.path.isdir(ckpt_dir_candidate) and any(f.endswith(".pt") for f in os.listdir(ckpt_dir_candidate)):
                    args.resume = os.path.join(exp_dir, candidate)
                    break
            if args.resume is None:
                print(f"No valid checkpoint found under {exp_dir}. Starting fresh training.")

    if args.resume is not None:
        assert os.path.exists(args.resume), f"Checkpoint file {args.resume} not found"
        print(f"Resuming training from {args.resume}")
        if args.resume_keep_config:
            print(f"Keeping requested config: {args.config}")
        else:
            args.config = os.path.join(args.resume, "config.yaml")
            print(f"Loading config from {args.config}")

    config = load_config(args.config)

    override_map = {
        "max_steps": args.max_steps_override,
        "save_every_n_steps": args.save_every_n_steps_override,
        "log_every_n_steps": args.log_every_n_steps_override,
        "val_every_n_steps": args.val_every_n_steps_override,
        "val_num": args.val_num_override,
        "trainval_num": args.trainval_num_override,
    }
    applied_overrides = {
        key: int(value) for key, value in override_map.items() if value is not None
    }
    config["training"].update(applied_overrides)
    if args.seed_override is not None:
        seed_override = int(args.seed_override)
        config["training"]["seed"] = seed_override
        config["training"]["probe_validation_seed"] = seed_override
        config["data"]["loader_seed"] = seed_override
        applied_overrides["seed"] = seed_override
    if applied_overrides:
        print(f"Applied training overrides: {applied_overrides}")

    seed = config["training"].get("seed")
    if seed is not None:
        set_seed(int(seed), device_specific=False)
        print(f"Using deterministic probe seed: {int(seed)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.exp_name, timestamp)

    config["training"]["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")
    config["training"]["val_dir"] = os.path.join(output_dir, "val")
    config["training"]["trainval_dir"] = os.path.join(output_dir, "trainval")

    report_to = config["training"].get("report_to", "wandb")
    if report_to is not None and str(report_to).strip().lower() in {"", "none", "null", "false"}:
        report_to = None

    accelerator = Accelerator(
        mixed_precision=config["training"]["mixed_precision"],
        log_with=report_to,
        project_dir=output_dir,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        split_batches=False,
    )

    run_name = args.exp_name + "_" + timestamp
    if report_to is not None:
        init_kwargs = {"wandb": {"name": run_name}} if report_to == "wandb" else None
        accelerator.init_trackers(
            project_name=config.get("project_name", "Fire3D"),
            config=config,
            init_kwargs=init_kwargs,
        )

    model = ObjectGen(config["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    train_loader = get_dataloader(config["data"], split="train")
    enable_validation = (
        bool(config["training"].get("enable_validation", True))
        and int(config["training"].get("val_every_n_steps", 1000)) > 0
    )
    if enable_validation:
        trainval_loader = get_dataloader(config["data"], split="trainval")
        val_loader = get_dataloader(config["data"], split="val")
    else:
        trainval_loader = None
        val_loader = None
        if accelerator.is_main_process:
            print("Validation disabled: skipping trainval/val dataloader construction.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Model trainable parameters: {trainable_count:,} ({trainable_count / 1e6:.2f}M)")
    print(f"Model frozen parameters: {frozen_params:,} ({frozen_params / 1e6:.2f}M)")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(float(b) for b in config["training"]["betas"]),
        eps=float(config["training"]["eps"]),
    )

    grad_clipper = AdaptiveGradClipper(
        max_norm=float(config["training"]["gradient_clip"]),
        clip_percentile=float(config["training"]["gradient_clip_percentile"]),
    )

    resume_ema_state_dict = None
    if args.init_from_checkpoint is not None:
        init_ckpt = _resolve_checkpoint_path(args.init_from_checkpoint)
        print(f"Initializing model weights from {init_ckpt}")
        ckpt_dict = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_dict["model"])
        print(f"Loaded model weights from {init_ckpt} at global step {ckpt_dict.get('global_step')}")

    if args.resume is not None:
        latest_ckpt = _latest_checkpoint_from_run_dir(args.resume)
        print(f"Loading latest checkpoint from {latest_ckpt}")
        ckpt_dict = torch.load(latest_ckpt, map_location="cpu", weights_only=False)

        model.load_state_dict(ckpt_dict["model"])
        optimizer.load_state_dict(ckpt_dict["optimizer"])
        grad_clipper.load_state_dict(ckpt_dict["gradient_clipper"])
        resume_ema_state_dict = ckpt_dict.get("ema_model")
        global_step = ckpt_dict["global_step"]
        print(f"Loaded checkpoint from {latest_ckpt} with global step {global_step}")

    with _maybe_disable_redundant_ddp_init_sync(config, args.resume, accelerator):
        if enable_validation:
            model, optimizer, train_loader, trainval_loader, val_loader = accelerator.prepare(
                model, optimizer, train_loader, trainval_loader, val_loader
            )
        else:
            model, optimizer, train_loader = accelerator.prepare(
                model, optimizer, train_loader
            )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

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
        trainer.load_ema_state_dict(resume_ema_state_dict)
        print(f"Resumed training from global step {global_step}")

    if args.verify_resume_only:
        if args.resume is None:
            raise ValueError("--verify_resume_only requires --resume")
        accelerator.wait_for_everyone()
        print(
            f"Verified resume-only load at global step {trainer.global_step}: "
            "model, optimizer, gradient clipper, and EMA loaded successfully."
        )
        return

    profile_mode = os.environ.get("FF_PROFILE_TRAINING", "").strip().lower()
    if profile_mode in {"1", "timing", "lightweight"}:
        trainer.fit_with_timing_profile(
            steps=int(os.environ.get("FF_PROFILE_STEPS", "12")),
            warmup=int(os.environ.get("FF_PROFILE_WARMUP", "2")),
        )
    elif profile_mode in {"torch", "profiler"}:
        trainer.fit_with_profiling(
            profiler_log_dir=os.environ.get("FF_PROFILE_LOG_DIR", "./logs/profiler"),
            wait=int(os.environ.get("FF_PROFILE_WAIT", "1")),
            warmup=int(os.environ.get("FF_PROFILE_WARMUP", "2")),
            active=int(os.environ.get("FF_PROFILE_ACTIVE", "2")),
            repeat=int(os.environ.get("FF_PROFILE_REPEAT", "1")),
        )
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
