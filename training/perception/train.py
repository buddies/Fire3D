import argparse
import yaml
import os
from datetime import datetime
import torch
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.datasets.scene_object_poses_dataset_multiple_syn2real import (
    get_dataloader,
    get_deterministic_val_dataset,
    sparse_collate_fn,
)
from models.object_pose_w_seg_voxelize_dino import ObjectPoseWSegVoxelize
from training.trainers.object_pose_w_seg_voxelize_dino import Trainer
from training.config import load_config
from utils.scene_pose_perception_loss import ObjectPoseWSegLocalUpLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, ConstantLR

def _override_scheduler_lr_from_config(optimizer, scheduler, config_lr, min_lr):
    """Keep resume progress, but use the LR values from the current config."""
    checkpoint_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    config_lrs = [float(config_lr) for _ in optimizer.param_groups]

    def find_base_lrs(sched):
        if hasattr(sched, "base_lrs"):
            return sched.base_lrs
        for child in getattr(sched, "_schedulers", []):
            child_base_lrs = find_base_lrs(child)
            if child_base_lrs is not None:
                return child_base_lrs
        return None

    checkpoint_base_lrs = find_base_lrs(scheduler) or checkpoint_lrs
    lr_changed = any(
        abs(float(old_lr) - float(config_lr)) > 1e-12
        for old_lr in checkpoint_base_lrs
    )
    if not lr_changed:
        return

    for group, lr in zip(optimizer.param_groups, config_lrs):
        group["lr"] = lr
        group["initial_lr"] = lr

    def update_scheduler(sched):
        if hasattr(sched, "base_lrs"):
            sched.base_lrs = list(config_lrs)
        if hasattr(sched, "_last_lr"):
            sched._last_lr = list(config_lrs)
        if isinstance(sched, LinearLR):
            sched.start_factor = 1e-8 / float(config_lr)
            sched.end_factor = 1.0
        if isinstance(sched, CosineAnnealingLR) and min_lr is not None:
            sched.eta_min = float(min_lr)
        for child in getattr(sched, "_schedulers", []):
            update_scheduler(child)

    update_scheduler(scheduler)
    print(
        "Overrode checkpoint LR with config LR: "
        f"checkpoint base_lrs={list(checkpoint_base_lrs)}, config lr={float(config_lr)}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "configs/training/perception/default.yaml"),
    )
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Initialize model weights only; keep the current config and use a fresh optimizer/scheduler.",
    )
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--num_workers_override", type=int, default=None)
    parser.add_argument("--warmup_steps_override", type=int, default=None)
    parser.add_argument(
        "--restart_scheduler_on_resume",
        action="store_true",
        help=(
            "Preserve model/optimizer/EMA/global step but replace the saved LR "
            "scheduler. Intended when extending a completed short-horizon trial."
        ),
    )
    parser.add_argument(
        "--resume_lr_rewarm_steps",
        type=int,
        default=0,
        help="Steps to linearly rewarm from the checkpoint LR to config training.lr.",
    )
    parser.add_argument(
        "--resume_start_lr",
        type=float,
        default=None,
        help="Optional rewarm start LR; defaults to the LR stored in the checkpoint optimizer.",
    )
    parser.add_argument("--no_final_checkpoint", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--mixed_precision_override",
        choices=("no", "fp16", "bf16"),
        default=None,
        help="Override training.mixed_precision even when resuming a saved config.",
    )
    parser.add_argument(
        "--model_dtype_override",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Override model.scene_decoder.dtype when resuming a saved config.",
    )
    args = parser.parse_args()

    # check whether to resume training
    if args.resume == "latest":
        # Find the latest timestamp directory under output_dir/exp_name that has checkpoints
        exp_dir = os.path.join(args.output_dir, args.exp_name)
        assert os.path.isdir(exp_dir), f"Experiment directory {exp_dir} not found"
        candidates = sorted(
            [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
            reverse=True  # lexicographic descending = latest timestamp first
        )
        args.resume = None
        for candidate in candidates:
            ckpt_dir_candidate = os.path.join(exp_dir, candidate, "checkpoints")
            if os.path.isdir(ckpt_dir_candidate) and any(f.endswith(".pt") for f in os.listdir(ckpt_dir_candidate)):
                args.resume = os.path.join(exp_dir, candidate)
                break
        # assert args.resume is not None, f"No valid checkpoint found under {exp_dir}"
        # print(f"Auto-resolved latest resume dir: {args.resume}")
        if args.resume is None:
            print(f"No valid checkpoint found under {exp_dir}. Starting fresh training.")

    if args.resume is not None:
        assert os.path.exists(args.resume), f"Checkpoint file {args.resume} not found"
        print(f"Resuming training from {args.resume}")

        args.config = os.path.join(args.resume, "config.yaml")
        print(f"Loading config from {args.config}")

    # 1. Load Config
    config = load_config(args.config)
    if args.max_steps_override is not None:
        config['training']['max_steps'] = int(args.max_steps_override)
    if args.num_workers_override is not None:
        config['data']['num_workers'] = int(args.num_workers_override)
    if args.warmup_steps_override is not None:
        config['training']['warmup_steps'] = int(args.warmup_steps_override)
    if args.no_final_checkpoint:
        config['training']['save_final_checkpoint'] = False
    if args.restart_scheduler_on_resume and args.resume is None:
        raise ValueError("--restart_scheduler_on_resume requires --resume")
    if args.mixed_precision_override is not None:
        config['training']['mixed_precision'] = args.mixed_precision_override
    if args.model_dtype_override is not None:
        config['model']['scene_decoder']['dtype'] = args.model_dtype_override
    print(
        "Effective precision: accelerate={}, scene_decoder={}".format(
            config['training']['mixed_precision'],
            config['model']['scene_decoder'].get('dtype'),
        )
    )

    # 2. Initialize Accelerator (bf16 for Transformer stability)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.exp_name, timestamp)

    config['training']['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
    config['training']['val_dir'] = os.path.join(output_dir, 'val')
    config['training']['trainval_dir'] = os.path.join(output_dir, 'trainval')

    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
        log_with=None if args.disable_wandb else "wandb",
        project_dir=output_dir,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        split_batches=False,
    )

    # Initialize wandb tracker with timestamp run name
    run_name = args.exp_name + "_" + timestamp
    if not args.disable_wandb:
        accelerator.init_trackers(
            project_name=config.get("project_name", "Fire3D"),
            config=config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # 3. Setup Model & Data
    rotation_config = dict(config.get('rotation_symmetry', {}))
    rotation_z_quarter_turns = bool(rotation_config.get('z_quarter_turns', False))
    rotation_local_up_quarter_turns = bool(
        rotation_config.get('local_up_quarter_turns', False)
    )
    if rotation_z_quarter_turns:
        raise ValueError(
            "This point-normalized local-up fork does not support "
            "rotation_symmetry.z_quarter_turns; use local_up_quarter_turns."
        )
    if not rotation_local_up_quarter_turns:
        raise ValueError(
            "This training fork requires "
            "rotation_symmetry.local_up_quarter_turns: true"
        )
    model_config = dict(config['model'])
    model_config['rotation_symmetry'] = {
        'z_quarter_turns': False,
        'local_up_quarter_turns': rotation_local_up_quarter_turns,
    }
    learned_background_segmentation = bool(
        model_config.get('scene_decoder', {}).get(
            'learned_background_segmentation', False
        )
    )
    if learned_background_segmentation:
        raise ValueError(
            "This fork keeps background as a normal pose/segmentation "
            "instance. Remove model.scene_decoder.learned_background_segmentation."
        )
    print(
        "Rotation symmetry: local_up_quarter_turns={}, "
        "legacy_z_quarter_turns={}".format(
            rotation_local_up_quarter_turns,
            rotation_z_quarter_turns,
        )
    )
    print("Background supervision: background pose and segmentation are instance 0")
    model = ObjectPoseWSegVoxelize(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    if args.init_checkpoint is not None:
        assert os.path.isfile(args.init_checkpoint), (
            f"Initialization checkpoint not found: {args.init_checkpoint}"
        )
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", mmap=True)
        state_dict = checkpoint.get("model", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_keys = (
            {"scene_decoder.background_seg_feat"}
            if learned_background_segmentation
            else set()
        )
        disallowed_missing_keys = (
            set(incompatible.missing_keys) - allowed_missing_keys
        )
        if disallowed_missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Initialization checkpoint mismatch: "
                f"missing={sorted(disallowed_missing_keys)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            print(
                "Initialized new parameters absent from the source "
                f"checkpoint: {sorted(incompatible.missing_keys)}"
            )
        print(
            f"Initialized model weights from {args.init_checkpoint}; "
            "optimizer/scheduler/global step start fresh"
        )

    train_loader = get_dataloader(config['data'], split='train')
    evaluation_config = dict(config.get('evaluation', {}))
    mini_val_scenes = int(evaluation_config.get('mini_val_scenes', 8))
    mini_val_repeats = int(evaluation_config.get('augmentations_per_scene', 5))
    mini_val_dataset = get_deterministic_val_dataset(
        config['data'],
        num_scenes=mini_val_scenes,
        repeats=mini_val_repeats,
        seed=int(evaluation_config.get('seed', 20260808)),
    )
    val_num_workers = int(config['data'].get('val_num_workers', 2))
    val_loader = DataLoader(
        mini_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=sparse_collate_fn,
        pin_memory=True,
        persistent_workers=val_num_workers > 0,
    )
    trainval_loader = None

    # 4. Optimizer

    # Optimizer: only register parameters that should be trained (exclude frozen e.g. utonia_model)
    params_for_optimizer = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_for_optimizer,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay']),
        betas=tuple(float(b) for b in config['training']['betas']),
        eps=float(config['training']['eps'])
    )

    # Scheduler: Linear warmup + Cosine decay
    max_steps = config['training']['max_steps']
    warmup_steps = config['training'].get('warmup_steps', 0)
    min_lr_raw = config['training'].get('min_lr', 1e-6)
    min_lr = None if min_lr_raw is None else float(min_lr_raw)

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8 / float(config['training']['lr']),
            end_factor=1.0,
            total_iters=warmup_steps
        )
        if min_lr is None:
            # Keep base LR after warmup when min_lr is null.
            cosine_scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=max_steps - warmup_steps
            )
        else:
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=min_lr
            )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        if min_lr is None:
            scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=max_steps
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
                eta_min=min_lr
            )

    if args.resume is not None:
        # load the checkpoint
        ckpt_dir = os.path.join(args.resume, "checkpoints")
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        ckpt_paths = [os.path.join(ckpt_dir, f) for f in ckpt_files]
        ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_ckpt = ckpt_paths[-1]
        print(f"Loading latest checkpoint from {latest_ckpt}")
        ckpt_path = latest_ckpt
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")

        # load the saved model, optimizer, and update the global step
        model.load_state_dict(ckpt_dict["model"], strict=False)  # allow missing keys for new parameters
        optimizer.load_state_dict(ckpt_dict["optimizer"])
        global_step = ckpt_dict["global_step"]
        if args.restart_scheduler_on_resume:
            target_lr = float(config['training']['lr'])
            checkpoint_lr = float(optimizer.param_groups[0]['lr'])
            start_lr = (
                checkpoint_lr
                if args.resume_start_lr is None
                else float(args.resume_start_lr)
            )
            rewarm_steps = int(args.resume_lr_rewarm_steps)
            remaining_steps = int(max_steps) - int(global_step)
            if remaining_steps <= 0:
                raise ValueError(
                    f"max_steps={max_steps} must exceed resumed global_step={global_step}"
                )
            if rewarm_steps < 0 or rewarm_steps >= remaining_steps:
                raise ValueError(
                    "resume_lr_rewarm_steps must satisfy "
                    f"0 <= steps < remaining_steps ({remaining_steps})"
                )
            if not 0.0 < start_lr <= target_lr:
                raise ValueError(
                    f"resume start LR must be in (0, target_lr={target_lr}], got {start_lr}"
                )

            # Reset the optimizer's LR metadata while retaining every moment
            # tensor loaded above. The new scheduler's local epoch zero is the
            # resumed global step; its horizon covers only the remaining run.
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
                param_group['initial_lr'] = target_lr

            cosine_steps = max(1, remaining_steps - rewarm_steps)
            if rewarm_steps > 0:
                rewarm_scheduler = LinearLR(
                    optimizer,
                    start_factor=start_lr / target_lr,
                    end_factor=1.0,
                    total_iters=rewarm_steps,
                )
                if min_lr is None:
                    tail_scheduler = ConstantLR(
                        optimizer, factor=1.0, total_iters=cosine_steps
                    )
                else:
                    tail_scheduler = CosineAnnealingLR(
                        optimizer, T_max=cosine_steps, eta_min=min_lr
                    )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[rewarm_scheduler, tail_scheduler],
                    milestones=[rewarm_steps],
                )
            elif min_lr is None:
                scheduler = ConstantLR(
                    optimizer, factor=1.0, total_iters=remaining_steps
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=remaining_steps, eta_min=min_lr
                )
            print(
                "Restarted scheduler at resumed step "
                f"{global_step}: start_lr={start_lr:.8g}, "
                f"target_lr={target_lr:.8g}, rewarm_steps={rewarm_steps}, "
                f"remaining_steps={remaining_steps}, min_lr={min_lr}"
            )
        else:
            scheduler.load_state_dict(ckpt_dict["scheduler"])
            _override_scheduler_lr_from_config(
                optimizer=optimizer,
                scheduler=scheduler,
                config_lr=float(config['training']['lr']),
                min_lr=min_lr,
            )

        print(f"Loaded checkpoint from {ckpt_path} with global step {global_step}")

    # 5. The "Prepare" Step (Crucial for Accelerate)
    # This handles device placement (Multi-GPU) automatically
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # 6. Initialize Loss Function
    loss_config = dict(config.get('loss', {}))
    loss_config['rotation_local_up_quarter_turns'] = (
        rotation_local_up_quarter_turns
    )
    loss_config['learned_background_segmentation'] = False
    loss_fn = ObjectPoseWSegLocalUpLoss(loss_config)

    # Create output directory if it doesn't exist
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        # Save config to output_dir before training
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # 7. Start Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        trainval_loader=trainval_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        config=config,
        loss_fn=loss_fn,
    )

    if args.resume is not None:
        if (
            config['training'].get('use_ema', False)
            and trainer.ema_model is not None
            and 'ema_model' in ckpt_dict
        ):
            trainer.ema_model.shadow.load_state_dict(
                ckpt_dict['ema_model'], strict=True
            )
            print(f"Restored EMA weights from {ckpt_path}")
        trainer.global_step = global_step
        print(f"Resumed training from global step {global_step}")

    trainer.fit()
    # trainer.fit_with_profiling()

    # End tracking (closes wandb run properly)
    accelerator.end_training()

if __name__ == "__main__":
    main()
