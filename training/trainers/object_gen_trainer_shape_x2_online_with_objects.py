"""Trainer for online Shape VAE X2 flow matching with objects.

Forked from the feature trainer so the old precomputed feature-latent flow path
keeps its behavior.
"""
import os
import gc
import time
import json
import random
from contextlib import contextmanager
import torch
from tqdm.auto import tqdm
import numpy as np
import torch
# from utils.visualize import visualize_scene_inference
from utils.loss import pose_l1_loss, latent_loss, ss_latent_loss, feat_loss
from utils.ema import EMA

class Trainer:
    def __init__(self, model, optimizer, grad_clipper, train_loader, trainval_loader, val_loader, accelerator, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.trainval_loader = trainval_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.config = config
        self.global_step = 0
        self.gradient_clip = grad_clipper

        # Training config
        self.max_steps = config['training']['max_steps']
        self.save_every_n_steps = config['training'].get('save_every_n_steps', 5000)
        self.log_every_n_steps = config['training'].get('log_every_n_steps', 10)
        self.val_every_n_steps = config['training'].get('val_every_n_steps', 1000)
        self.inference_every_n_steps = config['training'].get('inference_every_n_steps', 1000)
        self.checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
        self.val_dir = config['training'].get('val_dir', 'val')
        self.val_num = config['training'].get('val_num', 1)
        self.trainval_dir = config['training'].get('trainval_dir', 'trainval')
        self.enable_validation = (
            bool(config['training'].get('enable_validation', True))
            and int(self.val_every_n_steps) > 0
            and self.trainval_loader is not None
            and self.val_loader is not None
        )
        self.trainval_num = config['training'].get('trainval_num', 1)
        self.use_ema = config['training'].get('use_ema', False)
        self.ema_rate = config['training'].get('ema_rate', 0.9999)
        self.checkpoint_to_cpu = bool(config['training'].get('checkpoint_to_cpu', True))
        self.probe_loss_only_validation = bool(
            config['training'].get('probe_loss_only_validation', False)
        )
        self.probe_validation_seed = config['training'].get('probe_validation_seed')
        if self.probe_validation_seed is not None:
            self.probe_validation_seed = int(self.probe_validation_seed)
        self.local_metrics_jsonl = config['training'].get('local_metrics_jsonl')
        if self.local_metrics_jsonl == 'auto':
            self.local_metrics_jsonl = os.path.join(
                os.path.dirname(self.checkpoint_dir), 'metrics.jsonl'
            )
        # Normal checkpointing should not inject CUDA/NCCL synchronization.
        # Use the debug knobs below when localizing async CUDA faults.
        self.checkpoint_synchronize_cuda = self._env_bool(
            "FF_CHECKPOINT_SYNC_CUDA",
            bool(config['training'].get('checkpoint_synchronize_cuda', False)),
        )
        self.checkpoint_barrier = self._env_bool(
            "FF_CHECKPOINT_BARRIER",
            bool(config['training'].get('checkpoint_barrier', False)),
        )
        self.debug_cuda_sync_every_n_steps = self._env_int(
            "FF_DEBUG_CUDA_SYNC_EVERY_N_STEPS",
            int(config['training'].get('debug_cuda_sync_every_n_steps', 0) or 0),
        )
        self.debug_log_batch_every_n_steps = self._env_int(
            "FF_DEBUG_LOG_BATCH_EVERY_N_STEPS",
            int(config['training'].get('debug_log_batch_every_n_steps', 0) or 0),
        )
        self.profile_step_timing = self._env_bool(
            "FF_PROFILE_STEP_TIMING",
            bool(config['training'].get('profile_step_timing', False)),
        )
        self._last_profile = {}
        self._last_data_load_ms = 0.0
        self.ema_model = None

        # Create checkpoint directory
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.enable_validation:
                os.makedirs(self.val_dir, exist_ok=True)
                os.makedirs(self.trainval_dir, exist_ok=True)
            if self.local_metrics_jsonl:
                os.makedirs(os.path.dirname(self.local_metrics_jsonl) or '.', exist_ok=True)

            # Initialize EMA (Only on main process)
            if self.use_ema:
                # print(f"Initializing EMA model at step {self.global_step}")
                # Use your manual peeling logic here to get the clean model
                raw_model = self._get_raw_model()
                self.ema_model = EMA(raw_model, self.ema_rate)
                self.ema_model.to(self.accelerator.device)
                self.accelerator.print(f"EMA initialized with rate {self.ema_rate}")

    @staticmethod
    def _env_bool(name, default):
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name, default):
        value = os.environ.get(name)
        if value is None or value == "":
            return int(default)
        return int(value)

    def _get_raw_model(self):
        """Helper to peel off DDP and Compile wrappers manually."""
        m = self.model

        # 1. Peel off the Accelerator/Distributed wrapper (.module)
        if hasattr(m, "module"):
            m = m.module

        # 2. Peel off the torch.compile wrapper (_orig_mod)
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif "_orig_mod" in m.__dict__:
            m = m.__dict__["_orig_mod"]

        return m

    def _append_local_metrics(self, phase, values):
        if not self.local_metrics_jsonl or not self.accelerator.is_main_process:
            return
        row = {
            'phase': str(phase),
            'global_step': int(self.global_step),
        }
        row.update({
            str(key): float(value.detach().float().cpu()) if torch.is_tensor(value) else value
            for key, value in values.items()
        })
        with open(self.local_metrics_jsonl, 'a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')

    @contextmanager
    def _fixed_validation_rng(self, val_i, val_set_name):
        """Make probe validation noise/t/CFG fixed without perturbing train RNG."""
        if self.probe_validation_seed is None:
            yield
            return

        cpu_state = torch.random.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        split_offset = 0 if val_set_name == 'trainval' else 1_000_000
        seed = int(self.probe_validation_seed) + split_offset + int(val_i)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


    def load_ema_state_dict(self, ema_state_dict):
        """Load EMA weights from a checkpoint into the main-process EMA model."""
        if not self.use_ema:
            self.accelerator.print("Checkpoint has EMA weights, but EMA is disabled in config. Skipping EMA load.")
            return

        if ema_state_dict is None:
            self.accelerator.print("No EMA weights found in checkpoint. EMA will start from the resumed model weights.")
            return

        if not self.accelerator.is_main_process:
            return

        if self.ema_model is None:
            raw_model = self._get_raw_model()
            self.ema_model = EMA(raw_model, self.ema_rate)
            self.ema_model.to(self.accelerator.device)

        if isinstance(ema_state_dict, dict) and "shadow" in ema_state_dict:
            ema_state_dict = ema_state_dict["shadow"]

        self.ema_model.shadow.load_state_dict(ema_state_dict)
        self.ema_model.to(self.accelerator.device)
        self.accelerator.print("Loaded EMA weights from checkpoint")

    def train_step(self, batch):
        """Execute a single training step."""
        # self.model.train()

        with self.accelerator.accumulate(self.model):

            metrics_dict = {}

            # extract the inputs from the batch
            points = batch["points"]
            rgbs = batch["rgbs"]
            valid_point_masks = batch["valid_point_masks"]
            instance_ids = batch["instance_ids"]
            object_transforms = batch["object_transforms"]
            object_feats = batch["object_feats"]
            object_coords = batch["object_coords"]
            selected_indices = batch["selected_indices"]
            max_num_objects = batch["max_num_objects"]


            profile = self.profile_step_timing and torch.cuda.is_available()
            if profile:
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                opt_end = torch.cuda.Event(enable_timing=True)
                fwd_start.record()

            # forward pass
            results_dict = self.model(
                points=points,
                rgbs=rgbs,
                valid_point_masks=valid_point_masks,
                instance_ids=instance_ids,
                object_transforms=object_transforms,
                object_feats=object_feats,
                object_coords=object_coords,
                selected_indices=selected_indices,
                max_num_objects=max_num_objects,
            )

            if profile:
                fwd_end.record()

            loss_dict = results_dict["flow_loss_dict"]
            metrics_dict = results_dict["flow_metrics_dict"]

            loss = loss_dict["loss_flow_feat"]
            loss_dict["loss"] = loss

            self.accelerator.backward(loss)
            if profile:
                bwd_end.record()

            # Only clip gradients and when gradients are synced
            if self.accelerator.sync_gradients:
                self.gradient_clip(self.accelerator, self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if profile:
                opt_end.record()

            # Capture sync_gradients INSIDE the accumulate context before it resets
            did_sync = self.accelerator.sync_gradients

        if profile:
            torch.cuda.synchronize()
            profile_dict = {
                "trainer/forward_ms": fwd_start.elapsed_time(fwd_end),
                "trainer/backward_ms": fwd_end.elapsed_time(bwd_end),
                "trainer/clip_optim_zero_ms": bwd_end.elapsed_time(opt_end),
                "trainer/train_step_ms": fwd_start.elapsed_time(opt_end),
            }
            raw_model = self._get_raw_model()
            model_profile = getattr(raw_model, "_last_profile", None)
            if isinstance(model_profile, dict):
                profile_dict.update(model_profile)
            self._last_profile = profile_dict

        # --- EMA Update ---
        # Only update if gradients were synced (optimizer stepped) and we are on main process
        if did_sync and self.use_ema and self.accelerator.is_main_process:
            # Use the helper to get the raw model so keys match
            raw_model = self._get_raw_model()
            # print(f"Updating EMA model at step {self.global_step}")
            self.ema_model.update(raw_model)
        # ------------------

        return loss_dict, metrics_dict, did_sync

    def fit(self):
        """Main training loop based on max_steps (optimization steps, not micro-batches)."""
        self.model.train()
        train_iter = iter(self.train_loader)
        if self.profile_step_timing and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        # tqdm for progress tracking (only on main process)
        pbar = tqdm(
            initial=self.global_step,
            total=self.max_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )

        while self.global_step < self.max_steps:
            # Get next batch (restart iterator if exhausted)
            data_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            data_load_ms = (time.perf_counter() - data_start) * 1000.0

            if self._should_log_batch_for_step():
                self._log_batch_debug(batch, "before_train")

            # Train step
            try:
                loss_dict, metrics_dict, did_sync = self.train_step(batch)
                self._debug_cuda_sync_after_step(did_sync)
            except Exception as exc:
                self._log_batch_debug(batch, "train_exception")
                print(
                    f"[trainer-error] rank={self.accelerator.process_index} "
                    f"local_rank={self.accelerator.local_process_index} "
                    f"target_step={self.global_step + 1} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            loss = loss_dict["loss"]

            # pbar.set_description(f"Loss: {loss.item():.4f}")

            # Only log/validate/checkpoint on actual optimization steps
            if did_sync:
                self.global_step += 1
                pbar.update(1)

                # Log metrics
                if self.global_step % self.log_every_n_steps == 0:
                    log_dict = {f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                    log_dict.update({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in metrics_dict.items()})
                    log_dict["train/global_step"] = self.global_step
                    if self.profile_step_timing:
                        log_dict.update(self._last_profile)
                        log_dict["trainer/data_load_ms"] = data_load_ms
                        if torch.cuda.is_available():
                            log_dict["trainer/peak_memory_allocated_mib"] = (
                                torch.cuda.max_memory_allocated(self.accelerator.device) / (1024**2)
                            )
                            log_dict["trainer/peak_memory_reserved_mib"] = (
                                torch.cuda.max_memory_reserved(self.accelerator.device) / (1024**2)
                            )
                    self.accelerator.log(log_dict, step=self.global_step)
                    self._append_local_metrics('train', log_dict)

                # Save checkpoint. Avoid explicit NCCL barriers in normal training;
                # DDP will naturally synchronize ranks at the next gradient all-reduce.
                if self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                    if self.checkpoint_synchronize_cuda or self.checkpoint_barrier:
                        self._sync_before_checkpoint()
                    if self.accelerator.is_main_process:
                        self.save_checkpoint()
                    if self.checkpoint_barrier:
                        self.accelerator.wait_for_everyone()

                # Validation (ALL processes must run to keep prepared dataloaders in sync)
                if self.enable_validation and self.global_step > 0 and self.global_step % self.val_every_n_steps == 0:
                    self.accelerator.print(f"Validating trainval at step {self.global_step}")
                    self.validate(self.trainval_loader, self.trainval_num, "trainval")
                    self.accelerator.print(f"Validating val at step {self.global_step}")
                    self.validate(self.val_loader, self.val_num, "val")

                    # Sync all processes after validation
                    self.accelerator.wait_for_everyone()

                # if self.global_step > 0 and self.global_step % self.inference_every_n_steps == 0:
                #     self.accelerator.print(f"Inference trainval at step {self.global_step}")
                #     self.inference(self.trainval_loader, self.trainval_num, "trainval")
                #     self.accelerator.print(f"Inference val at step {self.global_step}")
                #     self.inference(self.val_loader, self.val_num, "val")
                #     self.accelerator.wait_for_everyone()

        pbar.close()

        # Save final checkpoint
        if self.checkpoint_synchronize_cuda or self.checkpoint_barrier:
            self._sync_before_checkpoint()
        if self.accelerator.is_main_process:
            self.save_checkpoint()
        if self.checkpoint_barrier:
            self.accelerator.wait_for_everyone()

        self.accelerator.print(f"Training complete! Total steps: {self.global_step}")

    def _should_log_batch_for_step(self):
        return (
            self.debug_log_batch_every_n_steps > 0
            and (self.global_step + 1) % self.debug_log_batch_every_n_steps == 0
        )

    def _shape_of(self, value):
        if torch.is_tensor(value):
            return tuple(value.shape)
        if isinstance(value, (list, tuple)):
            return f"list[{len(value)}]"
        return type(value).__name__

    def _log_batch_debug(self, batch, prefix):
        try:
            selected = batch.get("selected_indices")
            max_num_objects = batch.get("max_num_objects")
            summary = {
                "rank": self.accelerator.process_index,
                "local_rank": self.accelerator.local_process_index,
                "target_step": self.global_step + 1,
                "prefix": prefix,
                "data_name": batch.get("data_name"),
                "sample_type": batch.get("sample_type"),
                "num_objects": batch.get("num_objects"),
                "max_num_objects": max_num_objects,
                "points": self._shape_of(batch.get("points")),
                "rgbs": self._shape_of(batch.get("rgbs")),
                "valid_point_masks": self._shape_of(batch.get("valid_point_masks")),
                "instance_ids": self._shape_of(batch.get("instance_ids")),
                "object_feats": self._shape_of(batch.get("object_feats")),
                "object_coords": self._shape_of(batch.get("object_coords")),
                "object_transforms": self._shape_of(batch.get("object_transforms")),
                "selected_indices": self._shape_of(selected),
            }
            print(f"[batch-debug] {summary}", flush=True)
        except Exception as debug_exc:
            print(
                f"[batch-debug-failed] rank={self.accelerator.process_index} "
                f"target_step={self.global_step + 1} error={debug_exc}",
                flush=True,
            )

    def _debug_cuda_sync_after_step(self, did_sync):
        if (
            not did_sync
            or self.debug_cuda_sync_every_n_steps <= 0
            or not torch.cuda.is_available()
        ):
            return
        completed_step = self.global_step + 1
        if completed_step % self.debug_cuda_sync_every_n_steps != 0:
            return
        torch.cuda.synchronize()
        print(
            f"[cuda-debug-sync] rank={self.accelerator.process_index} "
            f"local_rank={self.accelerator.local_process_index} step={completed_step}",
            flush=True,
        )

    def _sync_before_checkpoint(self):
        """Optional debug sync before checkpointing; disabled for normal training."""
        if self.checkpoint_synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.checkpoint_barrier:
            self.accelerator.wait_for_everyone()

    def _to_cpu_state(self, state):
        if torch.is_tensor(state):
            return state.detach().cpu()
        if isinstance(state, dict):
            return {k: self._to_cpu_state(v) for k, v in state.items()}
        if isinstance(state, list):
            return [self._to_cpu_state(v) for v in state]
        if isinstance(state, tuple):
            return tuple(self._to_cpu_state(v) for v in state)
        return state

    def save_checkpoint(self):
        """Save model checkpoint."""
        m = self._get_raw_model()

        # Now m is your raw, original model
        checkpoint = {
            "model": m.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gradient_clipper": self.gradient_clip.state_dict(),
            "global_step": self.global_step,
        }

        # Save EMA
        if self.use_ema and self.ema_model is not None:
            # print(f"Saving EMA model at step {self.global_step}")
            checkpoint["ema_model"] = self.ema_model.state_dict()
        if self.checkpoint_to_cpu:
            checkpoint = self._to_cpu_state(checkpoint)

        save_path = os.path.join(self.checkpoint_dir, f"step_{self.global_step}.pt")

        # Use accelerator.save to handle multi-GPU sync during writing
        self.accelerator.save(checkpoint, save_path)
        del checkpoint
        gc.collect()
        self.accelerator.print(f"Saved checkpoint to {save_path}")



    @torch.no_grad()
    def val_step(self, batch, val_i, val_set_name='val'):
        """Execute a single validation step."""
        self.model.eval()

        metrics_dict = {}

        batch_size = batch["points"].shape[0]
        assert batch_size == 1, "Validation batch size must be 1"

        # extract the inputs from the batch
        points = batch["points"]
        rgbs = batch["rgbs"]
        valid_point_masks = batch["valid_point_masks"]
        instance_ids = batch["instance_ids"]
        object_transforms = batch["object_transforms"]
        object_feats = batch["object_feats"]
        object_coords = batch["object_coords"]
        selected_indices = batch["selected_indices"]
        max_num_objects = batch["max_num_objects"]


        # forward pass
        with self._fixed_validation_rng(val_i, val_set_name):
            results_dict = self.model(
                points=points,
                rgbs=rgbs,
                valid_point_masks=valid_point_masks,
                instance_ids=instance_ids,
                object_transforms=object_transforms,
                object_feats=object_feats,
                object_coords=object_coords,
                selected_indices=selected_indices,
                max_num_objects=max_num_objects,
                return_denoised_latents=not self.probe_loss_only_validation,
            )

        loss_dict = results_dict["flow_loss_dict"]
        metrics_dict = results_dict["flow_metrics_dict"]

        loss = loss_dict["loss_flow_feat"]
        loss_dict["loss"] = loss

        if not self.probe_loss_only_validation:
            feats_preds = results_dict["feat_preds"]
            shape_x2_target_feats = results_dict["shape_x2_target_feats"]
            feat_loss_dict = feat_loss(
                feat_preds=feats_preds,
                feat_gt=shape_x2_target_feats,
            )
            feat_loss_dict = {f"shape_x2_{k}": v for k, v in feat_loss_dict.items()}
            metrics_dict.update(feat_loss_dict)

        return loss_dict, metrics_dict

    @torch.no_grad()
    def inference_step(self, batch, val_i, val_set_name):
        """Execute a single validation step."""
        self.model.eval()

        batch_size = batch["points"].shape[0]
        assert batch_size == 1, "Validation batch size must be 1"

        # extract the inputs from the batch
        points = batch["points"]
        rgbs = batch["rgbs"]
        valid_point_masks = batch["valid_point_masks"]
        instance_ids = batch["instance_ids"]
        object_transforms = batch["object_transforms"]
        object_feats = batch["object_feats"]
        object_coords = batch["object_coords"]
        selected_indices = batch["selected_indices"]
        max_num_objects = batch["max_num_objects"]

        results_dict = self.model(
            points=points,
            rgbs=rgbs,
            valid_point_masks=valid_point_masks,
            instance_ids=instance_ids,
            object_transforms=object_transforms,
            object_feats=object_feats,
            object_coords=object_coords,
            selected_indices=selected_indices,
            max_num_objects=max_num_objects,
            return_denoised_latents=True,
        )

        gpu_idx = self.accelerator.process_index

        if val_set_name == "trainval":
            save_dir = os.path.join(self.trainval_dir, str(self.global_step), f"trainval_{val_i}_{gpu_idx}")
        else:
            save_dir = os.path.join(self.val_dir, str(self.global_step), f"val_{val_i}_{gpu_idx}")
        os.makedirs(save_dir, exist_ok=True)

        feat_preds = results_dict.get("feat_preds")
        if feat_preds is not None:
            self.accelerator.print(f"Saved online Shape VAE X2 inference tensors for {save_dir}")

    @torch.no_grad()
    def validate(self, selected_dataloader, val_num, val_set_name):
        """Run validation on a fixed number of scenes and log metrics.

        IMPORTANT: This must run on ALL processes to keep prepared dataloaders in sync.
        Only main process does logging/visualization, but all processes iterate the dataloader.
        """
        self.model.eval()
        val_iter = iter(selected_dataloader)

        # Accumulate losses over validation batches
        accumulated_loss_dict = {}
        accumulated_metrics_dict = {}
        num_batches = 0

        pbar = tqdm(
            range(val_num),
            desc=f"{val_set_name} Validation",
            disable=not self.accelerator.is_local_main_process,
        )
        for val_i in pbar:
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            # All processes run val_step to keep model/dataloader state in sync
            loss_dict, metrics_dict = self.val_step(batch, val_i, val_set_name)

            # Only accumulate metrics on main process for logging
            if self.accelerator.is_main_process:
                # Accumulate losses
                for k, v in loss_dict.items():
                    if k not in accumulated_loss_dict:
                        accumulated_loss_dict[k] = 0.0
                    accumulated_loss_dict[k] += v.item() if torch.is_tensor(v) else v

                # Accumulate metrics
                for k, v in metrics_dict.items():
                    if k not in accumulated_metrics_dict:
                        accumulated_metrics_dict[k] = 0.0
                    accumulated_metrics_dict[k] += v.item() if torch.is_tensor(v) else v

                num_batches += 1

        # Average and log only on main process
        if self.accelerator.is_main_process:
            if num_batches > 0:
                for k in accumulated_loss_dict:
                    accumulated_loss_dict[k] /= num_batches
                for k in accumulated_metrics_dict:
                    accumulated_metrics_dict[k] /= num_batches

            # Log validation metrics
            log_dict = {f"{val_set_name}/{k}": v for k, v in accumulated_loss_dict.items()}
            log_dict.update({f"{val_set_name}/{k}": v for k, v in accumulated_metrics_dict.items()})
            log_dict[f"{val_set_name}/global_step"] = self.global_step
            self.accelerator.log(log_dict, step=self.global_step)
            self._append_local_metrics(val_set_name, log_dict)

            self.accelerator.print(f"{val_set_name} Validation at step {self.global_step}: loss={accumulated_loss_dict.get('loss', 0):.4f}")

        # Set model back to train mode
        self.model.train()

    @torch.no_grad()
    def inference(self, selected_dataloader, val_num, val_set_name):
        """Run validation on a fixed number of scenes and log metrics.

        IMPORTANT: This must run on ALL processes to keep prepared dataloaders in sync.
        Only main process does logging/visualization, but all processes iterate the dataloader.
        """
        self.model.eval()
        val_iter = iter(selected_dataloader)

        # Accumulate losses over validation batches
        accumulated_loss_dict = {}
        accumulated_metrics_dict = {}
        num_batches = 0

        pbar = tqdm(
            range(val_num),
            desc=f"{val_set_name} Validation",
            disable=not self.accelerator.is_local_main_process,
        )
        for val_i in pbar:
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            # All processes run val_step to keep model/dataloader state in sync
            self.inference_step(batch, val_i, val_set_name)

        # Set model back to train mode
        self.model.train()


    def fit_with_timing_profile(self, steps=12, warmup=2, output_path=None):
        """Run a short synchronized timing profile without saving checkpoints."""
        self.model.train()
        train_iter = iter(self.train_loader)
        rows = []
        self.accelerator.print(
            f"Starting lightweight timing profile: steps={steps}, warmup={warmup}, "
            f"rank={self.accelerator.process_index}"
        )
        for profile_i in range(int(steps)):
            t0 = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            data_load_ms = (time.perf_counter() - t0) * 1000.0
            self._last_data_load_ms = data_load_ms

            loss_dict, metrics_dict, did_sync = self.train_step(batch)
            if did_sync:
                self.global_step += 1

            row = {
                "rank": int(self.accelerator.process_index),
                "profile_step": int(profile_i),
                "global_step": int(self.global_step),
                "sample_type": batch.get("sample_type"),
                "num_objects": int(batch.get("num_objects", -1)),
                "data_load_ms": float(data_load_ms),
                "loss": float(loss_dict["loss"].detach().float().cpu()),
            }
            row.update({k: float(v) for k, v in self._last_profile.items()})
            if profile_i >= int(warmup):
                rows.append(row)
            if self.accelerator.is_main_process:
                keep_keys = [
                    "profile_step",
                    "sample_type",
                    "num_objects",
                    "data_load_ms",
                    "trainer/train_step_ms",
                    "trainer/forward_ms",
                    "trainer/backward_ms",
                    "trainer/clip_optim_zero_ms",
                    "model/dino_or_colors_ms",
                    "model/sample_object_feats_ms",
                    "model/context_transform_ms",
                    "model/context_encoder_ms",
                    "model/cond_pack_ms",
                    "model/shape_x2_encode_ms",
                    "model/flow_training_ms",
                ]
                summary = {k: row[k] for k in keep_keys if k in row}
                self.accelerator.print("[timing-profile] " + json.dumps(summary, sort_keys=True))

        if self.accelerator.is_main_process and rows:
            numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
            averages = {}
            for key in numeric_keys:
                values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
                if values:
                    averages[key] = sum(values) / len(values)
            result = {"steps": rows, "averages_after_warmup": averages}
            if output_path is None:
                output_path = os.environ.get("FF_PROFILE_OUTPUT", "logs/shape_x2_online_timing_profile.json")
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            self.accelerator.print("[timing-profile-summary] " + json.dumps(averages, sort_keys=True))
            self.accelerator.print(f"Saved lightweight timing profile to {output_path}")

    def fit_with_profiling(self, profiler_log_dir='./logs/profiler', wait=1, warmup=2, active=2, repeat=1):
        """Main training loop with torch.profiler for performance analysis.

        Args:
            profiler_log_dir: Directory to save profiler traces for TensorBoard
            wait: Number of steps to wait before starting profiling
            warmup: Number of warmup steps before active profiling
            active: Number of steps to actively profile
            repeat: Number of times to repeat the wait/warmup/active cycle

        Note:
            To fully profile __getitem__ and collate_fn in data loading,
            set num_workers=0 in your DataLoader config during profiling.
            With num_workers > 0, only the main process is profiled.
        """
        self.model.train()
        train_iter = iter(self.train_loader)

        # Calculate total profiling steps
        total_profile_steps = (wait + warmup + active) * repeat

        # Create profiler log directory
        if self.accelerator.is_main_process:
            os.makedirs(profiler_log_dir, exist_ok=True)

        self.accelerator.print(f"Starting profiling for {total_profile_steps} steps...")
        self.accelerator.print(f"Schedule: wait={wait}, warmup={warmup}, active={active}, repeat={repeat}")


        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            pbar = tqdm(
                range(total_profile_steps),
                disable=not self.accelerator.is_local_main_process,
                desc="Profiling"
            )

            for step in pbar:
                # Get next batch (restart iterator if exhausted)
                # Use record_function to annotate data loading in the trace
                with torch.profiler.record_function("data_loading"):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)

                # Train step (forward, backward, optimizer)
                with torch.profiler.record_function("train_step"):
                    loss_dict, metrics_dict, did_sync = self.train_step(batch)
                    # loss = loss_dict["loss"]

                # pbar.set_description(f"Loss: {loss.item():.4f}")

                # Step the profiler (steps on every micro-batch for profiling purposes)
                prof.step()

                # Only increment global_step on actual optimization steps
                if did_sync:
                    self.global_step += 1

        self.accelerator.print(f"Profiling complete! Traces saved to {profiler_log_dir}")
        self.accelerator.print("View with: tensorboard --logdir=" + profiler_log_dir)
