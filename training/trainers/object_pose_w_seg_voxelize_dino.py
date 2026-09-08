import os
import torch
from tqdm.auto import tqdm
import numpy as np
import torch
from utils.loss import pose_l1_loss, dice_loss_matched
from utils.ema import EMA

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, trainval_loader, val_loader, accelerator, config, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.trainval_loader = trainval_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.config = config
        self.loss_fn = loss_fn
        self.global_step = 0

        # Training config
        self.max_steps = config['training']['max_steps']
        self.save_every_n_steps = config['training'].get('save_every_n_steps', 5000)
        self.log_every_n_steps = config['training'].get('log_every_n_steps', 10)
        self.val_every_n_steps = config['training'].get('val_every_n_steps', 1000)
        self.inference_every_n_steps = config['training'].get('inference_every_n_steps', 1000)
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
        self.val_dir = config['training'].get('val_dir', 'val')
        self.val_num = config['training'].get('val_num', 1)
        self.trainval_dir = config['training'].get('trainval_dir', 'trainval')
        self.trainval_num = config['training'].get('trainval_num', 1)
        self.warmup_steps = config['training'].get('warmup_steps', 0)
        # Preserve the historical sum-of-all-decoder-losses behavior unless a
        # config explicitly opts into depth-invariant auxiliary supervision.
        aux_loss_weight = config['training'].get('aux_loss_weight', None)
        self.aux_loss_weight = (
            None if aux_loss_weight is None else float(aux_loss_weight)
        )
        augmentation_schedule = config.get('data', {}).get('augmentation_schedule', {})
        self.augmentation_clean_steps = int(
            augmentation_schedule.get('clean_steps', self.warmup_steps)
        )
        self.augmentation_full_step = int(
            augmentation_schedule.get('full_intensity_step', self.augmentation_clean_steps)
        )
        self.augmentation_curve = str(augmentation_schedule.get('curve', 'linear'))
        self.instance_augment_start_step = int(
            config.get('data', {}).get('instance_augment', {}).get('start_step', 0)
        )
        self.use_ema = config['training'].get('use_ema', False)
        self.ema_rate = config['training'].get('ema_rate', 0.9999)
        self.ema_model = None
        self.abort_on_nonfinite_loss = bool(
            config['training'].get('abort_on_nonfinite_loss', True)
        )
        self.abort_on_nonfinite_grad = bool(
            config['training'].get('abort_on_nonfinite_grad', True)
        )

        # Create checkpoint directory
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.val_dir, exist_ok=True)
            os.makedirs(self.trainval_dir, exist_ok=True)

            # Initialize EMA (Only on main process)
            if self.use_ema:
                # print(f"Initializing EMA model at step {self.global_step}")
                # Use your manual peeling logic here to get the clean model
                raw_model = self._get_raw_model()
                self.ema_model = EMA(raw_model, self.ema_rate)
                self.ema_model.to(self.accelerator.device)
                self.accelerator.print(f"EMA initialized with rate {self.ema_rate}")

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

    def train_step(self, batch):
        """Execute a single training step."""
        self.model.train()

        with self.accelerator.accumulate(self.model):

            metrics_dict = {}

            # extract the inputs from the batch
            points = batch["points"]
            rgbs = batch["rgbs"]
            instance_ids = batch["instance_ids"]
            valid_point_masks = batch.get("valid_point_masks", None)
            point_valid_masks = batch.get("point_valid_masks", None)
            point_source_indices = batch.get("point_source_indices", None)
            object_translations = batch["object_translations"]
            object_angles = batch["object_angles"]
            object_scales = batch["object_scales"]
            object_valid_masks = batch["object_valid_masks"]
            max_num_objects = batch["max_num_objects"]

            # forward pass
            results, gt_dict = self.model(
                points=points,
                rgbs=rgbs,
                instance_ids=instance_ids,
                valid_point_masks=valid_point_masks,
                point_valid_masks=point_valid_masks,
                point_source_indices=point_source_indices,
                object_translations=object_translations,
                object_angles=object_angles,
                object_scales=object_scales,
                object_valid_masks=object_valid_masks,
                max_num_objects=max_num_objects,
            )

            gt_pos_bins = gt_dict["gt_pos_bins"].long()
            gt_angle_bins = gt_dict["gt_angle_bins"].long()
            gt_scale_bins = gt_dict["gt_scale_bins"].long()
            gt_masks = gt_dict["gt_masks"]
            background_gt_mask = gt_dict.get("background_gt_mask")
            loss_object_valid_masks = gt_dict.get(
                "object_valid_masks", object_valid_masks
            )
            loss_max_num_objects = gt_dict.get(
                "max_num_objects", max_num_objects
            )

            num_decoder_layers = len(results)

            all_layer_loss_dict = {}
            last_layer_loss_dict = None

            # Only compute per-token losses occasionally to reduce overhead
            should_log_details = (self.global_step % self.log_every_n_steps == 0)

            layer_losses = []
            for layer_i in range(num_decoder_layers):
                layer_result = results[layer_i]
                pos_bin_logits = layer_result["pos_bin_logits"]
                angle_bin_logits = layer_result["angle_bin_logits"]
                scale_bin_logits = layer_result["scale_bin_logits"]
                valid_logits = layer_result["valid_logits"]
                pred_masks_logits = layer_result["pred_masks_logits"]
                background_pred_masks_logits = layer_result.get(
                    "background_pred_masks_logits"
                )

                # Only compute per-token losses for the last layer (others just need the main loss)
                is_last_layer = (layer_i == num_decoder_layers - 1)

                # compute loss
                loss_dict = self.loss_fn(
                    pos_bin_logits=pos_bin_logits,
                    angle_bin_logits=angle_bin_logits,
                    scale_bin_logits=scale_bin_logits,
                    valid_logits=valid_logits,
                    pred_masks_logits=pred_masks_logits,
                    background_pred_masks_logits=
                        background_pred_masks_logits,
                    gt_pos_bins=gt_pos_bins,
                    gt_angle_bins=gt_angle_bins,
                    gt_scale_bins=gt_scale_bins,
                    gt_masks=gt_masks,
                    background_gt_mask=background_gt_mask,
                    max_num_objects=loss_max_num_objects,
                    object_valid_masks=loss_object_valid_masks,
                    include_background_segmentation=is_last_layer,
                    return_per_token_losses=(is_last_layer and should_log_details),
                )

                layer_loss = loss_dict["loss"]
                layer_losses.append(layer_loss)

                all_layer_loss_dict[f"layer_{layer_i}_loss"] = layer_loss
                if is_last_layer:
                    last_layer_loss_dict = loss_dict

            assert last_layer_loss_dict is not None, "Last layer loss dictionary is None"
            loss_dict = last_layer_loss_dict
            loss_dict.update(all_layer_loss_dict)
            if self.aux_loss_weight is None:
                # Backward-compatible path used by existing configs.
                loss = torch.stack(layer_losses).sum()
            else:
                final_loss = layer_losses[-1]
                if len(layer_losses) > 1:
                    aux_loss_mean = torch.stack(layer_losses[:-1]).mean()
                else:
                    aux_loss_mean = final_loss.new_zeros(())
                loss = final_loss + self.aux_loss_weight * aux_loss_mean
                loss_dict["final_layer_loss"] = final_loss
                loss_dict["aux_loss_mean"] = aux_loss_mean
                loss_dict["aux_loss_weight"] = self.aux_loss_weight

            loss_dict["total_loss"] = loss

            # Every rank participates in this reduction. If any rank sees a
            # non-finite loss, all ranks abort before backward so Adam state
            # cannot be contaminated; the watchdog resumes the last atomic
            # checkpoint.
            loss_finite = torch.isfinite(loss.detach()).to(
                device=loss.device, dtype=torch.int32
            )
            loss_finite = self.accelerator.reduce(loss_finite, reduction="min")
            if self.abort_on_nonfinite_loss and not bool(loss_finite.item()):
                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.print(
                    f"Non-finite total loss at global_step={self.global_step}; "
                    "aborting safely"
                )
                raise FloatingPointError("Non-finite det/seg total loss")

            if self.global_step % self.log_every_n_steps == 0:
                layer_i = num_decoder_layers - 1
                layer_result = results[layer_i]
                pos_bin_logits = layer_result["pos_bin_logits"]
                angle_bin_logits = layer_result["angle_bin_logits"]
                scale_bin_logits = layer_result["scale_bin_logits"]
                pose_l1_loss_dict = pose_l1_loss(
                    pos_bin_logits=pos_bin_logits,
                    angle_bin_logits=angle_bin_logits,
                    scale_bin_logits=scale_bin_logits,
                    tgt_pos_bins=gt_pos_bins,
                    tgt_angle_bins=gt_angle_bins,
                    tgt_scale_bins=gt_scale_bins,
                    max_num_objects=loss_max_num_objects,
                    object_valid_masks=loss_object_valid_masks,
                    rotation_z_quarter_turns=self.loss_fn.rotation_z_quarter_turns,
                )
                metrics_dict.update(pose_l1_loss_dict)

            self.accelerator.backward(loss)

            # Only clip gradients when gradients are synced
            if self.accelerator.sync_gradients:
                raw_model = self._get_raw_model()
                background_seg_feat = getattr(
                    getattr(raw_model, "scene_decoder", None),
                    "background_seg_feat",
                    None,
                )
                if (
                    background_seg_feat is not None
                    and background_seg_feat.grad is not None
                ):
                    loss_dict["background_seg_feat_grad_norm"] = (
                        background_seg_feat.grad.detach().float().norm()
                    )
                # # debug: check grad norm of each parameter
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter: {name}, Grad Norm: {param.grad.norm()}")
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                grad_finite = torch.isfinite(grad_norm.detach()).to(
                    device=grad_norm.device, dtype=torch.int32
                )
                grad_finite = self.accelerator.reduce(
                    grad_finite, reduction="min"
                )
                if self.abort_on_nonfinite_grad and not bool(grad_finite.item()):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.print(
                        f"Non-finite gradient norm at global_step={self.global_step}; "
                        "aborting before optimizer.step()"
                    )
                    raise FloatingPointError("Non-finite det/seg gradient norm")
                loss_dict["grad_norm"] = grad_norm.detach().float()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Capture sync_gradients INSIDE the accumulate context before it resets
            did_sync = self.accelerator.sync_gradients

        # --- EMA Update ---
        # Only update if gradients were synced (optimizer stepped) and we are on main process
        if did_sync and self.use_ema and self.accelerator.is_main_process:
            # Use the helper to get the raw model so keys match
            raw_model = self._get_raw_model()
            # print(f"Updating EMA model at step {self.global_step}")
            self.ema_model.update(raw_model)
        # ------------------

        return loss_dict, metrics_dict, did_sync

    def _augmentation_intensity(self):
        """Smoothly ramp all train-time corruption from clean to full strength."""
        if self.global_step < self.augmentation_clean_steps:
            return 0.0
        if self.augmentation_full_step <= self.augmentation_clean_steps:
            return 1.0
        u = np.clip(
            (self.global_step - self.augmentation_clean_steps)
            / float(self.augmentation_full_step - self.augmentation_clean_steps),
            0.0,
            1.0,
        )
        if self.augmentation_curve == 'smoothstep':
            return float(u * u * (3.0 - 2.0 * u))
        if self.augmentation_curve != 'linear':
            raise ValueError(f"Unknown augmentation schedule curve: {self.augmentation_curve}")
        return float(u)

    def _move_batch_to_device(self, value):
        """Recursively place an unprepared validation batch on this rank."""

        if torch.is_tensor(value):
            return value.to(self.accelerator.device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: self._move_batch_to_device(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._move_batch_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_batch_to_device(item) for item in value)
        return value

    @torch.no_grad()
    def val_step(self, batch, val_i):
        """Execute a single validation step."""
        self.model.eval()
        batch = self._move_batch_to_device(batch)

        metrics_dict = {}

        batch_size = batch["points"].shape[0]
        assert batch_size == 1, "Validation batch size must be 1"

        # extract the inputs from the batch
        points = batch["points"]
        rgbs = batch["rgbs"]
        instance_ids = batch["instance_ids"]
        valid_point_masks = batch.get("valid_point_masks", None)
        point_valid_masks = batch.get("point_valid_masks", None)
        point_source_indices = batch.get("point_source_indices", None)
        object_translations = batch["object_translations"]
        object_angles = batch["object_angles"]
        object_scales = batch["object_scales"]
        object_valid_masks = batch["object_valid_masks"]
        max_num_objects = batch["max_num_objects"]

        # forward pass
        results, gt_dict = self.model(
            points=points,
            rgbs=rgbs,
            instance_ids=instance_ids,
            valid_point_masks=valid_point_masks,
            point_valid_masks=point_valid_masks,
            point_source_indices=point_source_indices,
            object_translations=object_translations,
            object_angles=object_angles,
            object_scales=object_scales,
            object_valid_masks=object_valid_masks,
            max_num_objects=max_num_objects,
        )

        gt_pos_bins = gt_dict["gt_pos_bins"].long()
        gt_angle_bins = gt_dict["gt_angle_bins"].long()
        gt_scale_bins = gt_dict["gt_scale_bins"].long()
        gt_masks = gt_dict["gt_masks"]
        background_gt_mask = gt_dict.get("background_gt_mask")
        loss_object_valid_masks = gt_dict.get(
            "object_valid_masks", object_valid_masks
        )
        loss_max_num_objects = gt_dict.get(
            "max_num_objects", max_num_objects
        )

        num_decoder_layers = len(results)
        layer_result = results[num_decoder_layers - 1]

        pos_bin_logits = layer_result["pos_bin_logits"]
        angle_bin_logits = layer_result["angle_bin_logits"]
        scale_bin_logits = layer_result["scale_bin_logits"]
        valid_logits = layer_result["valid_logits"]
        pred_masks_logits = layer_result["pred_masks_logits"]
        background_pred_masks_logits = layer_result.get(
            "background_pred_masks_logits"
        )

        # compute loss
        loss_dict = self.loss_fn(
            pos_bin_logits=pos_bin_logits,
            angle_bin_logits=angle_bin_logits,
            scale_bin_logits=scale_bin_logits,
            valid_logits=valid_logits,
            pred_masks_logits=pred_masks_logits,
            background_pred_masks_logits=background_pred_masks_logits,
            gt_pos_bins=gt_pos_bins,
            gt_angle_bins=gt_angle_bins,
            gt_scale_bins=gt_scale_bins,
            gt_masks=gt_masks,
            background_gt_mask=background_gt_mask,
            max_num_objects=loss_max_num_objects,
            object_valid_masks=loss_object_valid_masks,
            return_per_token_losses=True,
        )

        pose_l1_loss_dict = pose_l1_loss(
            pos_bin_logits=pos_bin_logits,
            angle_bin_logits=angle_bin_logits,
            scale_bin_logits=scale_bin_logits,
            tgt_pos_bins=gt_pos_bins,
            tgt_angle_bins=gt_angle_bins,
            tgt_scale_bins=gt_scale_bins,
            max_num_objects=loss_max_num_objects,
            object_valid_masks=loss_object_valid_masks,
            rotation_z_quarter_turns=self.loss_fn.rotation_z_quarter_turns,
        )
        metrics_dict.update(pose_l1_loss_dict)

        return loss_dict, metrics_dict

    @torch.no_grad()
    def validate(self, selected_dataloader, val_num, val_set_name):
        """Run validation on a fixed number of scenes and log metrics.

        IMPORTANT: This must run on ALL processes to keep prepared dataloaders in sync.
        Only main process does logging/visualization, but all processes iterate the dataloader.
        """
        if selected_dataloader is None or int(val_num) <= 0:
            return
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
            loss_dict, metrics_dict = self.val_step(batch, val_i)

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

            self.accelerator.print(f"{val_set_name} Validation at step {self.global_step}: loss={accumulated_loss_dict.get('loss', 0):.4f}")

        # Set model back to train mode
        self.model.train()

    def fit(self):
        """Main training loop based on max_steps (optimization steps, not micro-batches)."""
        self.model.train()
        train_iter = iter(self.train_loader)

        # tqdm for progress tracking (only on main process)
        pbar = tqdm(
            initial=self.global_step,
            total=self.max_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )

        while self.global_step < self.max_steps:
            # Get next batch (restart iterator if exhausted)
            augmentation_intensity = self._augmentation_intensity()
            self.train_loader.dataset.augmentation_intensity = augmentation_intensity
            self.train_loader.dataset.enable_noise = augmentation_intensity > 0.0
            self.train_loader.dataset.enable_instance_augment = (
                augmentation_intensity > 0.0
                and self.global_step >= self.instance_augment_start_step
            )
            # print(f"Enable noise: {self.train_loader.dataset.enable_noise}; global step: {self.global_step}; warmup steps: {self.warmup_steps}")
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            loss_dict, metrics_dict, did_sync = self.train_step(batch)

            # loss = loss_dict["loss"]
            # pbar.set_description(f"Loss: {loss.item():.4f}")

            # Only log/validate/checkpoint/increment on actual optimization steps
            if did_sync:
                # Log metrics
                if self.global_step % self.log_every_n_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_dict = {f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                    log_dict.update({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in metrics_dict.items()})
                    log_dict["train/lr"] = current_lr
                    log_dict["train/global_step"] = self.global_step
                    log_dict["train/augmentation_intensity"] = augmentation_intensity
                    self.accelerator.log(log_dict, step=self.global_step)

                # Save checkpoint (only on main process, but all must wait)
                if self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                    if self.accelerator.is_main_process:
                        self.save_checkpoint()
                    # All processes must wait for checkpoint to complete to avoid deadlock
                    self.accelerator.wait_for_everyone()

                # Validation (ALL processes must run to keep prepared dataloaders in sync)
                if self.global_step > 0 and self.global_step % self.val_every_n_steps == 0:
                    if self.trainval_loader is not None and self.trainval_num > 0:
                        self.accelerator.print(f"Validating trainval at step {self.global_step}")
                        self.validate(self.trainval_loader, self.trainval_num, "trainval")
                    if self.val_loader is not None and self.val_num > 0:
                        val_set_name = self.config['training'].get(
                            'val_set_name', 'val'
                        )
                        self.accelerator.print(
                            f"Validating {val_set_name} at step {self.global_step}"
                        )
                        self.validate(self.val_loader, self.val_num, val_set_name)

                    # Sync all processes after validation
                    self.accelerator.wait_for_everyone()

                # if self.global_step > 0 and self.global_step % self.inference_every_n_steps == 0:
                #     self.accelerator.print(f"Inference trainval at step {self.global_step}")
                #     self.inference(self.trainval_loader, self.trainval_num, "trainval")
                #     self.accelerator.print(f"Inference val at step {self.global_step}")
                #     self.inference(self.val_loader, self.val_num, "val")
                #     self.accelerator.wait_for_everyone()



                self.global_step += 1
                pbar.update(1)

        pbar.close()

        # Save final checkpoint unless an explicit smoke/diagnostic run opts out.
        if (
            self.accelerator.is_main_process
            and self.config['training'].get('save_final_checkpoint', True)
        ):
            self.save_checkpoint()

        self.accelerator.print(f"Training complete! Total steps: {self.global_step}")

    def save_checkpoint(self):
        """Save model checkpoint."""
        m = self._get_raw_model()

        # Now m is your raw, original model
        checkpoint = {
            "model": m.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }

        # Save EMA
        if self.use_ema and self.ema_model is not None:
            # print(f"Saving EMA model at step {self.global_step}")
            checkpoint["ema_model"] = self.ema_model.state_dict()

        save_path = os.path.join(self.checkpoint_dir, f"step_{self.global_step}.pt")
        temporary_path = save_path + ".tmp"

        # Atomic rename keeps auto-resume from selecting a partially written
        # checkpoint after node/process failure.
        self.accelerator.save(checkpoint, temporary_path)
        os.replace(temporary_path, save_path)
        self.accelerator.print(f"Saved checkpoint to {save_path}")
