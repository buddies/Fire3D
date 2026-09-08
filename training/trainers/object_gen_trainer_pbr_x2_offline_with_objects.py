"""Modern trainer adapter for object-only LC64 PBR-X2 sparse flow."""

from __future__ import annotations

import torch

from training.trainers.object_gen_trainer_shape_x2_online_with_objects import Trainer as ShapeTrainer
from utils.loss import feat_loss


class Trainer(ShapeTrainer):
    """Reuse modern checkpoint/DDP/EMA loops with paired PBR batch keys."""

    def _model_kwargs(self, batch):
        return {
            "points": batch["points"],
            "rgbs": batch["rgbs"],
            "valid_point_masks": batch["valid_point_masks"],
            "instance_ids": batch["instance_ids"],
            "object_transforms": batch["object_transforms"],
            "object_shape_x2_feats": batch["object_shape_x2_feats"],
            "object_pbr_x2_feats": batch.get("object_pbr_x2_feats"),
            "object_coords": batch["object_coords"],
            "selected_indices": batch["selected_indices"],
            "max_num_objects": batch["max_num_objects"],
        }

    def train_step(self, batch):
        with self.accelerator.accumulate(self.model):
            profile = self.profile_step_timing and torch.cuda.is_available()
            if profile:
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                opt_end = torch.cuda.Event(enable_timing=True)
                fwd_start.record()

            results = self.model(**self._model_kwargs(batch))
            if profile:
                fwd_end.record()
            loss_dict = results["flow_loss_dict"]
            metrics_dict = results["flow_metrics_dict"]
            loss = loss_dict["loss_flow_pbr_x2"]
            loss_dict["loss"] = loss
            self.accelerator.backward(loss)
            if profile:
                bwd_end.record()

            if self.accelerator.sync_gradients:
                # The Shape trainer historically called this twice.  PBR-X2
                # intentionally performs exactly one clipping pass per update.
                self.gradient_clip(self.accelerator, self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if profile:
                opt_end.record()
            did_sync = self.accelerator.sync_gradients

        if profile:
            torch.cuda.synchronize()
            self._last_profile = {
                "trainer/forward_ms": fwd_start.elapsed_time(fwd_end),
                "trainer/backward_ms": fwd_end.elapsed_time(bwd_end),
                "trainer/clip_optim_zero_ms": bwd_end.elapsed_time(opt_end),
                "trainer/train_step_ms": fwd_start.elapsed_time(opt_end),
            }
            raw_model = self._get_raw_model()
            model_profile = getattr(raw_model, "_last_profile", None)
            if isinstance(model_profile, dict):
                self._last_profile.update(model_profile)

        if did_sync and self.use_ema and self.accelerator.is_main_process:
            self.ema_model.update(self._get_raw_model())
        return loss_dict, metrics_dict, did_sync

    @torch.no_grad()
    def val_step(self, batch, val_i, val_set_name="val"):
        self.model.eval()
        with self._fixed_validation_rng(val_i, val_set_name):
            results = self.model(
                **self._model_kwargs(batch),
                return_denoised_latents=not self.probe_loss_only_validation,
            )
        loss_dict = results["flow_loss_dict"]
        metrics_dict = results["flow_metrics_dict"]
        loss_dict["loss"] = loss_dict["loss_flow_pbr_x2"]
        if not self.probe_loss_only_validation:
            latent_metrics = feat_loss(
                feat_preds=results["pbr_x2_pred_feats"],
                feat_gt=results["pbr_x2_target_feats"],
            )
            metrics_dict.update({f"pbr_x2_{key}": value for key, value in latent_metrics.items()})
        return loss_dict, metrics_dict

    @torch.no_grad()
    def inference_step(self, batch, val_i, val_set_name):
        del val_i, val_set_name
        self.model.eval()
        return self.model(
            **self._model_kwargs(batch),
            return_denoised_latents=True,
        )

    def _log_batch_debug(self, batch, prefix):
        try:
            summary = {
                "rank": self.accelerator.process_index,
                "local_rank": self.accelerator.local_process_index,
                "target_step": self.global_step + 1,
                "prefix": prefix,
                "data_name": batch.get("data_name"),
                "num_objects": batch.get("num_objects"),
                "max_num_objects": batch.get("max_num_objects"),
                "points": self._shape_of(batch.get("points")),
                "rgbs": self._shape_of(batch.get("rgbs")),
                "valid_point_masks": self._shape_of(batch.get("valid_point_masks")),
                "shape_x2": self._shape_of(batch.get("object_shape_x2_feats")),
                "pbr_x2": self._shape_of(batch.get("object_pbr_x2_feats")),
                "coords": self._shape_of(batch.get("object_coords")),
            }
            print(f"[pbr-x2-batch-debug] {summary}", flush=True)
        except Exception as exc:
            print(f"[pbr-x2-batch-debug-failed] {type(exc).__name__}: {exc}", flush=True)
