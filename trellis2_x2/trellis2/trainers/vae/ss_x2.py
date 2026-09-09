"""Joint encoder-decoder training for the Fire3D sparse-structure VAE."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...utils.data_utils import recursive_to_device
from ..basic import BasicTrainer


class SSX2Trainer(BasicTrainer):
    """Train the SS-VAE encoder and decoder with reconstruction and KL loss."""

    def __init__(
        self,
        *args: Any,
        loss_type: str = "bce",
        lambda_kl: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if loss_type != "bce":
            raise ValueError("The released SS-VAE recipe supports BCE reconstruction loss")
        self.loss_type = loss_type
        self.lambda_kl = float(lambda_kl)
        super().__init__(*args, **kwargs)
        if set(self.models) != {"encoder", "decoder"}:
            raise ValueError(
                "SS-VAE training requires exactly the encoder and decoder models, "
                f"got {set(self.models)}"
            )

    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del kwargs
        target = ss.float()
        latent, mean, logvar = self.training_models["encoder"](
            target,
            sample_posterior=True,
            return_raw=True,
        )
        logits = self.training_models["decoder"](latent)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        kl = 0.5 * torch.mean(mean.square() + logvar.exp() - logvar - 1)
        loss = bce + self.lambda_kl * kl

        with torch.no_grad():
            prediction = logits.sigmoid() >= 0.5
            occupied = target.bool()
            intersection = (prediction & occupied).flatten(1).sum(1).float()
            union = (prediction | occupied).flatten(1).sum(1).float()
            pred_count = prediction.flatten(1).sum(1).float()
            gt_count = occupied.flatten(1).sum(1).float()

        terms = {"loss": loss, "bce": bce, "kl": kl}
        status = {
            "iou_05": (intersection / union.clamp_min(1)).mean(),
            "dice_05": (2 * intersection / (pred_count + gt_count).clamp_min(1)).mean(),
            "exact_support_05": (prediction == occupied).flatten(1).all(1).float().mean(),
            "occupied_count_bias_05": (pred_count - gt_count).mean(),
            "latent_mean_abs": mean.detach().abs().mean(),
            "latent_logvar_mean": logvar.detach().mean(),
        }
        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int = 4,
        verbose: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | str]]:
        del verbose
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        iterator = iter(dataloader)
        training_states = {name: model.training for name, model in self.models.items()}
        for model in self.models.values():
            model.eval()

        targets: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        remaining = min(int(num_samples), len(self.dataset))
        while remaining > 0:
            data = recursive_to_device(next(iterator), self.device)
            target = data["ss"][: min(remaining, batch_size)].float()
            latent = self.models["encoder"](target, sample_posterior=False)
            reconstruction = self.models["decoder"](latent).sigmoid()
            targets.append(self._orthographic_strip(target))
            reconstructions.append(self._orthographic_strip(reconstruction))
            remaining -= len(target)

        for name, model in self.models.items():
            model.train(training_states[name])
        return {
            "target_projections": {"value": torch.cat(targets), "type": "image"},
            "reconstruction_projections": {
                "value": torch.cat(reconstructions),
                "type": "image",
            },
        }

    @staticmethod
    def _orthographic_strip(occupancy: torch.Tensor) -> torch.Tensor:
        projections = (
            occupancy.amax(dim=-1),
            occupancy.amax(dim=-2),
            occupancy.amax(dim=-3),
        )
        return torch.cat(projections, dim=-1)
