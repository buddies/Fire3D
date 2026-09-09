#!/usr/bin/env python3
"""Train the Fire3D sparse-structure VAE encoder and decoder jointly."""

from __future__ import annotations

from pathlib import Path

from training.hcvae.train import REPO_ROOT, run_training


DEFAULT_CONFIG = REPO_ROOT / "configs/training/ssvae/default.yaml"


def main() -> None:
    run_training(
        default_config=Path(DEFAULT_CONFIG),
        description="Train the Fire3D sparse-structure VAE encoder and decoder",
        heading="Fire3D Sparse-Structure VAE Training",
    )


if __name__ == "__main__":
    main()
