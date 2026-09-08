"""Resolve stable release or legacy training VAE checkpoint pairs."""

from __future__ import annotations

import json
import re
from pathlib import Path


def resolve_vae_checkpoints(
    root: str | Path,
    *,
    use_ema: bool = True,
    step: int | None = None,
) -> tuple[dict, Path, Path, str]:
    root = Path(root).expanduser().resolve()
    config_path = root / "config.json"
    checkpoint_dir = root / "ckpts"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VAE config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    stable_encoder = checkpoint_dir / "encoder.pt"
    stable_decoder = checkpoint_dir / "decoder.pt"
    if step is None and stable_encoder.is_file() and stable_decoder.is_file():
        return config, stable_encoder, stable_decoder, "stable release"

    family = "ema" if use_ema else "raw"
    if use_ema:
        ema_rate = config.get("trainer", {}).get("args", {}).get("ema_rate", 0.9999)
        encoder_pattern = re.compile(rf"encoder_ema{re.escape(str(ema_rate))}_step(\d+)\.pt")
        encoder_name = lambda value: f"encoder_ema{ema_rate}_step{value:07d}.pt"
        decoder_name = lambda value: f"decoder_ema{ema_rate}_step{value:07d}.pt"
    else:
        encoder_pattern = re.compile(r"encoder_step(\d+)\.pt")
        encoder_name = lambda value: f"encoder_step{value:07d}.pt"
        decoder_name = lambda value: f"decoder_step{value:07d}.pt"
    if step is None:
        candidates = [
            int(match.group(1))
            for path in checkpoint_dir.glob("encoder*.pt")
            if (match := encoder_pattern.fullmatch(path.name))
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No stable or {family} encoder checkpoint under {checkpoint_dir}"
            )
        step = max(candidates)
    encoder = checkpoint_dir / encoder_name(step)
    decoder = checkpoint_dir / decoder_name(step)
    missing = [str(path) for path in (encoder, decoder) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete VAE checkpoint pair: {missing}")
    return config, encoder, decoder, f"{family} training checkpoint {step}"
